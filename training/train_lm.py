import jax
import optax
from jax import random
import jax.numpy as jnp
from flax.training import train_state

import wandb
from tqdm import trange
from functools import partial

from lm.model.transformer_utils import causal_mask
from .train_utils import setup, try_restore_for, create_sharding, save_checkpoint

def create_lm_train_state(rng, model, learning_rate):
  params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32), jnp.ones((1, 1), dtype=jnp.int32),
                      jnp.ones((1, 1, 1), dtype=jnp.bool))["params"]
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=(2,))
def lm_train_step(state, batch, pad_token_id):
    def loss_fn(params):
        mask = causal_mask(batch, pad_token_id)
        positions = jnp.arange(0, batch.shape[1])
        logits = state.apply_fn({"params": params}, batch, positions, mask)

        logits = logits[:, :-1, :]
        labels = batch[:, 1:]

        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

        loss_mask = (labels != pad_token_id)
        loss = jnp.mean(per_token_loss * loss_mask)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state, eval_step, get_eval_dataset, max_eval_batches=None):
    val_iter = iter(get_eval_dataset())
    total_eval_loss = 0.0
    num_eval_batches = 0
    while True:
        try:
            val_batch = next(val_iter)
        except StopIteration:
            break
        eval_loss = eval_step(state, val_batch)
        total_eval_loss += eval_loss
        num_eval_batches += 1
        if max_eval_batches and num_eval_batches >= max_eval_batches:
            break

    avg_eval_loss = total_eval_loss / num_eval_batches

    return {"eval_loss": avg_eval_loss}

@partial(jax.jit, static_argnums=(2,))
def lm_eval_step(state, batch, pad_token_id):
    mask = causal_mask(batch, pad_token_id)
    positions = jnp.arange(0, batch.shape[1])
    logits = state.apply_fn({"params": state.params}, batch, positions, mask)

    logits = logits[:, :-1, :]
    labels = batch[:, 1:]

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    loss_mask = (labels != pad_token_id)
    loss = jnp.mean(per_token_loss * loss_mask)

    return loss

def train(model, train_ds, get_eval_ds, config, rng=random.key(0)):
    total_steps = config.get("num_steps")
    run_id = config.get("run_id", "default-run")
    checkpoint_base = config.get("ckpt_base_dir", "/tmp/checkpoints/")
    checkpoint_dir = f"{checkpoint_base}/{run_id}"

    setup(project_name=config.get("project_name", "MolSAE"),
          run_id=run_id,
          checkpoint_dir=checkpoint_dir,
          resume=config.get("resume", True))
    
    sharding = create_sharding()
    
    state = create_lm_train_state(rng, model, learning_rate=config.get("learning_rate"))
    state, train_step = try_restore_for(state, checkpoint_dir)

    with trange(train_step, total_steps, initial=train_step, total=total_steps) as steps:

        for step in steps:
            batch = jax.device_put(next(train_ds), sharding)
            state, loss = lm_train_step(state, batch)

            steps.set_postfix(loss=loss)

            if (step + 1) % 100 == 0:
                wandb.log({"train_loss": loss}, step=step + 1)

            if (step + 1) % 10_000 == 0:
                save_checkpoint(checkpoint_dir, step + 1, state)

            if step % 1000 == 0:
                eval_results = evaluate(state, get_eval_ds)
                wandb.log(eval_results, step=step)

    wandb.finish()