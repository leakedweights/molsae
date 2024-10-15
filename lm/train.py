import jax
import optax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax.training import train_state
from functools import partial

from model.transformer_utils import causal_mask

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

def evaluate(state, get_eval_dataset, max_eval_batches=None):
    val_iter = iter(get_eval_dataset())
    total_eval_loss = 0.0
    num_eval_batches = 0
    while True:
        try:
            val_batch = next(val_iter)
        except StopIteration:
            break
        eval_loss = lm_eval_step(state, val_batch)
        total_eval_loss += eval_loss
        num_eval_batches += 1
        if max_eval_batches and num_eval_batches >= max_eval_batches:
            break

    avg_eval_loss = total_eval_loss / num_eval_batches

    return {"eval_loss": avg_eval_loss}


def save_checkpoint(dir, step, state):
    options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(
        dir,
        options=options,
    ) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))