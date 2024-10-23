import jax
import jax.numpy as jnp

from functools import partial

from .model.transformer_utils import causal_mask


@partial(jax.jit, static_argnums=(1, 7, 8))
def generate_next_token(key, apply_fn, params, tokens, position, mask, forbidden_tokens=None, temperature=1.0, top_k=-1):
    logits = apply_fn({"params": params}, tokens, position, mask)
    logits = logits[:, -1, :]

    if top_k > 0:
        top_k_logits = jax.lax.top_k(logits, top_k)[0]
        min_top_k_logits = top_k_logits[:, -1]
        logits = jnp.where(
            logits < min_top_k_logits[:, None], -jnp.inf, logits)

    if forbidden_tokens is not None:
        forbidden_mask = jnp.isin(jnp.arange(
            logits.shape[-1]), forbidden_tokens)
        logits = jnp.where(forbidden_mask, -jnp.inf, logits)

    probs = jnp.exp(logits / temperature) / \
        jnp.sum(jnp.exp(logits / temperature))
    next_token_id = jax.random.categorical(key, jnp.log(probs), axis=-1)

    return next_token_id, probs


def strip_molecules(input, bom_token, eom_token):
    substrs = input.split(eom_token)
    molecules = [substr.replace(bom_token, "") for substr in substrs
                 if bom_token in substr
                 and substr != bom_token
                 and substr != ""]

    return molecules


@partial(jax.jit, static_argnums=(1, 2, 5, 6, 7))
def generate_sequence(key, apply_fn, params, initial_tokens, pad_token_id, max_length, forbidden_tokens=None, temperature=1.0, top_k=-1):
    batch_size, seq_length = initial_tokens.shape

    tokens = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    tokens = tokens.at[:, :seq_length].set(initial_tokens)
    positions = jnp.arange(max_length)[None, :]  # Shape (1, max_length)
    positions = jnp.broadcast_to(positions, (batch_size, max_length))

    mask = causal_mask(max_length, pad_token_id)
    mask = mask[None, :, :]  # Shape (1, max_length, max_length)

    # Initialize tokens_mask and attention_mask
    tokens_mask = (tokens != pad_token_id).astype(jnp.int32)
    attention_mask = attention_mask(tokens, pad_token_id)

    def generate_step(carry, step):
        key, tokens, tokens_mask, attention_mask = carry

        # Compute logits
        logits = apply_fn({"params": params}, tokens,
                          positions, attention_mask)
        logits = logits[:, step - 1, :]  # Get logits for current step

        # Apply top-k filtering
        if top_k > 0:
            top_k_logits = jax.lax.top_k(logits, top_k)[0]
            min_top_k_logits = top_k_logits[:, -1]
            logits = jnp.where(
                logits < min_top_k_logits[:, None], -jnp.inf, logits)

        # Apply forbidden tokens mask
        if forbidden_tokens is not None:
            forbidden_mask = jnp.isin(jnp.arange(
                logits.shape[-1]), forbidden_tokens)
            logits = jnp.where(forbidden_mask, -jnp.inf, logits)

        # Sample next token
        probs = jax.nn.softmax(logits / temperature)
        key, subkey = jax.random.split(key)
        next_token_id = jax.random.categorical(subkey, jnp.log(probs), axis=-1)

        # Update tokens
        tokens = tokens.at[:, step].set(next_token_id)
        tokens_mask = tokens_mask.at[:, step].set(1)

        # Update attention_mask efficiently
        new_attention = tokens_mask[:, None,
                                    step:step+1] * mask[:, step:step+1, :]
        attention_mask = attention_mask.at[:,
                                           step:step+1, :].set(new_attention)

        carry = (key, tokens, tokens_mask, attention_mask)
        return carry, next_token_id

    # Prepare the steps to loop over
    steps = jnp.arange(seq_length + 1, max_length + 1)
    initial_carry = (key, tokens, tokens_mask, attention_mask)

    # Run the generation loop
    carry, _ = jax.lax.scan(generate_step, initial_carry, steps)
    key, tokens, tokens_mask, attention_mask = carry

    return tokens
