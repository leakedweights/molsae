import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnums=(6, 7))
def generate_next_token(key, state, tokens, position, mask, forbidden_tokens=None, temperature=1.0, top_k=-1):
    logits = state.apply_fn({"params": state.params}, tokens, position, mask)
    logits = logits[:, -1, :]

    if top_k > 0:
        top_k_logits = jax.lax.top_k(logits, top_k)[0]
        min_top_k_logits = top_k_logits[:, -1]
        logits = jnp.where(logits < min_top_k_logits[:, None], -jnp.inf, logits)

    if forbidden_tokens is not None:
        forbidden_mask = jnp.isin(jnp.arange(logits.shape[-1]), forbidden_tokens)
        logits = jnp.where(forbidden_mask, -jnp.inf, logits)

    probs = jnp.exp(logits / temperature) / jnp.sum(jnp.exp(logits / temperature))
    next_token_id = jax.random.categorical(key, jnp.log(probs), axis=-1)

    return next_token_id, probs

def strip_molecules(input, bom_token, eom_token):
    substrs = input.split(eom_token)
    molecules = [substr.replace(bom_token, "") for substr in substrs
                 if bom_token in substr
                 and substr != bom_token
                 and substr != ""]
    
    return molecules