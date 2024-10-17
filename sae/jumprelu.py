import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial

def step(x, theta):
        return (x > theta).astype(x.dtype)

def rect(z):
    return step(z, -0.5) - step(z, 0.5)

@jax.custom_vjp
def heaviside(x, theta, bandwidth):
    return (x > theta).astype(x.dtype)

def heaviside_fwd(x, theta, bandwidth):
    output = heaviside(x, theta, bandwidth)
    residuals = (x, theta, bandwidth)
    return output, residuals

def heaviside_bwd(residuals, tangents):
    x, theta, bandwidth = residuals

    output_grad = (-1 / bandwidth) * rect((x - theta) / bandwidth)
    theta_grad = tangents * jnp.sum(output_grad)

    x_grad = jnp.zeros_like(x)
    bandwidth_grad = jnp.zeros_like(bandwidth)
    return (x_grad, theta_grad, bandwidth_grad)

heaviside.defvjp(heaviside_fwd, heaviside_bwd)

@jax.custom_vjp
@partial(jax.vmap, in_axes=(0, None, None))
def jumprelu(x, theta, bandwidth):
    return x * heaviside(x, theta, bandwidth)

def jumprelu_fwd(x, theta, bandwidth):
    out = jumprelu(x, theta, bandwidth)
    residuals = (x, theta, bandwidth)
    return out, residuals

def jumprelu_bwd(residuals, tangents):
    x, theta, bandwidth = residuals
    x_grad = jnp.zeros_like(x)
    d_out_d_theta = - x / bandwidth * rect((x - theta) / bandwidth)
    threshold_grad = jnp.sum(d_out_d_theta * tangents, axis=0)
    bandwidth_grad = jnp.zeros_like(bandwidth)

    return x_grad, threshold_grad, bandwidth_grad

jumprelu.defvjp(jumprelu_fwd, jumprelu_bwd)

class SAE(nn.Module):
    latent_size: int
    kde_bandwidth: float = 1e-3

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]

        W_enc = self.param('W_enc', nn.initializers.he_uniform(), (input_dim, self.latent_size))
        b_enc = self.param('b_enc', nn.initializers.zeros, (self.latent_size,))

        z_enc = jnp.matmul(x, W_enc) + b_enc
        theta = self.param('theta', nn.initializers.he_uniform(), z_enc.shape)
        enc = jumprelu(z_enc, theta, self.kde_bandwidth)

        W_dec = self.param('W_dec', nn.initializers.he_uniform(), (self.latent_size, input_dim))
        b_dec = self.param('b_dec', nn.initializers.zeros, (input_dim,))

        dec = jnp.matmul(enc, W_dec) + b_dec

        return enc, dec
