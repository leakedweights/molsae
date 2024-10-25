import jax
import jax.numpy as jnp
import flax.linen as nn

"""JumpReLU utils from https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD"""

KDE_BANDWIDTH = 0.001


def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).astype(x.dtype)


@jax.custom_vjp
def step(x, threshold):
    return (x > threshold).astype(x.dtype)


def step_fwd(x, threshold):
    out = step(x, threshold)
    cache = x, threshold
    return out, cache


def step_bwd(cache, output_grad):
    x, threshold = cache
    x_grad = jnp.zeros_like(output_grad)
    threshold_grad = jnp.sum(
        -(1.0 / KDE_BANDWIDTH) *
        rectangle((x - threshold) / KDE_BANDWIDTH) * output_grad,
        axis=None,
    )
    threshold_grad = jnp.reshape(threshold_grad, threshold.shape)
    return x_grad, threshold_grad


step.defvjp(step_fwd, step_bwd)


@jax.custom_vjp
def jumprelu(x, threshold):
    return x * (x > threshold)


def jumprelu_fwd(x, threshold):
    out = jumprelu(x, threshold)
    cache = x, threshold
    return out, cache


def jumprelu_bwd(cache, output_grad):
    x, threshold = cache
    x_grad = (x > threshold).astype(x.dtype) * output_grad
    threshold_grad = jnp.sum(
        -(threshold / KDE_BANDWIDTH)
        * rectangle((x - threshold) / KDE_BANDWIDTH)
        * output_grad,
        axis=None,
    )
    threshold_grad = jnp.reshape(threshold_grad, threshold.shape)
    return x_grad, threshold_grad


jumprelu.defvjp(jumprelu_fwd, jumprelu_bwd)


class JSAE(nn.Module):
    """Flax module converted from https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD"""

    # TODO: fix theta application

    d_model: int
    hidden_size: int
    use_pre_enc_bias: bool = True

    @nn.compact
    def __call__(self, x):
        W_enc = self.param(
            'W_enc', nn.initializers.glorot_uniform(), (self.d_model, self.hidden_size))
        b_enc = self.param('b_enc', nn.initializers.zeros, (self.hidden_size,))
        W_dec = self.param(
            'W_dec', nn.initializers.glorot_uniform(), (self.hidden_size, self.d_model))
        b_dec = self.param('b_dec', nn.initializers.zeros, (self.d_model,))
        log_threshold = self.param(
            'log_threshold', nn.initializers.zeros, (self.hidden_size,))

        if self.use_pre_enc_bias:
            x = x - b_dec

        pre_activations = x @ W_enc + b_enc
        threshold = jnp.exp(log_threshold)
        feature_magnitudes = jumprelu(pre_activations, threshold)

        x_reconstructed = feature_magnitudes @ W_dec + b_dec
        return x_reconstructed, pre_activations


class RSAE(nn.Module):
    """ReLU SAE from https://transformer-circuits.pub/2023/monosemantic-features"""

    d_model: int
    hidden_size: int
    use_pre_enc_bias: bool = True

    @nn.compact
    def __call__(self, x):
        W_enc = self.param(
            'W_enc', nn.initializers.glorot_uniform(), (self.d_model, self.hidden_size))
        b_enc = self.param('b_enc', nn.initializers.zeros, (self.hidden_size,))
        W_dec = self.param(
            'W_dec', nn.initializers.glorot_uniform(), (self.hidden_size, self.d_model))
        b_dec = self.param('b_dec', nn.initializers.zeros, (self.d_model,))

        if self.use_pre_enc_bias:
            x = x - b_dec

        pre_activations = x @ W_enc + b_enc
        feature_magnitudes = nn.relu(pre_activations)

        x_reconstructed = feature_magnitudes @ W_dec + b_dec
        return x_reconstructed, feature_magnitudes
