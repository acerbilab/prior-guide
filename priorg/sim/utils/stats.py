import jax.numpy as jnp
import jax
import math
import numpy as np
import scipy.spatial.distance as distance


# Scipy stats implementation missing in JAX


def RMSE(gt, samples):
    """Calculate Root Mean Square Error.
    
    Args:
        gt: Ground truth values
        samples: Samples to evaluate
        
    Returns:
        Root mean squared error
    """
    gt = np.expand_dims(gt, axis=-1)
    gt = np.repeat(gt, samples.shape[-1], axis=-1)
    dist = np.sqrt(np.mean((gt - samples) ** 2))
    return dist


def kernel_matrix(X, Y, lengthscale):
    """Compute the RBF kernel matrix between X and Y.
    
    Args:
        X: First set of samples
        Y: Second set of samples
        lengthscale: RBF kernel lengthscale
        
    Returns:
        Kernel matrix
    """
    X_sqnorms = np.sum(np.square(X), axis=1)
    Y_sqnorms = np.sum(np.square(Y), axis=1)
    XY = np.dot(X, Y.transpose())
    sqD = -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
    return np.exp(-sqD / (2 * lengthscale ** 2))


def MMD_unweighted(x, y, lengthscale):
    """Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    
    Args:
        x: Samples from distribution P
        y: Samples from distribution Q
        lengthscale: RBF kernel lengthscale
        
    Returns:
        Maximum Mean Discrepancy value
    """
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    z = np.concatenate((x, y), axis=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * np.sum(kxx) - (2 / (m * n)) * np.sum(kxy) + (1 / n ** 2) * np.sum(kyy)


def median_heuristic(y):
    """Calculate median heuristic for kernel lengthscale estimation.
    
    Args:
        y: Sample data
        
    Returns:
        Estimated optimal lengthscale
    """
    a = distance.cdist(y, y, 'sqeuclidean')
    return np.sqrt(np.median(a / 2))









# Estimate the differential entropy of a continuous random variable.

def differential_entropy(values, window_length=None, base=None, axis=0, method="auto"):
    values = jnp.asarray(values)
    values = jnp.moveaxis(values, axis, -1)
    n = values.shape[-1]

    if window_length is None:
        window_length = jnp.floor(jnp.sqrt(n) + 0.5)

    if not 2 <= 2 * window_length < n:
        raise ValueError(
            f"Window length ({window_length}) must be positive and less "
            f"than half the sample size ({n})."
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None.")

    sorted_data = jnp.sort(values, axis=-1)

    method = method.lower()
    if method not in methods:
        message = f"`method` must be one of {set(methods)}"
        raise ValueError(message)

    if method == "auto":
        if n <= 10:
            method = "van es"
        elif n <= 1000:
            method = "ebrahimi"
        else:
            method = "vasicek"

    res = methods[method](sorted_data, window_length)

    if base is not None:
        res /= jnp.log(base)

    return res


def _pad_along_last_axis(X, m):
    shape = jnp.array(X.shape)
    shape = shape.at[-1].set(m)
    Xl = jnp.broadcast_to(X[..., 0:1], shape)
    Xr = jnp.broadcast_to(X[..., -1:], shape)
    return jnp.concatenate((Xl, X, Xr), axis=-1)


def _vasicek_entropy(X, m):
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m :] - X[..., : -2 * m :]
    logs = jnp.log(n / (2 * m) * differences)
    return jnp.mean(logs, axis=-1)


def _van_es_entropy(X, m):
    n = X.shape[-1]
    difference = X[..., m:] - X[..., :-m]
    term1 = 1 / (n - m) * jnp.sum(jnp.log((n + 1) / m * difference), axis=-1)
    k = jnp.arange(m, n + 1)
    return term1 + jnp.sum(1 / k) + jnp.log(m) - jnp.log(n + 1)


def _ebrahimi_entropy(X, m):
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m :] - X[..., : -2 * m :]
    i = jnp.arange(1, n + 1, dtype=jnp.float32)
    ci = jnp.where(i <= m, 1 + (i - 1) / m, 1 + (n - i) / m)
    logs = jnp.log(n * differences / (ci * m))
    return jnp.mean(logs, axis=-1)


def _correa_entropy(X, m):
    i = jnp.arange(1, n + 1, dtype=jnp.int32)
    dj = jnp.arange(-m, m + 1)[:, None]
    j = i + dj
    j0 = j + m - 1
    Xibar = jnp.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = jnp.sum(difference * dj, axis=-2)
    den = n * jnp.sum(difference**2, axis=-2)
    return -jnp.mean(jnp.log(num / den), axis=-1)


methods = {
    "vasicek": _vasicek_entropy,
    "van es": _van_es_entropy,
    "correa": _correa_entropy,
    "ebrahimi": _ebrahimi_entropy,
    "auto": _vasicek_entropy,
}
