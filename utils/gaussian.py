import torch

# Training samples
def generate_gaussian_data(d: int, Gamma: torch.Tensor, n_samples: int, scaling: bool = False):
    """
    Generates labeled training data for the density of a d-dimensional Gaussian.
    """
    
    x_samples = torch.rand(n_samples, d) * 6 - 3

    # Precompute constants for the Gaussian density function
    Gamma_inv = torch.inverse(Gamma)
    scaling_factor = (1-int(scaling)) + int(scaling) * torch.tensor(1) / torch.sqrt((2 * torch.pi) ** d * torch.det(Gamma_inv))
    
    # Compute the corresponding y = exp(x.T @ Gamma @ x) for each sample
    y_samples = torch.zeros(n_samples)

    for i in range(n_samples):
        x = x_samples[i]
        # Gaussian density function: scaling * exp(-0.5 * (x.T @ Gamma_inv @ x))
        exponent = -0.5 * (x.T @ Gamma @ x)
        y_samples[i] = scaling_factor * torch.exp(exponent)

    return x_samples, y_samples


# Density function
def gaussian_density(x: torch.Tensor, Gamma: torch.Tensor, scaling: bool = False) -> torch.Tensor:
    """
    Evaluates the density of a d-dimensional Gaussian on single or batched data points.
    """
    
    d = x.shape[-1]

    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add batch dimension, making x shape (1, d)

    # Precompute constants for the Gaussian density function
    Gamma_inv = torch.inverse(Gamma)
    scaling_factor = (1 - int(scaling)) + int(scaling) * torch.tensor(1) / torch.sqrt((2 * torch.pi) ** d * torch.det(Gamma_inv))

    # Compute the corresponding y = exp(x.T @ Gamma @ x) for each point in the batch
    exponent = -0.5 * torch.einsum('bi,ij,bj->b', x, Gamma, x)

    # Gaussian density function: scaling * exp(-0.5 * (x.T @ Gamma_inv @ x))
    y = scaling_factor * torch.exp(exponent)

    return y