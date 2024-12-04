"""
This module contains the implementation of various basis functions, including
Legendre, B-Spline, Fourier, and Hermite bases, originally implemented in JAX by Charles Miranda
from Weierstrass Institut. The current version is a PyTorch adaptation, with modifications
for use in functional tensor trains.
"""

import torch
import math
import numpy as np
import numpy.polynomial.polyutils as pu

from abc import ABC, abstractmethod
from typing import Union, Tuple

# Function to compute Legendre polynomials using recurrence relation
# def legendre_recurrence(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#     """
#     Computes a set of Legendre polynomials up to the order specified by the shape of the 
#     coefficient matrix `c`, using a recurrence relation. Then multiplies these polynomials 
#     by the coefficient matrix `c` to return the result.

#     Parameters:
#     ----------
#     x : torch.Tensor
#         A tensor containing the input points where the Legendre polynomials will be evaluated. 
    
#     c : torch.Tensor
#         A square matrix of coefficients (shape n x n)

#     Returns:
#     -------
#     torch.Tensor
#         A tensor representing the result of the matrix multiplication between `c` and the 
#         computed Legendre polynomials. The shape of the output tensor depends on the shape 
#         of `x` and the degree of the polynomial (derived from `c`).

#     Notes:
#     ------
#     This function uses the recurrence relation for Legendre polynomials:
    
#     P_0(x) = 1
#     P_1(x) = x
#     P_n(x) = ((2n - 1) * x * P_(n-1)(x) - (n - 1) * P_(n-2)(x)) / n

#     It first computes all the Legendre polynomials up to the degree specified by `c`, then 
#     combines them with the coefficient matrix `c` via matrix multiplication.
#     """
    
#     # Assert that c is a square matrix
#     assert c.ndim == 2 and c.shape[0] == c.shape[1], "Matrix c must be square"
    
#     # Initialize the first two Legendre polynomials:
#     # P_0(x) = 1 and P_1(x) = x, where p_init stores these initial values
#     p_init = torch.zeros((2,) + x.shape, dtype=float, device=x.device)  # Initialize a tensor for P_0 and P_1
#     p_init[0] = 1.0  # P_0(x) = 1
#     p_init[1] = x    # P_1(x) = x

#     # Get the maximum degree for the polynomials (based on the size of matrix c)
#     n_max = c.shape[0] - 1

#     # Define the recurrence relation for generating Legendre polynomials
#     def body_fun(i, p_im1, p_i):
#         """
#         Given the previous two polynomials P_(i-1) and P_i, this function computes P_(i+1) 
#         using the recurrence relation:
        
#         P_(i+1)(x) = ((2i + 1) * x * P_i(x) - i * P_(i-1)(x)) / (i + 1)
#         """
#         p_ip1 = ((2. * i + 1.) * x * p_i - i * p_im1) / (i + 1)
#         return p_ip1

#     # Start by collecting the initial polynomials P_0(x) and P_1(x)
#     polys = [p_init[0], p_init[1]]
    
#     # Compute the Legendre polynomials up to degree `n_max` using the recurrence relation
#     for i in range(1, n_max):
#         p_ip1 = body_fun(i, polys[-2], polys[-1])  # Compute P_(i+1)
#         polys.append(p_ip1)  # Append the newly computed polynomial to the list

#     # Stack the computed polynomials along a new dimension to form a tensor of shape (n_max+1, ...)
#     p_n = torch.stack(polys, dim=0)

#     # Multiply the coefficient matrix `c` with the stacked polynomials `p_n` to perform a weighted sum
#     return torch.matmul(c, p_n)
# def legendre_recurrence(x, n_max):
#     """
#     Compute the Legendre polynomials up to degree n_max at a given point or array of points x using PyTorch.
    
#     Args:
#         x (torch.Tensor): The point(s) at which the Legendre polynomials are to be evaluated.
#         n_max (int): The maximum degree of Legendre polynomials to compute.
    
#     Returns:
#         torch.Tensor: A sequence of Legendre polynomial values of shape (batch_size, dimensionality, n_max+1),
#                       evaluated at point(s) x. The i-th entry of the output array corresponds
#                       to the Legendre polynomial of degree i.
#     """
#     # Ensure x is a torch tensor with dtype float64
#     x = torch.as_tensor(x, dtype=torch.float64)

#     if x.ndim == 1:
#         x = x.unsqueeze(0)  # Add a batch dimension if x is 1D

#     # Initialize the Legendre polynomials for degree 0 and 1
#     shape = x.shape + (n_max + 1,)
#     legendre_polys = torch.zeros(shape, dtype=torch.float64, device=x.device)
#     legendre_polys[..., 0] = 1.0  # P_0(x) = 1
#     if n_max >= 1:
#         legendre_polys[..., 1] = x  # P_1(x) = x
    
#     # Use the recurrence relation to compute higher-degree polynomials
#     polys = [legendre_polys[..., 0], legendre_polys[..., 1]]
#     for n in range(1, n_max):
#         # Compute P_{n+1}(x) using the Legendre recurrence relation
#         p_ip1 = ((2 * n + 1) * x * polys[-1] - n * polys[-2]) / (n + 1)
#         polys.append(p_ip1)

#     # Stack the computed polynomials along the last dimension
#     legendre_polys = torch.stack(polys, dim=-1)

#     # Remove the added batch dimension if x was originally 1D
#     if legendre_polys.shape[0] == 1:
#         return legendre_polys.squeeze(0)
#     else:
#         return legendre_polys
# New: Legendre recurrence for derivatives of order m
# def legendre_recurrence(x, n_max, m_derivative=0):
#     """
#     Compute the Legendre polynomials up to degree n_max and their m-th derivatives
#     at a given point or array of points x using PyTorch.
    
#     Args:
#         x (torch.Tensor): The point(s) at which the Legendre polynomials are to be evaluated.
#         n_max (int): The maximum degree of Legendre polynomials to compute.
#         m_derivative (int): The order of the derivative to compute.
    
#     Returns:
#         torch.Tensor: A sequence of Legendre polynomial values or their derivatives of shape 
#                       (batch_size, dimensionality, n_max+1), evaluated at point(s) x.
#                       The i-th entry of the output array corresponds to the Legendre 
#                       polynomial of degree i or its m-th derivative.
#     """
#     # Ensure x is a torch tensor with dtype float64
#     x = torch.as_tensor(x, dtype=torch.float64)

#     if x.ndim == 1:
#         x = x.unsqueeze(0)  # Add a batch dimension if x is 1D

#     # Initialize the Legendre polynomials for degree 0 and 1
#     shape = x.shape + (n_max + 1,)
#     legendre_polys = torch.zeros(shape, dtype=torch.float64, device=x.device)
#     legendre_polys[..., 0] = 1.0  # P_0(x) = 1
#     if n_max >= 1:
#         legendre_polys[..., 1] = x  # P_1(x) = x

#     # Use the recurrence relation to compute higher-degree polynomials
#     polys = [legendre_polys[..., 0], legendre_polys[..., 1]]

#     # Initialize the derivatives for each order up to m_derivative
#     derivs = torch.zeros_like(legendre_polys)
#     if m_derivative > 0:
#         # Compute derivatives
#         for n in range(n_max + 1):
#             if n < m_derivative:
#                 derivs[..., n] = 0.0
#             elif n == m_derivative:
#                 derivs[..., n] = math.factorial(2 * m_derivative) / (2 ** m_derivative * math.factorial(m_derivative))
#             else:
#                 derivs[..., n] = ((2 * n - 1) * x * derivs[..., n - 1] - (n - 1 + m_derivative) * derivs[..., n - 2]) / (n - m_derivative)
#         legendre_polys = derivs
#     else:
#         # Compute polynomials
#         for n in range(1, n_max):
#             # Compute P_{n+1}(x) using the Legendre recurrence relation
#             p_ip1 = ((2 * n + 1) * x * polys[-1] - n * polys[-2]) / (n + 1)
#             polys.append(p_ip1)
#         # Stack the computed polynomials along the last dimension
#         legendre_polys = torch.stack(polys, dim=-1)
    
#     # Remove the added batch dimension if x was originally 1D
#     if legendre_polys.shape[0] == 1:
#         return legendre_polys.squeeze(0)
#     else:
#         return legendre_polys
def legendre_recurrence(x, n_max, m_derivative=0):
    """
    Compute the Legendre polynomials up to degree n_max and their m-th derivatives
    at a given point or array of points x using PyTorch.

    Args:
        x (torch.Tensor): The point(s) at which the Legendre polynomials are to be evaluated.
        n_max (int): The maximum degree of Legendre polynomials to compute.
        m_derivative (int): The order of the derivative to compute.

    Returns:
        torch.Tensor: A sequence of Legendre polynomial values or their derivatives of shape 
                      (batch_size, dimensionality, n_max+1), evaluated at point(s) x.
                      The i-th entry of the output array corresponds to the Legendre 
                      polynomial of degree i or its m-th derivative.
    """
    # Ensure x is a torch tensor with dtype float64
    x = torch.as_tensor(x, dtype=torch.float64)

    # Add batch dimension if x is 1D
    added_batch_dim = False
    if x.ndim == 1:
        x = x.unsqueeze(0)
        added_batch_dim = True

    # Initialize tensors for Legendre polynomials
    batch_size, dims = x.shape
    polys = [torch.ones((batch_size, dims), dtype=torch.float64, device=x.device)]  # P_0(x) = 1

    if n_max >= 1:
        polys.append(x)  # P_1(x) = x

    # Compute the Legendre polynomials or their derivatives
    if m_derivative == 0:
        for n in range(1, n_max):
            p_next = ((2 * n + 1) * x * polys[-1] - n * polys[-2]) / (n + 1)
            polys.append(p_next)
            
        # Stack all computed polynomials along the last dimension
        result = torch.stack(polys, dim=-1)

    else:
        # Compute derivatives
        derivs = []
        for n in range(n_max + 1):
            if n < m_derivative:
                derivs.append(torch.zeros_like(x))
            elif n == m_derivative:
                d_next = (
                    math.factorial(2 * m_derivative) / (
                    (2 ** m_derivative) * math.factorial(m_derivative)
                    ) * torch.ones_like(x)
                )
                derivs.append(d_next)
            else:
                d_next = (
                    ((2 * n - 1) * x * derivs[n - 1] - (n - 1 + m_derivative) * derivs[n - 2]) /
                    (n - m_derivative)
                )
                derivs.append(d_next)

        # Stack all computed derivatives along the last dimension
        result = torch.stack(derivs, dim=-1)

    # Remove the added batch dimension if x was originally 1D
    if added_batch_dim:
        result = result.squeeze(0)

    return result

def hermite_recurrence(x, n_max):
    """
    Compute the Hermite polynomials up to degree n_max at a given point or array of points x using PyTorch.
    
    Args:
        x (torch.Tensor): The point(s) at which the Hermite polynomials are to be evaluated.
        n_max (int): The maximum degree of Hermite polynomials to compute.
    
    Returns:
        torch.Tensor: A sequence of Hermite polynomial values of shape (x.shape + (n_max+1,)),
                      evaluated at point(s) x. The i-th entry of the output array corresponds
                      to the Hermite polynomial of degree i.
    """
    # Ensure x is a torch tensor with dtype float64
    x = torch.as_tensor(x, dtype=torch.float64)

    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add a batch dimension if x is 1D

    # Initialize the Hermite polynomials for degree 0 and 1
    shape = x.shape + (n_max + 1,)
    hermite_polys = torch.zeros(shape, dtype=torch.float64, device=x.device)
    hermite_polys[..., 0] = 1.0  # P_0(x) = 1
    if n_max >= 1:
        hermite_polys[..., 1] = x  # P_1(x) = x

    # Use the recurrence relation to compute higher-degree polynomials
    polys = [hermite_polys[..., 0], hermite_polys[..., 1]]
    for n in range(1, n_max):
        # Compute P_{n+1}(x) without inplace operations
        p_ip1 = x * polys[-1] - n * polys[-2]
        polys.append(p_ip1)

    # Stack the computed polynomials along the last dimension
    hermite_polys = torch.stack(polys, dim=-1)

    # Remove the added batch dimension if x was originally 1D
    if hermite_polys.shape[0] == 1:
        return hermite_polys.squeeze(0)
    else:
        return hermite_polys


class OrthonormalBasis(ABC):
    @abstractmethod
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def support(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def truncated_support(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def __eq__(self, other: "OrthonormalBasis") -> bool:
        pass


# Orthonormal Legendre 
# class OrthonormalLegendre1D(OrthonormalBasis):
#     """
#     A class representing an orthonormal Legendre polynomial basis for function approximation in one dimension.

#     This class uses Legendre polynomials, which are orthogonal over the interval [-1, 1],
#     and can be rescaled to work over arbitrary domains. The coefficients of the basis
#     functions are stored and used to evaluate the basis at specified points.

#     Attributes:
#     ----------
#     _domain : Tuple[float, float]
#         The domain over which the Legendre basis is defined.
#     _coefficients : Union[torch.Tensor, torch.Tensor]
#         The coefficients of the Legendre basis functions.
#     """

#     def __init__(self, coefficients: Union[torch.Tensor, torch.Tensor], domain: Tuple[float, float] = (-1.0, 1.0)) -> None:
#         """
#         Initializes the OrthonormalLegendre1D basis object.

#         Parameters:
#         ----------
#         coefficients : Union[torch.Tensor, torch.Tensor]
#             The coefficients of the Legendre polynomial basis functions. This determines
#             the contribution of each polynomial in the expansion.
#         domain : Tuple[float, float], optional
#             The domain over which the Legendre basis is defined. Defaults to (-1.0, 1.0).
#             The domain must satisfy domain[0] < domain[1].
#         """
#         super().__init__()
#         a, b = domain
#         if a >= b:
#             raise ValueError("'domain[0]' must be smaller than 'domain[1]'")
#         self._domain = domain
#         self._coefficients = coefficients

#     @staticmethod
#     def normalisation_factors(dimension: int, order: int = 1) -> torch.Tensor:
#         """
#         Computes the normalization factors for the Legendre basis functions.

#         The normalization factors are used to ensure that the basis functions are orthonormal
#         over the given domain.

#         Parameters:
#         ----------
#         dimension : int
#             The number of basis functions (polynomial degrees).
#         order : int, optional
#             The order of the tensors. Defaults to 1 for 1D functions.

#         Returns:
#         -------
#         torch.Tensor
#             The normalization factors for the Legendre polynomials, shaped according to the
#             specified order.
#         """
#         normalisation_factors = torch.sqrt(2 * torch.arange(dimension) + 1)
#         return normalisation_factors.reshape((dimension,) + (1,) * (order - 1))

#     @property
#     def support(self) -> Tuple[float, float]:
#         """
#         Returns the domain (support) of the Legendre basis functions.

#         Returns:
#         -------
#         Tuple[float, float]
#             A tuple (min, max) representing the domain over which the Legendre polynomials
#             are defined.
#         """
#         return self._domain

#     truncated_support = support

#     @property
#     def dimension(self) -> int:
#         """
#         Returns the number of basis functions (the dimension of the Legendre space).

#         Returns:
#         -------
#         int
#             The number of Legendre polynomial basis functions, which is determined by the
#             number of coefficients provided during initialization.
#         """
#         return self._coefficients.shape[0]

#     @property
#     def coefficients(self) -> Union[torch.Tensor, torch.Tensor]:
#         """
#         Returns the coefficients of the Legendre basis functions.

#         These coefficients determine the contribution of each polynomial in the expansion.

#         Returns:
#         -------
#         Union[torch.Tensor, torch.Tensor]
#             The coefficients of the Legendre polynomials.
#         """
#         return self._coefficients

#     @property
#     def _unnormalised_coefficients(self) -> torch.Tensor:
#         """
#         Returns the unnormalized coefficients by applying the normalization factors.

#         The coefficients are multiplied by the corresponding normalization factors to ensure
#         that the Legendre basis functions are orthonormal over the domain.

#         Returns:
#         -------
#         torch.Tensor
#             The unnormalized coefficients of the Legendre basis functions.
#         """
#         normalisation_factors = self.normalisation_factors(self.dimension, self.coefficients.ndim)
#         return normalisation_factors * self.coefficients

#     def __call__(self, points: torch.Tensor) -> torch.Tensor:
#         """
#         Evaluates the Legendre basis functions at a given point or set of points.

#         The points are rescaled to the interval [-1, 1], and the Legendre polynomials are
#         evaluated using the provided coefficients.

#         Parameters:
#         ----------
#         points : torch.Tensor
#             The points at which to evaluate the Legendre polynomials. The input should
#             be a tensor of shape (m,), where `m` is the number of points.

#         Returns:
#         -------
#         torch.Tensor
#             The evaluated Legendre basis functions at the given points, shaped as (m, n),
#             where `n` is the number of basis functions.
#         """
#         shift, scaling = pu.mapparms(self.support, [-1.0, 1.0])
#         points = shift + scaling * points
#         return legendre_recurrence(points, self._unnormalised_coefficients).T

#     def __eq__(self, other: "OrthonormalLegendre1D") -> bool:
#         """
#         Compares two OrthonormalLegendre1D instances for equality.

#         Two instances are considered equal if they have the same domain and coefficients.

#         Parameters:
#         ----------
#         other : OrthonormalLegendre1D
#             Another instance to compare with.

#         Returns:
#         -------
#         bool
#             True if the two instances are equivalent, False otherwise.
#         """
#         if not isinstance(other, OrthonormalLegendre1D):
#             return False
#         return self.support == other.support and torch.allclose(self.coefficients, other.coefficients)
# class OrthonormalLegendre1D(OrthonormalBasis):
#     """
#     A class representing an orthonormal Legendre polynomial basis for function approximation in one dimension.

#     This class uses Legendre polynomials, which are orthogonal over the interval [-1, 1],
#     and can be rescaled to work over arbitrary domains.

#     Attributes:
#     ----------
#     _dimension : torch.Tensor
#         The dimension of the Legendre polynomial basis.
#     _domain : Tuple[float, float]
#         The domain over which the Legendre basis is defined.
#     """

#     def __init__(self, dimension: torch.Tensor, domain: Tuple[float, float] = (-1.0, 1.0)) -> None:
#         """
#         Initializes the OrthonormalLegendre1D basis object.

#         Parameters:
#         ----------
#         dimension : torch.Tensor
#             The dimension of the Legendre polynomial basis.
#         domain : Tuple[float, float], optional
#             The domain over which the Legendre basis is defined. Defaults to (-1.0, 1.0).
#             The domain must satisfy domain[0] < domain[1].
#         """
#         super().__init__()
#         a, b = domain
#         if a >= b:
#             raise ValueError("'domain[0]' must be smaller than 'domain[1]'")
#         self._dimension = dimension
#         self._domain = domain

#     @property
#     def support(self) -> Tuple[float, float]:
#         """
#         Returns the domain (support) of the Legendre basis functions.

#         Returns:
#         -------
#         Tuple[float, float]
#             A tuple (min, max) representing the domain over which the Legendre polynomials
#             are defined.
#         """
#         return self._domain

#     truncated_support = support

#     @property
#     def dimension(self) -> int:
#         """
#         Returns the number of basis functions (the dimension of the Legendre space).

#         Returns:
#         -------
#         int
#             The number of Legendre polynomial basis functions.
#         """
#         return self._dimension

#     def __call__(self, points: torch.Tensor) -> torch.Tensor:
#         """
#         Evaluates the Legendre basis functions at a given point or set of points.

#         The points are rescaled to the interval [-1, 1], and the Legendre polynomials are
#         evaluated.

#         Parameters:
#         ----------
#         points : torch.Tensor
#             The points at which to evaluate the Legendre polynomials. The input should
#             be a tensor of shape (m,), where `m` is the number of points.

#         Returns:
#         -------
#         torch.Tensor
#             The evaluated Legendre basis functions at the given points, shaped as (m, n),
#             where `n` is the number of basis functions.
#         """
#         shift, scaling = pu.mapparms(self.support, [-1.0, 1.0])
#         points = shift + scaling * points
#         return legendre_recurrence(points, self._dimension - 1)

#     def __eq__(self, other: "OrthonormalLegendre1D") -> bool:
#         """
#         Compares two OrthonormalLegendre1D instances for equality.

#         Two instances are considered equal if they have the same domain.
        
#         Parameters:
#         ----------
#         other : OrthonormalLegendre1D
#             Another instance to compare with.

#         Returns:
#         -------
#         bool
#             True if the two instances are equivalent, False otherwise.
#         """
#         if not isinstance(other, OrthonormalLegendre1D):
#             return False
#         return self.support == other.support
class OrthonormalLegendre1D(ABC):
    """
    A class representing an orthonormal Legendre polynomial basis for function approximation in one dimension.

    This class uses Legendre polynomials, which are orthogonal over the interval [-1, 1],
    and can be rescaled to work over arbitrary domains.

    Attributes:
    ----------
    _dimension : int
        The dimension of the Legendre polynomial basis.
    _domain : Tuple[float, float]
        The domain over which the Legendre basis is defined.
    """

    def __init__(self, dimension: int, domain: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """
        Initializes the OrthonormalLegendre1D basis object.

        Parameters:
        ----------
        dimension : int
            The dimension of the Legendre polynomial basis.
        domain : Tuple[float, float], optional
            The domain over which the Legendre basis is defined. Defaults to (-1.0, 1.0).
            The domain must satisfy domain[0] < domain[1].
        """
        super().__init__()
        a, b = domain
        if a >= b:
            raise ValueError("'domain[0]' must be smaller than 'domain[1]'")
        self._dimension = dimension
        self._domain = domain

    @property
    def support(self) -> Tuple[float, float]:
        """
        Returns the domain (support) of the Legendre basis functions.

        Returns:
        -------
        Tuple[float, float]
            A tuple (min, max) representing the domain over which the Legendre polynomials
            are defined.
        """
        return self._domain

    truncated_support = support

    @property
    def dimension(self) -> int:
        """
        Returns the number of basis functions (the dimension of the Legendre space).

        Returns:
        -------
        int
            The number of Legendre polynomial basis functions.
        """
        return self._dimension

    def __call__(self, points: torch.Tensor, m_derivative: int = 0) -> torch.Tensor:
        """
        Evaluates the Legendre basis functions or their derivatives at a given point or set of points.

        The points are rescaled to the interval [-1, 1], and the Legendre polynomials or
        their m-th derivatives are evaluated.

        Parameters:
        ----------
        points : torch.Tensor
            The points at which to evaluate the Legendre polynomials. The input should
            be a tensor of shape (m,), where `m` is the number of points.
        m_derivative : int, optional
            The order of the derivative to compute. Defaults to 0 (no derivative).

        Returns:
        -------
        torch.Tensor
            The evaluated Legendre basis functions or their derivatives at the given points,
            shaped as (m, n), where `n` is the number of basis functions.
        """
        # Map the points to the interval [-1, 1]
        shift, scaling = pu.mapparms(self.support, [-1.0, 1.0])
        points = shift + scaling * points
        
        # Evaluate the Legendre polynomials or their derivatives
        return legendre_recurrence(points, self._dimension - 1, m_derivative)

    def __eq__(self, other: "OrthonormalLegendre1D") -> bool:
        """
        Compares two OrthonormalLegendre1D instances for equality.

        Two instances are considered equal if they have the same domain.

        Parameters:
        ----------
        other : OrthonormalLegendre1D
            Another instance to compare with.

        Returns:
        -------
        bool
            True if the two instances are equivalent, False otherwise.
        """
        if not isinstance(other, OrthonormalLegendre1D):
            return False
        return self.support == other.support


# B-splines
class BSpline1D(OrthonormalBasis):
    """
    A class representing a one-dimensional B-spline basis for function approximation.

    This class provides a B-spline basis that can be used to approximate functions
    over a given domain, defined by a knot vector and a specified degree.

    Attributes:
    ----------
    _knots : torch.Tensor
        The knot vector, which defines the partition of the domain.
    _degree : int
        The degree of the B-spline (e.g., linear for degree=1, quadratic for degree=2, etc.).
    """

    def __init__(self, x: torch.Tensor, degree: int) -> None:
        """
        Initializes the B-spline basis object.

        Parameters:
        ----------
        x : torch.Tensor
            The knot vector, which defines the partition of the domain.
        degree : int
            The degree of the B-spline (e.g., linear for degree=1, quadratic for degree=2, etc.).
        """
        super().__init__()
        self._knots = x
        self._degree = degree

    @property
    def dimension(self) -> int:
        """
        Calculates the number of basis functions (the dimension of the B-spline space).

        Returns:
        -------
        int
            The number of basis functions, which is equal to (number of knots) - (degree) - 1.
        """
        return len(self._knots) - 1 - self._degree

    @property
    def support(self) -> tuple:
        """
        Returns the support of the B-spline, defined by the minimum and maximum of the knot vector.

        Returns:
        -------
        tuple[float, float]
            A tuple (min, max) representing the interval over which the B-spline is defined.
        """
        return (self._knots.min().item(), self._knots.max().item())

    truncated_support = support
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the B-spline basis functions at a given point or set of points.

        Parameters:
        ----------
        points : torch.Tensor
            The points at which to evaluate the B-spline. Should be a tensor of shape (m,), 
            where `m` is the number of points.

        Returns:
        -------
        torch.Tensor
            A tensor representing the evaluated B-spline basis functions of the specified 
            degree at the given points.
        """
        t = self._knots
        m = len(t)
        
        # Initialize the basis function array (Bx) with zeros.
        # Bx[i, j] stores the value of the i-th B-spline basis function of degree j.
        Bx = torch.zeros((m - 1, self._degree + 1), dtype=points.dtype, device=points.device)

        # Base case: B-spline basis functions of degree 0 (piecewise constant functions)
        for i in range(m - 1):
            Bx[i, 0] = (points > t[i]) * (points <= t[i + 1])

        # Cox-de Boor recursion formula
        for j in range(1, self._degree + 1):
            for i in range(m - 1 - j):  
                Bx_ij = (points - t[i]) / (t[i + j] - t[i]) * Bx[i, j - 1] + \
                        (t[i + j + 1] - points) / (t[i + j + 1] - t[i + 1]) * Bx[i + 1, j - 1]
                Bx = Bx.clone()
                Bx[i, j] = Bx_ij

        # Return the final B-spline basis functions of the specified degree
        return Bx[: m - 1 - self._degree, -1]

    def __eq__(self, other: 'BSpline1D') -> bool:
        """
        Checks if two B-spline objects are equal by comparing their knots and degree.

        Parameters:
        ----------
        other : BSpline1D
            Another B-spline object to compare with.

        Returns:
        -------
        bool
            True if the B-splines are equivalent, False otherwise.
        """
        result = isinstance(other, BSpline1D)
        result &= torch.allclose(self._knots, other._knots)
        result &= self._degree == other._degree
        return result


# Fourier
class Fourier1D(OrthonormalBasis):
    """
    A class representing a one-dimensional Fourier basis for function approximation.
    This class provides a Fourier basis that can be used to represent functions 
    in terms of sine and cosine basis functions, defined over a specified domain.
    The basis functions can be optionally normalized.

    Attributes:
    ----------
    _n : int
        The number of basis functions (sine and cosine terms).
    _domain : tuple[float, float]
        The domain over which the Fourier basis functions are defined.
    normalize : bool
        Whether to normalize the Fourier basis functions.
    period : float
        The period of the Fourier basis functions (default is 1.0).
    """

    def __init__(self, n: int, domain: tuple[float, float] = (0.0, 1.0), normalize=True, period=1.0) -> None:
        """
        Initializes the Fourier1D basis.

        Parameters:
        ----------
        n : int
            The number of basis functions (sine and cosine terms).
        domain : tuple[float, float], optional
            The domain over which the Fourier basis functions are defined, 
            defaults to (0.0, 1.0).
        normalize : bool, optional
            Whether to normalize the Fourier basis functions. Defaults to True.
        period : float, optional
            The period of the Fourier basis functions. Defaults to 1.0.
        """
        self._n = n
        self._domain = domain
        self.normalize = normalize
        self.period = period

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Fourier basis functions at a given point or set of points.

        Parameters:
        ----------
        points : torch.Tensor
            The points at which to evaluate the Fourier basis functions. 
            The input should be a tensor of shape (m,) for a single point or 
            (batch_size, m) for multiple points.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (batch_size, n) or (n,) representing the evaluated 
            Fourier basis functions at the given points, where `n` is the number 
            of basis functions. Outside the domain, the values are set to 0.
        """
        # Reshape points to ensure proper broadcasting
        points = points.unsqueeze(-1) if points.ndim == 1 else points  # Shape (batch_size, 1) if batched
        
        # Mask to set values outside the domain to 0
        domain_mask = (points >= self._domain[0]) & (points <= self._domain[1])

        # Initialize the y tensor with zeros
        y = torch.zeros((points.shape[0], self._n), dtype=points.dtype, device=points.device)

        # Only compute the Fourier basis where the domain mask is True
        if torch.any(domain_mask):
            y[:, 0] = 1
            for k in range(1, self._n):
                p = torch.tensor(math.ceil(k / 2), dtype=points.dtype, device=points.device)
                factor = 2 * torch.pi * p / self.period
                if (k - 1) % 2 == 0:
                    coef = (4 * torch.pi * p * (self._domain[1] - self._domain[0])
                            - torch.sin(4 * torch.pi * p * self._domain[0])
                            + torch.sin(4 * torch.pi * p * self._domain[1])) / (8 * torch.pi * p)
                    coef = torch.sqrt(coef)
                    y[:, k] = torch.cos(factor * points.squeeze(-1)) / coef if self.normalize else torch.cos(factor * points.squeeze(-1))
                else:
                    coef = (4 * torch.pi * p * (self._domain[1] - self._domain[0])
                            + torch.sin(4 * torch.pi * p * self._domain[0])
                            - torch.sin(4 * torch.pi * p * self._domain[1])) / (8 * torch.pi * p)
                    coef = torch.sqrt(coef)
                    y[:, k] = torch.sin(factor * points.squeeze(-1)) / coef if self.normalize else torch.sin(factor * points.squeeze(-1))
            # Apply domain mask to set points outside the domain to 0
            y = y * domain_mask.float()

        return y.squeeze(0) if points.ndim == 1 else y

    @property
    def dimension(self) -> int:
        """
        Returns the number of basis functions in the Fourier series.

        Returns:
        -------
        int
            The number of basis functions (sine and cosine terms).
        """
        return self._n

    @property
    def support(self) -> tuple:
        """
        Returns the domain of the Fourier basis.

        Returns:
        -------
        tuple[float, float]
            The domain over which the Fourier basis functions are defined.
        """
        return self._domain
    
    truncated_support = support

    def __eq__(self, other) -> bool:
        """
        Compares two Fourier1D instances for equality.

        Two Fourier1D instances are considered equal if they have the 
        same number of basis functions and the same domain.

        Parameters:
        ----------
        other : Fourier1D
            Another Fourier1D instance to compare with.

        Returns:
        -------
        bool
            True if the two instances are equal, False otherwise.
        """
        result = isinstance(other, Fourier1D)
        result &= self._n == other._n
        result &= self._domain == other._domain
        result &= self.period == other.period
        return result


# Orthonormal Hermite
class OrthonormalHermite1D(OrthonormalBasis):
    def __init__(self, dimension: torch.Tensor) -> None:
        """
        Initialize the Orthonormal Hermite 1D class with the given coefficients.
        Args:
            dimension (torch.Tensor): The degree of the Hermite polynomials.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """
        Return the basis dimension.
        """
        return self._dimension

    @property
    def support(self) -> Tuple[float, float]:
        """
        Return the support of Hermite polynomials, which is the entire real line.
        """
        return (-float("inf"), float("inf"))
    
    truncated_support = support

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Hermite polynomials at the given points using the Hermite recurrence relation.
        Args:
            points (torch.Tensor): The points where the polynomials are to be evaluated.
        
        Returns:
            torch.Tensor: The values of the Hermite polynomials at the given points.
        """
        # Use the hermite_recurrence function to compute the polynomials
        n_max = self._dimension - 1  # Maximum degree of the Hermite polynomials
        return hermite_recurrence(points, n_max)
        
    def __eq__(self, other: "OrthonormalBasis") -> bool:
        """
        Check if two OrthonormalHermite1D instances are equal.
        Args:
            other (OrthonormalBasis): Another instance to compare with.
        
        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, OrthonormalHermite1D):
            return False
        return torch.equal(self.dimension, other.dimension)