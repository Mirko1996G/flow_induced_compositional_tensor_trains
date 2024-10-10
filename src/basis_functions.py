"""
This module contains the implementation of various basis functions, including
Legendre, B-Spline, and Fourier bases, originally implemented in JAX by Charles Miranda
from Weierstrass Institut. The current version is a PyTorch adaptation, with modifications
for use in functional tensor trains.
"""

import torch
import math
import numpy as np
import numpy.polynomial.polyutils as pu

from abc import ABC, abstractmethod
from typing import Union, Tuple

def legendre_recurrence(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes a set of Legendre polynomials up to the order specified by the shape of the 
    coefficient matrix `c`, using a recurrence relation. Then multiplies these polynomials 
    by the coefficient matrix `c` to return the result.

    Parameters:
    ----------
    x : torch.Tensor
        A tensor containing the input points where the Legendre polynomials will be evaluated. 
    
    c : torch.Tensor
        A square matrix of coefficients (shape n x n)

    Returns:
    -------
    torch.Tensor
        A tensor representing the result of the matrix multiplication between `c` and the 
        computed Legendre polynomials. The shape of the output tensor depends on the shape 
        of `x` and the degree of the polynomial (derived from `c`).

    Notes:
    ------
    This function uses the recurrence relation for Legendre polynomials:
    
    P_0(x) = 1
    P_1(x) = x
    P_n(x) = ((2n - 1) * x * P_(n-1)(x) - (n - 1) * P_(n-2)(x)) / n

    It first computes all the Legendre polynomials up to the degree specified by `c`, then 
    combines them with the coefficient matrix `c` via matrix multiplication.
    """
    
    # Assert that c is a square matrix
    assert c.ndim == 2 and c.shape[0] == c.shape[1], "Matrix c must be square"
    
    # Initialize the first two Legendre polynomials:
    # P_0(x) = 1 and P_1(x) = x, where p_init stores these initial values
    p_init = torch.zeros((2,) + x.shape, dtype=float, device=x.device)  # Initialize a tensor for P_0 and P_1
    p_init[0] = 1.0  # P_0(x) = 1
    p_init[1] = x    # P_1(x) = x

    # Get the maximum degree for the polynomials (based on the size of matrix c)
    n_max = c.shape[0] - 1

    # Define the recurrence relation for generating Legendre polynomials
    def body_fun(i, p_im1, p_i):
        """
        Given the previous two polynomials P_(i-1) and P_i, this function computes P_(i+1) 
        using the recurrence relation:
        
        P_(i+1)(x) = ((2i + 1) * x * P_i(x) - i * P_(i-1)(x)) / (i + 1)
        """
        p_ip1 = ((2. * i + 1.) * x * p_i - i * p_im1) / (i + 1)
        return p_ip1

    # Start by collecting the initial polynomials P_0(x) and P_1(x)
    polys = [p_init[0], p_init[1]]
    
    # Compute the Legendre polynomials up to degree `n_max` using the recurrence relation
    for i in range(1, n_max):
        p_ip1 = body_fun(i, polys[-2], polys[-1])  # Compute P_(i+1)
        polys.append(p_ip1)  # Append the newly computed polynomial to the list

    # Stack the computed polynomials along a new dimension to form a tensor of shape (n_max+1, ...)
    p_n = torch.stack(polys, dim=0)

    # Multiply the coefficient matrix `c` with the stacked polynomials `p_n` to perform a weighted sum
    return torch.matmul(c, p_n)

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
class OrthonormalLegendre1D(OrthonormalBasis):
    """
    A class representing an orthonormal Legendre polynomial basis for function approximation in one dimension.

    This class uses Legendre polynomials, which are orthogonal over the interval [-1, 1],
    and can be rescaled to work over arbitrary domains. The coefficients of the basis
    functions are stored and used to evaluate the basis at specified points.

    Attributes:
    ----------
    _domain : Tuple[float, float]
        The domain over which the Legendre basis is defined.
    _coefficients : Union[torch.Tensor, torch.Tensor]
        The coefficients of the Legendre basis functions.
    """

    def __init__(self, coefficients: Union[torch.Tensor, torch.Tensor], domain: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """
        Initializes the OrthonormalLegendre1D basis object.

        Parameters:
        ----------
        coefficients : Union[torch.Tensor, torch.Tensor]
            The coefficients of the Legendre polynomial basis functions. This determines
            the contribution of each polynomial in the expansion.
        domain : Tuple[float, float], optional
            The domain over which the Legendre basis is defined. Defaults to (-1.0, 1.0).
            The domain must satisfy domain[0] < domain[1].
        """
        super().__init__()
        a, b = domain
        if a >= b:
            raise ValueError("'domain[0]' must be smaller than 'domain[1]'")
        self._domain = domain
        self._coefficients = coefficients

    @staticmethod
    def normalisation_factors(dimension: int, order: int = 1) -> torch.Tensor:
        """
        Computes the normalization factors for the Legendre basis functions.

        The normalization factors are used to ensure that the basis functions are orthonormal
        over the given domain.

        Parameters:
        ----------
        dimension : int
            The number of basis functions (polynomial degrees).
        order : int, optional
            The order of the tensors. Defaults to 1 for 1D functions.

        Returns:
        -------
        torch.Tensor
            The normalization factors for the Legendre polynomials, shaped according to the
            specified order.
        """
        normalisation_factors = torch.sqrt(2 * torch.arange(dimension) + 1)
        return normalisation_factors.reshape((dimension,) + (1,) * (order - 1))

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
            The number of Legendre polynomial basis functions, which is determined by the
            number of coefficients provided during initialization.
        """
        return self._coefficients.shape[0]

    @property
    def coefficients(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Returns the coefficients of the Legendre basis functions.

        These coefficients determine the contribution of each polynomial in the expansion.

        Returns:
        -------
        Union[torch.Tensor, torch.Tensor]
            The coefficients of the Legendre polynomials.
        """
        return self._coefficients

    @property
    def _unnormalised_coefficients(self) -> torch.Tensor:
        """
        Returns the unnormalized coefficients by applying the normalization factors.

        The coefficients are multiplied by the corresponding normalization factors to ensure
        that the Legendre basis functions are orthonormal over the domain.

        Returns:
        -------
        torch.Tensor
            The unnormalized coefficients of the Legendre basis functions.
        """
        normalisation_factors = self.normalisation_factors(self.dimension, self.coefficients.ndim)
        return normalisation_factors * self.coefficients

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Legendre basis functions at a given point or set of points.

        The points are rescaled to the interval [-1, 1], and the Legendre polynomials are
        evaluated using the provided coefficients.

        Parameters:
        ----------
        points : torch.Tensor
            The points at which to evaluate the Legendre polynomials. The input should
            be a tensor of shape (m,), where `m` is the number of points.

        Returns:
        -------
        torch.Tensor
            The evaluated Legendre basis functions at the given points, shaped as (m, n),
            where `n` is the number of basis functions.
        """
        shift, scaling = pu.mapparms(self.support, [-1.0, 1.0])
        points = shift + scaling * points
        return legendre_recurrence(points, self._unnormalised_coefficients).T

    def __eq__(self, other: "OrthonormalLegendre1D") -> bool:
        """
        Compares two OrthonormalLegendre1D instances for equality.

        Two instances are considered equal if they have the same domain and coefficients.

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
        return self.support == other.support and torch.allclose(self.coefficients, other.coefficients)


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
    """

    def __init__(self, n: int, domain: tuple[float, float] = (0.0, 1.0), normalize=True) -> None:
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
        """
        self._n = n
        self._domain = domain
        self.normalize = normalize

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Fourier basis functions at a given point or set of points.

        Parameters:
        ----------
        points : torch.Tensor
            The points at which to evaluate the Fourier basis functions. 
            The input should be a tensor of shape (m,), where `m` is the 
            number of points.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (n,) representing the evaluated Fourier basis 
            functions at the given points, where `n` is the number of basis 
            functions.
        """
        # Initialize the y tensor to ones
        y = torch.ones((self._n,), dtype=points.dtype, device=points.device)

        for k in range(1, self._n):
            p = torch.tensor(math.ceil(k / 2))
            if (k - 1) % 2 == 0:
                coef = (
                    4 * torch.pi * p * (self._domain[1] - self._domain[0])
                    - torch.sin(4 * torch.pi * p * self._domain[0])
                    + torch.sin(4 * torch.pi * p * self._domain[1])
                ) / (8 * torch.pi * p)
                coef = torch.sqrt(coef)
                y[k] = torch.cos(2 * torch.pi * p * points) / coef if self.normalize else torch.cos(2 * torch.pi * p * points)
            else:
                coef = (
                    4 * torch.pi * p * (self._domain[1] - self._domain[0])
                    + torch.sin(4 * torch.pi * p * self._domain[0])
                    - torch.sin(4 * torch.pi * p * self._domain[1])
                ) / (8 * torch.pi * p)
                coef = torch.sqrt(coef)
                y[k] = torch.sin(2 * torch.pi * p * points) / coef if self.normalize else torch.sin(2 * torch.pi * p * points)

        return y

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
        return result