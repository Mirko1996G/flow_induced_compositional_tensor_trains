import torch
import torch.nn as nn
import numpy as np

from src.basis_functions import OrthonormalBasis

# Full coefficient tensor of shape dim_1 x ... dim_d x d_out
class Func(nn.Module):
    def __init__(self, d_in: int, d_out: int, bases: list[OrthonormalBasis], time_dependent: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert len(bases) == d_in + int(time_dependent)
        self.time_dependent = time_dependent
        self.bases = bases
        self.dims = [basis.dimension for basis in bases] + [d_out]
        self.coefficient_tensor = nn.Parameter(torch.randn(tuple(self.dims)).to(torch.float32))  # Coefficient tensor initialization
    
    def __call__(self, t: float, x: torch.Tensor):
        result = self.coefficient_tensor  # Start with the full coefficient tensor
        if self.time_dependent:
            # If time-dependent, concatenate time t with x and evaluate the basis functions for each element in [t, x]
            bases_eval = torch.stack([basis(torch.cat([torch.tensor([t]), x], dim=0)[i]) for i, basis in enumerate(self.bases)])
        else:
            # Otherwise, evaluate the basis functions for each element in x
            bases_eval = torch.stack([basis(x[i]) for i, basis in enumerate(self.bases)])
        # Perform successive contractions over the dimensions
        for i in range(len(self.bases)):
            # Contract the i-th dimension of the coefficient tensor with the i-th evaluated basis
            result = torch.einsum('i...,i->...', result, bases_eval[i].float())
        return result


# Time-dependent TT coefficient tensor with cores of shape r_i x dim_i x r_i+1
class FuncTT(nn.Module):
    def __init__(self, d_in: int, d_out: int, bases: list[OrthonormalBasis], rank: int, time_dependent: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert len(bases) == d_in + int(time_dependent)
        self.time_dependent = time_dependent
        self.ranks = [1] + [rank] * (d_in - 1 + int(time_dependent)) + [d_out]
        self.bases = bases
        self.dims = [basis.dimension for basis in bases]

        # Initialize TT cores as learnable parameters
        self.size = np.sum(np.asarray(self.ranks[:-1]) * np.asarray(self.dims) * np.asarray(self.ranks[1:]))
        self.tt_cores = nn.ParameterList(
            [nn.Parameter(torch.randn(self.ranks[i], self.dims[i], self.ranks[i+1]).to(torch.float64) / torch.sqrt(torch.tensor(self.size, dtype=torch.float32)))
                for i in range(len(self.dims))
            ]
        )
    
    def __call__(self, t: float, x: torch.Tensor):
        result = torch.ones(1)
        if self.time_dependent:
            # Evaluate the basis functions for each element in [t, x]
            bases_eval = torch.stack([basis(torch.cat([torch.tensor([t]), x], dim=0)[i]) for i, basis in enumerate(self.bases)])
        else:
            # Evaluate the basis functions for each element in x
            bases_eval = torch.stack([basis(x[i]) for i, basis in enumerate(self.bases)])
        # Perform successive contractions between each core and basis for all dimensions
        for k in range(len(bases_eval)):
            result = torch.einsum("i,ijk,j->k", result.float(), self.tt_cores[k].float(), bases_eval[k].float())
        return result