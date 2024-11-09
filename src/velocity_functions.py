import torch
import torch.nn as nn
import numpy as np

# Full coefficient tensor of shape dim_1 x ... dim_d x d_out
# class Func(nn.Module):
#     def __init__(self, d_in: int, d_out: int, bases: list, time_dependent: bool = False, **kwargs):
#         super().__init__(**kwargs)
#         assert len(bases) == d_in + int(time_dependent)
#         self.time_dependent = time_dependent
#         self.bases = bases
#         self.dims = [basis.dimension for basis in bases] + [d_out]
#         self.coefficient_tensor = nn.Parameter(torch.randn(tuple(self.dims)).to(torch.float32))  # Coefficient tensor initialization
    
#     def __call__(self, t: float, x: torch.Tensor):
#         result = self.coefficient_tensor  # Start with the full coefficient tensor
#         if self.time_dependent:
#             # If time-dependent, concatenate time t with x and evaluate the basis functions for each element in [t, x]
#             bases_eval = torch.stack([basis(torch.cat([torch.tensor([t]), x], dim=0)[i]) for i, basis in enumerate(self.bases)])
#         else:
#             # Otherwise, evaluate the basis functions for each element in x
#             bases_eval = torch.stack([basis(x[i]) for i, basis in enumerate(self.bases)])
#         # Perform successive contractions over the dimensions
#         for i in range(len(self.bases)):
#             # Contract the i-th dimension of the coefficient tensor with the i-th evaluated basis
#             result = torch.einsum('i...,i->...', result, bases_eval[i].float())
#         return result

class Func(nn.Module):
    def __init__(self, d_in: int, d_out: int, bases: list, time_dependent: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert len(bases) == d_in + int(time_dependent)
        self.time_dependent = time_dependent
        self.bases = bases
        self.dims = [basis.dimension for basis in bases] + [d_out]
        self.coefficient_tensor = nn.Parameter(torch.randn(tuple(self.dims)).to(torch.float32))  # Coefficient tensor initialization
    
    def __call__(self, t: float, x: torch.Tensor):
        # Check if input `x` has a batch dimension
        if x.ndim == 1:
            # If not batched, add a batch dimension (batch_size=1)
            x = x.unsqueeze(0)  # Shape becomes [1, d_in]

        batch_size = x.shape[0]  # Get the batch size
        result = self.coefficient_tensor  # Start with the full coefficient tensor

        # Handle time-dependent case
        if self.time_dependent:
            t_tensor = torch.full((batch_size, 1), t, dtype=torch.float32, device=x.device)
            input_with_time = torch.cat([t_tensor, x], dim=1)  # Concatenate time with input
            bases_eval = [basis(input_with_time[:, i]) for i, basis in enumerate(self.bases)]
        else:
            bases_eval = [basis(x[:, i]) for i, basis in enumerate(self.bases)]

        # Perform successive contractions over the dimensions
        for i in range(len(self.bases)):
            # Contract the i-th dimension of the coefficient tensor with the i-th evaluated basis
            # Using batch contraction logic, ensure it handles the batch dimension
            result = torch.einsum('i...,...i->...', result, bases_eval[i].float())

        # If the input was not batched, return a 1D tensor (remove batch dimension)
        if batch_size == 1:
            return result.squeeze(0)
        
        return result

    
# Time-dependent TT coefficient tensor with cores of shape r_i x dim_i x r_i+1
class FuncTT(nn.Module):
    def __init__(self, d_in: int, d_out: int, bases: list, rank: int, time_dependent: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert len(bases) == d_in + int(time_dependent)
        self.time_dependent = time_dependent
        self.ranks = [1] + [rank] * (d_in - 1 + int(time_dependent)) + [d_out]
        self.bases = bases
        self.dims = [basis.dimension for basis in bases]

        # Initialize TT cores as learnable parameters
        # self.size = np.sum(np.asarray(self.ranks[:-1]) * np.asarray(self.dims) * np.asarray(self.ranks[1:]))
        # self.tt_cores = nn.ParameterList(
        #     [nn.Parameter(torch.randn(self.ranks[i], self.dims[i], self.ranks[i+1]).to(torch.float64) / torch.sqrt(torch.tensor(self.size, dtype=torch.float32)))
        #         for i in range(len(self.dims))
        #     ]
        # )
        
        self.size = np.sum(np.asarray(self.ranks[:-1]) * np.asarray(self.dims) * np.asarray(self.ranks[1:]))
        self.tt_cores = nn.ParameterList(
            [nn.Parameter(torch.randn(self.ranks[i], self.dims[i], self.ranks[i+1]).to(torch.float64) / torch.sqrt(torch.tensor(self.size, dtype=torch.float32)))
                for i in range(len(self.dims) - 1)
            ]
            + [nn.Parameter(torch.zeros(self.ranks[len(self.dims) - 1], self.dims[len(self.dims) - 1], self.ranks[len(self.dims)]).to(torch.float64) / torch.sqrt(torch.tensor(self.size, dtype=torch.float32)))
            ]
        )

    def __call__(self, t: float, x: torch.Tensor):
        # Check if input `x` has a batch dimension
        if x.ndim == 1:
            # If not batched, add a batch dimension (batch_size=1)
            x = x.unsqueeze(0)  # Shape becomes [1, d_in]

        batch_size = x.shape[0]  # Get the batch size
        result = torch.ones((batch_size, 1), dtype=torch.float64, device=x.device)  # Initialize result with batch dimension
        
        # Handle time-dependent case
        if self.time_dependent:
            t_tensor = torch.full((batch_size, 1), t, dtype=torch.float64, device=x.device)
            input_with_time = torch.cat([t_tensor, x], dim=1)  # Concatenate time with input
            bases_eval = [basis(input_with_time[:, i]) for i, basis in enumerate(self.bases)]
        else:
            bases_eval = [basis(x[:, i]) for i, basis in enumerate(self.bases)]

        # Perform successive contractions between each core and basis for all dimensions
        for k in range(len(bases_eval)):
            result = torch.einsum("bi,ijk,bj->bk", result.float(), self.tt_cores[k].float(), bases_eval[k].float())

        # If the input was not batched, return a 1D tensor (remove batch dimension)
        if batch_size == 1:
            return result.squeeze(0)
        
        return result