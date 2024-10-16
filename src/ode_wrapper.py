import torch
import torch.nn as nn
import numpy as np

from typing import Union, Tuple
from torchdiffeq import odeint


# TT-parametrized ODE system inspired by the continuous normalizing flow formulation in Chen et al. 2018
class ContinuousNormalizingFlowODE(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func # Function defining the velocity field driving the flow
        self.transport_cost = None
        self.trace_integral = None

    # Incorporated instantaneous change of variables formula
    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivatives dx/dt, dz/dt, and dl/dt for the ODE system.
        This includes:
        - dx/dt = func(t, x)
        - dz/dt = 0.5 * ||func(t, x)||_2^2
        - dl/dt = tr(Jacobian(func(t, x)))
        """
        x = state[:-2]  # Extract x(t) from the state
        z = state[-2]   # Extract z(t) from the state
        l = state[-1]   # Extract l(t) from the state
        dxdt = self.func(t, x)
        dzdt = 0.5 * torch.norm(dxdt, p=2) ** 2
        
        def func_wrt_x(x):
            return self.func(t, x)
        
        jacobian = torch.autograd.functional.jacobian(func_wrt_x, x)
        trace_jacobian = torch.trace(jacobian)
        dldt = trace_jacobian
        return torch.cat([dxdt, dzdt.unsqueeze(0), dldt.unsqueeze(0)], dim=0)

    def solve_ode(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE for a given initial condition and time steps."""
        # Appended the initial transport cost (z(0) = 0) to the state
        y0_with_z_and_l = torch.cat([y0, torch.tensor([0.], dtype=y0.dtype), torch.tensor([0.], dtype=y0.dtype)], dim=0)
        solution = odeint(self.forward, y0_with_z_and_l, ts, rtol=rtol, atol=atol, method=method)
        return solution

    def evaluate_final_state(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE and return the final state, transport cost, and the trace integral."""
        solution = self.solve_ode(y0, ts, rtol, atol, method)
        # Extract the final state, transport cost, and trace integral
        final_state = solution[-1, :-2]
        transport_cost = solution[-1, -2]  
        trace_integral = solution[-1, -1]  
        return final_state, transport_cost, trace_integral

    # Solve ODE backward in time
    def solve_ode_backward(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE backward in time."""
        # Reverse the time vector for backward integration
        ts_backward = torch.flip(ts, dims=[0])
        # Solve the ODE starting from the final state y0 at ts[-1]
        solution = odeint(self.func, y0, ts_backward, rtol, atol, method)
        return solution[-1, :]

    def __call__(self, y0: torch.Tensor, ts: torch.Tensor = torch.tensor(np.linspace(0., 1., 8)), 
                 rtol=1e-3, atol=1e-6, method='euler', backward=False) -> torch.Tensor:
        """Main method to solve the ODE for batches or single initial conditions."""
        if backward:  # Solve ODE backward in time
            if y0.ndim == 2:  # Handle batched initial conditions
                final_states = []
                for i in range(y0.shape[0]):
                    final_state = self.solve_ode_backward(y0[i], ts, rtol, atol, method)
                    final_states.append(final_state)
                return torch.stack(final_states)
            else:  # Handle single initial condition
                final_state = self.solve_ode_backward(y0, ts, rtol, atol, method)
                return final_state
        else:  # Solve ODE foward in time
            if y0.ndim == 2:  # Handle batched initial conditions
                final_states = []
                transport_costs = []
                trace_integrals = []
                for i in range(y0.shape[0]):
                    final_state, transport_cost, trace_integral = self.evaluate_final_state(y0[i], ts, rtol, atol, method)
                    final_states.append(final_state)
                    transport_costs.append(transport_cost)
                    trace_integrals.append(trace_integral)
                self.transport_cost = torch.stack(transport_costs)
                self.trace_integral = torch.stack(trace_integrals)
                return torch.stack(final_states)
            else:  # Handle single initial condition
                final_state, transport_cost, trace_integral = self.evaluate_final_state(y0, ts, rtol, atol, method)
                self.transport_cost = transport_cost
                self.trace_integral = trace_integral
                return final_state
            

# TT-parametrized ODE system inducing coordinate flow inspired by Dektor and Venturi 2023
class CoordinateFlowODE(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func # Function defining the velocity field driving the flow
        self.transport_cost = None

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivatives dx/dt and dz/dt for the ODE system.
        This includes:
        - dx/dt = func(t, x)
        - dz/dt = 0.5 * ||func(t, x)||_2^2
        """
        x = state[:-1]  # Extract x(t) from the state
        z = state[-1]   # Extract z(t) from the state
        dxdt = self.func(t, x)
        dzdt = 0.5 * torch.norm(dxdt, p=2) ** 2
        return torch.cat([dxdt, dzdt.unsqueeze(0),], dim=0)

    def solve_ode(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE for a given initial condition and time steps."""
        # Appended the initial transport cost (z(0) = 0) to the state
        y0_with_z = torch.cat([y0, torch.tensor([0.], dtype=y0.dtype)], dim=0)
        solution = odeint(self.forward, y0_with_z, ts, rtol=rtol, atol=atol, method=method)
        return solution

    def evaluate_final_state(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE and return the final state, transport cost."""
        solution = self.solve_ode(y0, ts, rtol, atol, method)
        # Extract the final state and transport cost
        final_state = solution[-1, :-1]
        transport_cost = solution[-1, -1] 
        return final_state, transport_cost

    # Solve ODE backward in time
    def solve_ode_backward(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE backward in time."""
        # Reverse the time vector for backward integration
        ts_backward = torch.flip(ts, dims=[0])
        # Solve the ODE starting from the final state y0 at ts[-1]
        solution = odeint(self.func, y0, ts_backward, rtol, atol, method)
        return solution[-1, :]

    def __call__(self, y0: torch.Tensor, ts: torch.Tensor = torch.tensor(np.linspace(0., 1., 8)), 
                 rtol=1e-3, atol=1e-6, method='euler', backward=False) -> torch.Tensor:
        """Main method to solve the ODE for batches or single initial conditions."""
        if backward:  # Solve ODE backward in time
            if y0.ndim == 2:  # Handle batched initial conditions
                final_states = []
                for i in range(y0.shape[0]):
                    final_state = self.solve_ode_backward(y0[i], ts, rtol, atol, method)
                    final_states.append(final_state)
                return torch.stack(final_states)
            else:  # Handle single initial condition
                final_state = self.solve_ode_backward(y0, ts, rtol, atol, method)
                return final_state
        else:  # Solve ODE foward in time
            if y0.ndim == 2:  # Handle batched initial conditions
                final_states = []
                transport_costs = []
                for i in range(y0.shape[0]):
                    final_state, transport_cost = self.evaluate_final_state(y0[i], ts, rtol, atol, method)
                    final_states.append(final_state)
                    transport_costs.append(transport_cost)
                self.transport_cost = torch.stack(transport_costs)
                return torch.stack(final_states)
            else:  # Handle single initial condition
                final_state, transport_cost = self.evaluate_final_state(y0, ts, rtol, atol, method)
                self.transport_cost = transport_cost
                return final_state


# Extended formulation of the TT-parametrized ODE system for Gaussian density regression task inspired by David Sommer
class ExtendedFlowODE(nn.Module):
    # def __init__(self, d: int, bases: list, rank: int = 6, time_dependent: bool = False, **kwargs):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        # self.func = FuncTT(d_in=d, d_out=1, bases=bases, rank=rank, time_dependent=time_dependent)
        self.func = func
        self.transport_cost = None

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative dx/dt, dz/dt for the ODE system.
        This includes:
        - dx_1/dt = func(t, x_{2:d+1}) * x_1
        - dx_{2:d+1}/dt = 0_d
        - dz/dt = 0.5 * ||func(t, x)||_2^2
        """
        x_1 = state[0]
        x_2 = state[1:-1]
        z = state[-1]
        dxdt = self.func(t, x_2) * x_1
        dzdt = 0.5 * torch.norm(dxdt, p=2) ** 2
        return torch.cat([dxdt, torch.zeros(len(state) - 2), dzdt.unsqueeze(0)], dim=0)

    def solve_ode(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE for a given initial condition and time steps."""
        # Appended the initial transport cost (z(0) = 0) to the state
        y0_with_z = torch.cat([torch.tensor([1.], dtype=y0.dtype), y0, torch.tensor([0.], dtype=y0.dtype)], dim=0)
        solution = odeint(self.forward, y0_with_z, ts, rtol=rtol, atol=atol, method=method)
        return solution
    
    def evaluate_final_state(self, y0: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str) -> torch.Tensor:
        """Solve the ODE and return the final state and transport cost."""
        solution = self.solve_ode(y0, ts, rtol, atol, method)
        # Extract the final state and transport cost
        final_state = solution[-1, 0]
        transport_cost = solution[-1, -1]  
        return final_state, transport_cost
    
    def __call__(self, y0: torch.Tensor, ts: torch.Tensor = torch.tensor(np.linspace(0., 1., 8)), 
                 rtol=1e-3, atol=1e-6, method='euler', backward=False) -> torch.Tensor:
        """Main method to solve the ODE for batches or single initial conditions."""
        if y0.ndim == 2:  # Handle batched initial conditions
            final_states = []
            transport_costs = []
            for i in range(y0.shape[0]):
                final_state, transport_cost = self.evaluate_final_state(y0[i], ts, rtol, atol, method)
                final_states.append(final_state)
                transport_costs.append(transport_cost)
            self.transport_cost = torch.stack(transport_costs)
            return torch.stack(final_states)
        
        else:  # Handle single initial condition
            final_state, transport_cost = self.evaluate_final_state(y0, ts, rtol, atol, method)
            self.transport_cost = transport_cost
            return final_state