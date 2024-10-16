import torch
import numpy as np
import matplotlib.pyplot as plt

from torchdiffeq import odeint
from itertools import cycle

# Function for plotting of the evolution of the ODE flow-driven trajectroies over time
def plot_trajectories_over_time(func, initial_points, ts, rtol=1e-3, atol=1e-6, method='euler', backward=False):
    """
    Plots the trajectories of the ODE with rhs func over time starting from a set of initial points for a multi-dimensional vector field.
    """
    # Generate trajectories
    with torch.no_grad():
        trajectories = []
        for point in initial_points:
            # Simulate the ODE over time
            traj = odeint(func, point, ts, rtol=rtol, atol=atol, method=method)
            trajectories.append(traj.squeeze(1).numpy())  # Store the full trajectory

    # Convert list to numpy array for plotting
    trajectories = np.array(trajectories)

    # Get the dimensionality of the output from the shape of the first trajectory
    output_dim = trajectories.shape[-1]

    # Define a color cycle
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Plotting
    plt.figure(figsize=(10, 5))
    for i, traj in enumerate(trajectories):
        color = next(color_cycle)  # Get the next color from the cycle
        
        # Plot each dimension's trajectory with matching color
        for dim in range(output_dim):
            linestyle = '--' if dim % 2 == 0 else '-'  # Alternate line styles for visual distinction
            plt.plot(ts.numpy(), traj[:, dim], label=f'Trajectory {i+1} (dim {dim+1})', linestyle=linestyle, color=color)
    
    plt.xlabel('Time (t)')
    plt.ylabel('Values')
    plt.title('Trajectories Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function for plotting the vector field driving the ODE flow at different points in time
def plot_vector_fields_at_times(func, x_range, y_range, plot_time):
    """
    Plots the vector field at specified times using the ODE with rhs func.
    """
    # Generate grid points for the vector field
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)

    # Set up the figure and subplots
    fig, axes = plt.subplots(1, len(plot_time), figsize=(6 * len(plot_time), 6))

    # Ensure axes is iterable even if there's only one plot
    if len(plot_time) == 1:
        axes = [axes]

    # Iterate over the specified times to create each plot
    for idx, t in enumerate(plot_time):
        ax = axes[idx]
        xy = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)

        # Simulate the ODE to get the vector field at time t
        vector_field = []
        for point in xy:
            # Use integration to simulate vector field evolution
            result = func(t, point)
            vector_field.append(result.detach().numpy())  # Get "velocity" as change over time

        u = np.array(vector_field)

        # Reshape vector field components
        U, V = u[:, 0].reshape(X.shape), u[:, 1].reshape(Y.shape)
        
        # Plot the vector field using quiver
        ax.quiver(X, Y, U, V, color='blue')
        ax.set_title(f"Vector Field at Time t={t.item():.2f}")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()