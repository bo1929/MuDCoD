from mudcod.network_simulations import MuSDynamicDCBM


def print_shape(adj, z):
    print(
        f"Shape of the time series of multi-subject adjacency matrices is: {adj.shape}",
    )
    print("(number of subjects, time-horizon, n, n)")
    print(
        f"Shape of the time series of multi-subject community labels is : {z.shape}",
    )
    print("(number of subjects, time-horizon, n)\n")


# Multi-subject Dynamic DCBM
model_dcbm = MuSDynamicDCBM(
    n=500,  # Total number of vertices
    model_order_K=10,  # Model order, i.e. number of class labels.
    p_in=(0.2, 0.25),  # In class connectivity parameter
    p_out=(0.1, 0.1),  # Out class connectivity parameter
    time_horizon=4,  # Total number of time steps
    r_time=0.2,  # Probability of changing class labels in the next time step
    num_subjects=8,  # Total number of subjects
    r_subject=0.2,  # Probability of changing class labels while evolving
    seed=0,
)

# setting 1 (SSoT): signal sharing over time, subjects evolve independently.
print("Setting SSoT: signal sharing over time, subjects evolve independently.")
adj_mus_dynamic, z_mus_dynamic = model_dcbm.simulate_mus_dynamic_dcbm(setting=1)
print_shape(adj_mus_dynamic, z_mus_dynamic)

# setting 2 (SSoS): strong signal sharing among subjects.
print("Setting SSoS: strong signal sharing among subjects.")
adj_mus_dynamic, z_mus_dynamic = model_dcbm.simulate_mus_dynamic_dcbm(setting=2)
print_shape(adj_mus_dynamic, z_mus_dynamic)

# You can similarly generate single-subject dynamic DCBM networks and static DCBM networks.
