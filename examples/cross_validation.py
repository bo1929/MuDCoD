import numpy as np
from mudcod.community_detection import MuDCoD, StaticSpectralCoD
from mudcod.network_simulations import MuSDynamicDCBM
from sklearn.metrics.cluster import adjusted_rand_score

T = 4
S = 4

# Multi-subject Dynamic DCBM
model_dcbm = MuSDynamicDCBM(
    n=100,  # Total number of vertices
    model_order_K=2,  # Model order, i.e. number of class labels.
    p_in=(0.2, 0.25),  # In class connectivity parameter
    p_out=(0.1, 0.1),  # Out class connectivity parameter
    time_horizon=T,  # Total number of time steps
    r_time=0.2,  # Probability of changing class labels in the next time step
    num_subjects=S,  # Total number of subjects
    r_subject=0.2,  # Probability of changing class labels while evolving
    seed=0,
)

# setting 2 (SSoS): strong signal sharing among subjects.
print(
    "Setting SSoS: signal sharing over time, subjects evolve conditioned on a common ancestor."
)
adj_mus_dynamic, z_mus_dynamic_true = model_dcbm.simulate_mus_dynamic_dcbm(setting=2)

# Some reasonable alpha-beta grids.
alpha_values = [0.01, 0.05, 0.1]
beta_values = [0.01, 0.05, 0.1]
# Reansoble values for number of iterations and maximum model order.
max_K = 20
n_iter = 25

# Initialize.
mudcod = MuDCoD(verbose=False)

obj_max, alpha, beta = -np.inf, 0, 0
for cv_alpha in alpha_values:
    for cv_beta in beta_values:
        modu, logllh = mudcod.cross_validation(
            adj_mus_dynamic,
            num_folds=5,
            alpha=cv_alpha * np.ones((T, 2)),
            beta=cv_beta * np.ones(S),
            n_iter=n_iter,
            max_K=max_K,
            opt_K="empirical",
            n_jobs=1,
        )
        print(
            f"Cross validation for alpha={cv_alpha} and beta={cv_beta} ~ "
            f"modularity:{modu}, loglikelihood:{logllh}"
        )
        if logllh > obj_max:
            alpha = cv_alpha
            beta = cv_beta

print(f"Chosen values: alpha={alpha} and beta={beta}")

# Predict communities using MuDCoD.
pred_MuDCoD = MuDCoD(verbose=False).fit_predict(
    adj_mus_dynamic,
    alpha=alpha * np.ones((T, 2)),
    beta=beta * np.ones(S),
    max_K=max_K,
    n_iter=n_iter,
    opt_K="null",
    monitor_convergence=True,
)

# Predict communities using static spectral clustering,
# seperately for each time step and subject.
pred_StaticSpectralCoD = np.empty_like(pred_MuDCoD)

for sbj in range(adj_mus_dynamic.shape[0]):
    for tp in range(adj_mus_dynamic.shape[1]):
        pred_StaticSpectralCoD[sbj, tp, :] = StaticSpectralCoD(
            verbose=False
        ).fit_predict(adj_mus_dynamic[sbj, tp, :, :])

# Compare mean adjusted Rand index scores.
ari_MuDCoD, ari_StaticSpectralCoD = 0, 0
for sbj in range(adj_mus_dynamic.shape[0]):
    for tp in range(adj_mus_dynamic.shape[1]):
        ari_MuDCoD += adjusted_rand_score(
            z_mus_dynamic_true[sbj, tp, :], pred_MuDCoD[sbj, tp, :]
        )
        ari_StaticSpectralCoD += adjusted_rand_score(
            z_mus_dynamic_true[sbj, tp, :], pred_StaticSpectralCoD[sbj, tp, :]
        )

print(f"mean(ARI) of MuDCoD: {ari_MuDCoD/T/S}")
print(f"mean(ARI) of StaticSpectralCoD: {ari_StaticSpectralCoD/T/S}")
