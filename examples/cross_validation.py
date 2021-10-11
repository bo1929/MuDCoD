# One easy example for MuSPCES.
import sys
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score  # noqa: E402

sys.path.append("../")

from mudcod.dcbm import MuSDynamicDCBM  # noqa: E402
from mudcod.muspces import MuSPCES  # noqa: E402
from mudcod.static import Static  # noqa: E402

model_dcbm = MuSDynamicDCBM(
    n=100,  # Total number of vertices
    k=2,  # Model order, i.e. number of class labels.
    p_in=(0.2, 0.25),  # In class connectivity parameter
    p_out=0.1,  # Out class connectivity parameter
    time_horizon=4,  # Total number of time steps
    r_time=0.2,  # Probability of changing class labels in the next time step
    num_subjects=8,  # Total number of subjects
    r_subject=0.2,  # Probability of changing class labels while evolving
)
# scenario 3 (SSoS): strong signal sharing among subjects.
adj_ms_series, z_true = model_dcbm.simulate_ms_dynamic_dcbm(scenario=3)

# Initialize.
muspces = MuSPCES(verbose=False)

# Some reasonable alpha-beta grids.
alpha_values = [0.025, 0.05, 0.075, 0.1]
beta_values = [0.025, 0.05, 0.075, 0.1]
# Reansoble values for number of iterations and max model order.
k_max = 10
n_iter = 30

# Only fit and do not compute smoothed spectral embeddings.
# This is for cross-validation.
# Adjacency matrix must be unlaplacianized for CV, hence degree_correction=False.
muspces.fit(adj_ms_series[:, :, :, :], n_iter=0, degree_correction=False)

obj_max = 0
for cv_alpha in alpha_values:
    for cv_beta in beta_values:
        modu, logllh = muspces.cross_validation(
            n_splits=5,
            alpha=cv_alpha * np.ones((adj_ms_series.shape[1], 2)),
            beta=cv_beta * np.ones(adj_ms_series.shape[0]),
            k_max=k_max,
            n_iter=30,
            n_jobs=1,
            verbose=False,
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
muspces.fit(
    adj_ms_series[:, :, :, :],
    alpha=alpha * np.ones((adj_ms_series.shape[1], 2)),
    beta=beta * np.ones(adj_ms_series.shape[0]),
    k_max=k_max,
    n_iter=n_iter,
    monitor_convergence=True,
    degree_correction=True,
)
z_pred_mudcod = muspces.predict()
# Show convergence
## print(f"Convergence of MuDCoD: {muspces.convergence_monitor}.")

# Predict communities using static spectral clustering
# seperately for each time step and subject
static = Static(verbose=False)
z_pred_static = np.empty_like(z_pred_mudcod)

for sbj in range(adj_ms_series.shape[0]):
    for th in range(adj_ms_series.shape[1]):
        z_pred_static[sbj, th, :] = static.fit_predict(adj_ms_series[sbj, th, :, :])

# Compare mean adjusted Rand index scores.
for sbj in range(adj_ms_series.shape[0]):
    for th in range(adj_ms_series.shape[1]):
        ari_mudcod = adjusted_rand_score(z_true[sbj, th, :], z_pred_mudcod[sbj, th, :])
        ari_static = adjusted_rand_score(z_true[sbj, th, :], z_pred_static[sbj, th, :])

print(f"mean(ARI) of MuDCoD: {np.mean(ari_mudcod)}")
print(f"mean(ARI) of static: {np.mean(ari_static)}")
