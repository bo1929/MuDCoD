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

# Some reasonable hyper-parameters.
alpha = 0.05 * np.ones((adj_ms_series.shape[1], 2))
beta = 0.05 * np.ones(adj_ms_series.shape[0])
k_max = 10
n_iter = 30

# Predict communities using MuDCoD.
muspces.fit(
    adj_ms_series[:, :, :],
    alpha=alpha,
    beta=beta,
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
