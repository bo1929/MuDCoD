# MuDCoD: Multi-subject Dynamic Community Detection
MuDCoD (Multi-subject Dynamic Community Detection) provides robust community
detection in time-varying personalized networks modules. It allow signal
sharing between time-steps and subjects by applying eigenvector smoothing.
When available, MuDCoD leverages common signals among networks of the subjects
and performs robustly when subjects do not share any apparent information.

![Alt text](docs/toy-ms-dyn-nw.png?raw=true "Multi-subject Dynamic Networks") 

## Installation

## Running
See the examples directory for simple examples of Multi-subject Dynamic DCBM,
community detection with MuDCoD and cross validation to choose alpha and beta.

For a Python interpreter to be able to import `mudcod`, it should be on your
Python path. The current working directory is (usually) included in the Python
path. So you can probably run the examples by running commands like `python
examples/community_detection.py` inside the directory which you clone. You
might also want to add `mudcod` to your global Python path by installing it via
`pip` or copying it to your site-packages directory.

## Dependencies
You are able to install dependencies by using `poetry install`. However, be
aware that installed dependencies do not necessarily include all libraries used
in simulation and experiment scripts (`simulations/` and `experiments/`). The
goal is to keep actual dependencies as minimal as possible. So, if you want to
re-produce simulation or experiment results, you need to go over the imported
libraries and install them separately. A tool like `pipreqs` can help. This is
not the case for the examples (`examples/`), `poetry install` is sufficient to
run them.

## Simple Demonstration

## File Structure
The code is mostly organized and readable.

<!-- * `mudcod/`: Root folder of the package. -->
<!--   * `dcbm.py`: Degree Corrected Block Models for dynamic and multi-subject dynamic settings. -->
<!--   * `spectral.py`: Mixin class for common spectral methods such as model order selection eigen value completion. -->
<!--   * `nw.py`: Loss functions (modularity and loglikelihood) and network similarity measures<sup>[1](#myfootnote1)</sup>. -->
<!--   * `static.py`: Implementation of static spectral clustering. -->
<!--   * `pisces.py`: Implementation of PisCES. -->
<!--   * `muspces.py`: Implementation of MuDCoD. -->
<!--   * `utils/`: Utilities. -->
<!--     * `visualization.py`: Plotting methods for networks and results. -->
<!--     * `sutils.py`: Read/write and logging utilities. -->
<!-- * `simulations/`: Simulation scripts for performance comparison of community detection methods. -->
<!--   * `classes-DCBM/`: Put DCBM parameter configurations here, example `.yaml` files are provided. -->
<!--   * `configuration/`: Simulation configurations will be automatically saved here when you run cross-validation. -->
<!--   * `log/`: Output directory for log files. -->
<!--   * `report.py`: Reports and summaries of results of performed simulations. -->
<!--   * `run_simulation.sh`: Bash script that can be used as a CLI to perform many simulation replicates. -->
<!--   * `simulation.py`: A CLI for Multi-subject Dynamic DCBM simulations, perform cross-validation and/or community detection. -->
<!-- * `examples/`: -->
<!--   * `mus_dyn_dcbm.py` -->
<!--   * `cross_validation.py` -->
<!--   * `community_detection.py` -->
<!-- * `docs/`: Documentation. -->

<!-- Below folders are not included in the package. -->
<!-- They are used to applying MuDCoD on a scRNA-seq study of long-term human induced pluripotent stem cell (iPSC) across multiple donors [2]. -->
<!-- It might be useful to check them out if you want to apply MuDCoD on such data. -->
<!-- * `experiments/` -->
<!-- * `notebooks/` -->
<!-- * `results/` -->

<!-- <a name="myfootnote1">1</a>: Network similarity measures are adapted from [netrd library](https://github.com/netsiphd/netrd). -->

## Multi-subject Dynamic Degree Corrected Block Model
There are three classes, namely `DCBM`, `DynamicDCBM`, and `MuSDynamicDCBM`.

We use the `MuSDynamicDCBM` class to generate simulation networks with a given parameter configuration.

For example, you can initialize a class instance as below.
```python
mus_dynamic_dcbm = MuSDynamicDCBM(
  n=500,
  k=10,
  p_in=(0.2, 0.4),
  p_out=(0.05, 0.1),
  time_horizon=8,
  r_time=0.4,
  num_subjects=16,
  r_subject=0.2,
  seed=0
)
```
This will initialize a multi-subject dynamic degree corrected block model with $500$ nodes, $10$ communities, $16$ subjects, and $8$ number of time steps.
The connectivity matrix values will be sampled from $\textnormal{Uniform}(0.2, 0.4)$ for the nodes within the same community and from  $\textnormal{Uniform}(0.05, 0.1)$ for the nodes in different communities.
The parameter for the network evolution along the time will be $0.4$, note that higher values imply more rapid temporal change.
Similarly, `r_subject=0.2` parameterize the degree of dissimilarity among subjects.
After initializing the `MuSDynamicDCBM`, we can generate an instance of multi-subject time series of networks by running the below line of code.
```python
mus_dynamic_dcbm.simulate_mus_dynamic_dcbm(setting=setting)
```
Different setting values correspond to the following scenarios. In our experiments, we use `setting=1` and `setting=2`.

* `setting=0`: Totally independent subjects, evolve independently.

* `setting=1`: Subjects are siblings at the initial time step, then they evolve independently. (SSoT)

* `setting=2`: Subjects are siblings at each time point. (SSoS)

* `setting=3`: Subjects are parents of each other at time 0, then they evolve independently.

* `setting=4`: Subjects are parents of each other at each time point.

## References
* [1] Liu, F., Choi, D., Xie, L., Roeder, K. Global spectral clustering in dynamic networks. Proceedings of the National Academy of Sciences 115(5), 927–932 (2018). https://doi.org/10.1073/pnas.1718449115
* [2] Jerber, J., Seaton, D.D., Cuomo, A.S.E. et al. Population-scale single-cell RNA-seq profiling across dopaminergic neuron differentiation. Nat Genet 53, 304–312 (2021). https://doi.org/10.1038/s41588-021-00801-6
