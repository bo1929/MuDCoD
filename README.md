# MuDCoD: Multi-subject Dynamic Community Detection
MuDCoD (Multi-subject Dynamic Community Detection) provides robust community detection in time-varying personalized networks modules.
It allow signal sharing between time-steps and subjects by applying eigenvector smoothing.
When available, MuDCoD leverages common signals among networks of the subjects and performs robustly when subjects do not share any apparent information. Documentation can be found [here](https://bo1929.github.io/documentations/MuDCoD/community_detection.html).

![Alt text](docs/toy-ms-dyn-nw.png?raw=true "Multi-subject Dynamic Networks")

## Installation
You can either clone the repository and crate a new virtual environment using poetry as described below, or simply use `pip install mudcod`.
1. Clone the repository, and change current directory to `mudcod/`.
  ``` bash
  git clone https://github.com/bo1929/MuDCoD.git
  cd MuDCoD
  ```
2. Create a new virtual environment with Python version 3.9 with using any version management tool, such as [`conda`](https://www.anaconda.com/products/distribution) and [`pyenv`](https://github.com/pyenv/pyenv).
    * You can use following `conda` commands.
    ``` bash
    conda create -n mudcod python=3.9.0
    conda activate mudcod
    ```
    * Alternatively, `pyenv` sets a local application-specific Python version by writing the version name to a file called `.python-version`, and automatically switches to that version when you are in the directory.
    ``` bash
    pyenv local 3.9.0
    ```
3. Use [poetry](https://python-poetry.org/docs/) to install dependencies in the `mudcod/` directory, and spawn a new shell.
  ```bash
  poetry install
  poetry shell
  ```

## Running
See the examples directory for simple examples of Multi-subject Dynamic DCBM, community detection with MuDCoD and cross-validation to choose alpha and beta, $\alpha$ and $\beta$.

For a Python interpreter to be able to import `mudcod`, it should be on your Python path.
The current working directory is (usually) included in the Python path.
So you can probably run the examples by running commands like `python examples/community_detection.py` inside the directory which you clone.
You might also want to add `mudcod` to your global Python path by installing it via `pip` or copying it to your site-packages directory.

## Dependencies
You are able to install dependencies by using `poetry install`.
However, be aware that installed dependencies do not necessarily include all libraries used in experiment scripts (files in the `experiments/` directory).
The goal was keeping actual dependencies as minimal as possible.
So, if you want to re-produce experiments on simulation data or on single-cell RNA-seq datasets, you need to go over the imported libraries and install them separately.
A tool like `pipreqs` or `pigar` might help in that case.
This is not the case for the examples (`examples/`), `poetry install` and/or `pip install mudcod` are sufficient to run them.

## Community Detection Tutorial
As described in the [documentation](https://bo1929.github.io/documentations/MuDCoD/community_detection.html) MuDCoD takes multi-dimensional `numpy` arrays as the input network argument.
Hence, whether you have constructed networks separately for each subject at different time points, or if you have your networks in a different format, it is necessary to format them appropriately.
Below, we apply MuDCoD to both simulated networks and real-data networks constructed from scRNA-seq data.

### Finding Communities in Simulated Networks
To learn more about our simulation model refer to the [documentation](https://bo1929.github.io/documentations/MuDCoD/community_detection.html) and the below section titled [Multi-subject Dynamic Degree Corrected Block Model](#multi-subject-dynamic-degree-corrected-block-model).
1. First construct a `MuSDynamicDCBM` instance, i.e., simulation model, with desired parameters.
```python
mus_dynamic_dcbm = MuSDynamicDCBM(
  n=500,
  k=10,
  p_in=(0.2, 0.4),
  p_out=(0.05, 0.1),
  time_horizon=8,
  num_subjects=16,
  r_time=0.4,
  r_subject=0.2,
)
adj_mus_dynamic, z_mus_dynamic_true = model_dcbm.simulate_mus_dynamic_dcbm(setting=1)
```
The first dimension of `numpy` arrays `adj_mus_dynamic` and `z_mus_dynamic_true` is for subjects, and the second dimension is for time points.
For networks last two dimension is an $500 \times 500$ adjacency matrix where $n$ is the number of nodes.
For community membership arrays (e.g., `z_mus_dynamic_true`) last dimension is an array of $500$ labels.

2. Set some reasonable hyper-parameters.
If you have doubts, default values should work fine for `max_K` and `n_iter`. You can use cross-validation (see `examples/cross_validation.py`) for `alpha` and `beta`.
```python
T = 8
S = 16
alpha = 0.05 * np.ones((T, 2))
beta = 0.05 * np.ones(S)
max_K = 10
n_iter = 30
```

3. Run MuDCoD iterative algorithm to find smoothed spectral representations of nodes, and then predict by clustering them to communities.
```python
pred_MuDCoD = MuDCoD(verbose=False).fit_predict(
  adj_mus_dynamic,
  alpha=alpha,
  beta=beta,
  max_K=max_K,
  n_iter=n_iter,
  opt_K="null",
  monitor_convergence=True,
)
```
You can compare `pred_MuDCoD` and `z_mus_dynamic_true` to evaluate the accuracy based on the Multi-subject Dynamic Degree Corrected Block Model. Note that they are both $16 \times 8 \times 500$ arrays.

### Finding Communities in Networks Constructed from scRNA-seq Data
Construction of gene co-expression networks from noisy and sparse scRNA-seq data is a challenging problem, and is itself a subject worth to conduct research on.
We suggest to use Dozer [3] to filter genes and construct robust networks that will enable finding meaningful gene modules.

1. Follow the instructions and use the code snippets provided [here](https://htmlpreview.github.io/?https://github.com/shanlu01/Dozer/blob/main/vignettes/introduction.html) to filter genes and construct gene co-expression networks using Dozer.
It is sufficient to follow until "Section 4: Gene Centrality Analysis" (not included) for our purposes.
Successfully running given code snippets will results in outputting networks (weighted, i.e., co-expression values) in files with `.rda` extension.

2. Use `experiments/construct_adjacency_matrix.py` with desired threshold (in terms of percentile, in our experiments 5%) to output binarized networks in `.npy` format to a given path: `/path/to/adj`.

3. Read networks from disk with `numpy.load`, and use MuDCoD as below.
3. Run MuDCoD iterative algorithm to find smoothed spectral representations of nodes, and then predict by clustering them to communities.
```python
adj = numpy.load("/path/to/adj")
pred_comm = MuDCoD(verbose=False).fit_predict(
  adj,
  alpha=0.05 * np.ones((adj.shape[1], 2)),
  beta=0.05 * np.ones(adj.shape[0]),
  max_K=50,
  n_iter=30,
  opt_K="null",
  monitor_convergence=True,
)
```

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
Different setting values correspond to the following scenarios. In our simulation experiments presented in the manuscript, we use `setting=1` and `setting=2`.

* `setting=0`: Totally independent subjects, evolve independently.

* `setting=1`: Subjects are siblings at the initial time step, then they evolve independently. (SSoT)

* `setting=2`: Subjects are siblings at each time point. (SSoS)

* `setting=3`: Subjects are parents of each other at time 0, then they evolve independently.

* `setting=4`: Subjects are parents of each other at each time point.

## References
* [1]: Liu, F., Choi, D., Xie, L., Roeder, K. Global spectral clustering in dynamic networks. Proceedings of the National Academy of Sciences 115(5), 927–932 (2018). https://doi.org/10.1073/pnas.1718449115
* [2]: Jerber, J., Seaton, D.D., Cuomo, A.S.E. et al. Population-scale single-cell RNA-seq profiling across dopaminergic neuron differentiation. Nat Genet 53, 304–312 (2021). https://doi.org/10.1038/s41588-021-00801-6
* [3]: Lu, Shan, and Sündüz Keleş. "Debiased personalized gene coexpression networks for population-scale scRNA-seq data." Genome Research 33.6 (2023): 932-947.
