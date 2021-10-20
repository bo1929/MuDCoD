# MuDCoD: Multi-subject Dynamic Community Detection
 
![Alt text](docs/toy-ms-dyn-nw.png?raw=true "Multi-subject Dynamic Networks") 

## Installation

## Running
See the examples directory for simple examples of Multi-subject Dynamic DCBM,
community detection with MuDCoD and cross validation to choose alpha and beta.

For a Python interpreter to be able to import `mudcod`, it should be on your
Python path. Since the current working directory is (usually) included in the
Python path. So you can probably run the examples by running commands like
`python examples/community_detection.py` inside the directory, which you cloned
with git clone. You might also want to add `mudcod` to your global Python path
by installing it via `pip` or copying it to your site-packages directory.

## Simple Demonstration

## File Structure
The code is mostly organized and readable.

* `mudcod/`: Root folder of the package.
  * `dcbm.py`: Degree Corrected Block Models for dynamic and multi-subject dynamic settings.
  * `spectral.py`: Mixin class for common spectral methods such as model order selection eigen value completion.
  * `nw.py`: Loss functions (modularity and loglikelihood) and network similarity measures<sup>[1](#myfootnote1)</sup>.
  * `static.py`: Implementation of static spectral clustering.
  * `pisces.py`: Implementation of PisCES.
  * `muspces.py`: Implementation of MuDCoD.
  * `utils/`: Utilities.
    * `visualization.py`: Plotting methods for networks and results.
    * `sutils.py`: Read/write and logging utilities.
* `simulations/`: Simulation scripts for performance comparison of community detection methods.
  * `classes-DCBM/`: Put DCBM parameter configurations here, example `.yaml` files are provided.
  * `configuration/`: Simulation configurations will be automatically saved here when you run cross-validation.
  * `log/`: Output directory for log files.
  * `report.py`: Reports and summaries of results of performed simulations.
  * `run_simulation.sh`: Bash script that can be used as a CLI to perform many simulation replicates.
  * `simulation.py`: A CLI for Multi-subject Dynamic DCBM simulations, perform cross-validation and/or community detection.
* `examples/`:
  * `mus_dyn_dcbm.py`
  * `cross_validation.py`
  * `community_detection.py`
* `docs/`: Documentation.

Below folders are not included in the package.
They are used to applying MuDCoD on a scRNA-seq study of long-term human induced pluripotent stem cell (iPSC) across multiple donors [2].
It might be useful to check them out if you want to apply MuDCoD on such data.
* `experiments/`
* `notebooks/`
* `results/`

<a name="myfootnote1">1</a>: Network similarity measures are adapted from [netrd library](https://github.com/netsiphd/netrd).

## References
* [1] Liu, F., Choi, D., Xie, L., Roeder, K. Global spectral clustering in dynamic networks. Proceedings of the National Academy of Sciences 115(5), 927–932 (2018). https://doi.org/10.1073/pnas.1718449115
* [2] Jerber, J., Seaton, D.D., Cuomo, A.S.E. et al. Population-scale single-cell RNA-seq profiling across dopaminergic neuron differentiation. Nat Genet 53, 304–312 (2021). https://doi.org/10.1038/s41588-021-00801-6
