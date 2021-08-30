import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

MAIN_DIR = Path(__file__).absolute().parent.parent
SIMULATION_DIR = MAIN_DIR / "simulations"
RESULT_DIR = MAIN_DIR / "results"
RESULTS_PATH = RESULT_DIR / "simulation_results"
FIGURE_DIR = RESULT_DIR / "simulation_figures"
sys.path.append(str(MAIN_DIR))


import dypoces.utils.visualization as VIS  # noqa: E402

from dypoces.utils import sutils  # noqa: E402

sns.set_theme(style="whitegrid")
sutils.safe_create_dir(RESULTS_PATH)
sutils.safe_create_dir(FIGURE_DIR)

objkey = "loglikelihood"


def get_dataframe(result_dict, index_row, columns, multi_index):
    resultsDf = pd.DataFrame(result_dict["val"], index=index_row, columns=columns)

    if not multi_index:
        resultsDf.reset_index(inplace=True)

    return resultsDf


def get_info_from_name(simul_name):
    mkint = lambda x: int(x[2:])  # noqa: E731
    mkfloat = lambda x: float("0." + x[2:])  # noqa: E731

    simul_info = simul_name.split("_")

    if simul_name.startswith(".") or len(simul_info) != 6:
        raise ValueError(f"Unkown simulation name format: {simul_name}.")

    class_dcbm, case_msd, th, rt, ns, rs = (
        simul_info[0],
        simul_info[1],
        simul_info[2],
        simul_info[3],
        simul_info[4],
        simul_info[5],
    )

    ns, th = mkint(ns), mkint(th)
    rs, rt = mkfloat(rs), mkfloat(rt)

    return class_dcbm, case_msd, th, rt, ns, rs


def read_cv_results(results_path, multi_index=False):
    # {{{
    results_path = Path(results_path)
    result_dirs = [f for f in results_path.iterdir() if f.is_dir()]
    result_dict = defaultdict(list)
    num_result = len(result_dirs)

    for i, path in enumerate(sorted(result_dirs)):
        simul_name = str(path.stem)
        class_dcbm, case_msd, th, rt, ns, rs = get_info_from_name(simul_name)

        percent = round(100 * i / num_result, 2)
        print(f"Procesing:%{percent}", class_dcbm, case_msd, th, rt, ns, rs, end="\r")

        cv_path = path / "cross_validation"
        muspces_cv_path = cv_path.glob("muspces*.csv")
        for i, mpath in enumerate(sorted(muspces_cv_path)):
            tempDf = pd.read_csv(mpath)
            alpha = float(tempDf["alpha"].iloc[0])
            beta = float(tempDf["beta"].iloc[0])
            result_dict["row"].append(
                (
                    i,
                    class_dcbm,
                    case_msd,
                    th,
                    rt,
                    ns,
                    rs,
                    alpha,
                    beta,
                    "muspces",
                )
            )
            result_dict["val"].append(tempDf[objkey].mean())

        pisces_cv_path = cv_path.glob("pisces*.csv")
        for i, ppath in enumerate(sorted(pisces_cv_path)):
            tempDf = pd.read_csv(ppath)
            alpha = float(tempDf["alpha"].iloc[0])
            beta = np.nan
            result_dict["row"].append(
                (i, class_dcbm, case_msd, th, rt, ns, rs, alpha, beta, "pisces")
            )
            result_dict["val"].append(tempDf[objkey].mean())

    index_row = pd.MultiIndex.from_tuples(
        result_dict["row"],
        names=[
            "id",
            "class-dcbm",
            "case-msd",
            "time-horizon",
            "r_time",
            "num-subjects",
            "r_subject",
            "alpha",
            "beta",
            "method",
        ],
    )
    return get_dataframe(result_dict, index_row, ["loglikelihood"], multi_index)


# }}}


def read_comm_results(results_path, multi_index=False):
    # {{{
    results_path = Path(results_path)
    result_dirs = [f for f in results_path.iterdir() if f.is_dir()]
    result_dict = defaultdict(list)
    num_result = len(result_dirs)

    for i, path in enumerate(sorted(result_dirs)):
        simul_name = str(path.stem)
        class_dcbm, case_msd, th, rt, ns, rs = get_info_from_name(simul_name)

        percent = round(100 * i / num_result, 2)
        print(f"Procesing:%{percent}", class_dcbm, case_msd, th, rt, ns, rs, end="\r")

        communities_path = path / "communities"
        muspces_comm_path = communities_path.glob("muspces*.csv")
        for i, mpath in enumerate(sorted(muspces_comm_path)):
            temp = np.genfromtxt(mpath, delimiter=",")
            result_dict["row"].append(
                (i, class_dcbm, case_msd, th, rt, ns, rs, "muspces")
            )
            result_dict["val"].append(np.mean(temp))

        pisces_comm_path = communities_path.glob("pisces*.csv")
        for i, ppath in enumerate(sorted(pisces_comm_path)):
            temp = np.genfromtxt(ppath, delimiter=",")
            result_dict["row"].append(
                (i, class_dcbm, case_msd, th, rt, ns, rs, "pisces")
            )
            result_dict["val"].append(np.mean(temp))

        static_comm_path = communities_path.glob("static*.csv")
        for i, spath in enumerate(sorted(static_comm_path)):
            temp = np.genfromtxt(spath, delimiter=",")
            result_dict["row"].append(
                (i, class_dcbm, case_msd, th, rt, ns, rs, "static")
            )
            result_dict["val"].append(np.mean(temp))

    index_row = pd.MultiIndex.from_tuples(
        result_dict["row"],
        names=[
            "id",
            "class-dcbm",
            "case-msd",
            "time-horizon",
            "r_time",
            "num-subjects",
            "r_subject",
            "method",
        ],
    )
    return get_dataframe(result_dict, index_row, ["mean(ARI)"], multi_index)


# }}}


def plot_community_detection_results(results_path, figure_path):
    # {{{
    results_df = read_comm_results(results_path)

    unique_col_val = {}
    for col in results_df.columns:
        unique_col_val[col] = sorted(results_df[col].unique())

    x = "time-horizon"
    y = "mean(ARI)"
    hue = "method"
    col = "r_time"
    row = "r_subject"

    output_path = figure_path / "community_detection"
    sutils.safe_create_dir(output_path)

    for class_dcbm in unique_col_val["class-dcbm"]:
        for case_msd in unique_col_val["case-msd"]:
            for ns in unique_col_val["num-subjects"]:
                mask_trip = (
                    (results_df["class-dcbm"] == class_dcbm)
                    & (results_df["case-msd"] == case_msd)
                    & (results_df["num-subjects"] == ns)
                )
                data = results_df[mask_trip]
                title = f"dcbm-class: {class_dcbm} "
                title += f"case-msd: {case_msd} "
                title += f"num-subjects: {str(ns)}"
                g = VIS.point_catplot(data, x, y, hue, col, row, title)
                g.fig.savefig(
                    output_path / f"{class_dcbm}_{case_msd}_ns{ns}.png",
                    bbox_inches="tight",
                )
    # }}}


def plot_cross_validation_results(results_path, figure_path, objkey="loglikelihood"):
    # {{{
    results_df = read_cv_results(results_path)

    unique_col_val = {}
    for col in results_df.columns:
        unique_col_val[col] = sorted(results_df[col].unique())

    sutils.safe_create_dir(figure_path / "cross_validation")

    for class_dcbm in unique_col_val["class-dcbm"]:
        for case_msd in unique_col_val["case-msd"]:
            for ns in unique_col_val["num-subjects"]:
                for th in unique_col_val["time-horizon"]:
                    title = f"dcbm-class: {class_dcbm}, "
                    title += f"case-msd: {case_msd}, "
                    title += f"time horizon: {th}, "
                    title += f"num. of subjects: {ns}, "
                    cvfig_path = figure_path / "cross_validation"
                    dmsnw_name = f"{class_dcbm}_{case_msd}_th{th}_ns{ns}"
                    # == MuSPCES ==
                    mask_trip = (
                        (results_df["class-dcbm"] == class_dcbm)
                        & (results_df["case-msd"] == case_msd)
                        & (results_df["num-subjects"] == ns)
                        & (results_df["time-horizon"] == th)
                    )
                    mthd = "muspces"
                    muspces_mask = mask_trip & (results_df["method"] == mthd)
                    data = results_df[muspces_mask]
                    g = VIS.heatmap_facetgrid(
                        data=data,
                        x="alpha",
                        y="beta",
                        hue=objkey,
                        row="r_subject",
                        col="r_time",
                        title=title + f"method: {mthd}",
                    )
                    g.fig.tight_layout()
                    g.fig.savefig(
                        cvfig_path / f"{dmsnw_name}_{mthd}.png", bbox_inches="tight"
                    )
                    # == PisCES ==
                    mthd = "pisces"
                    muspces_mask = mask_trip & (results_df["method"] == mthd)
                    data = results_df[muspces_mask]
                    g = VIS.point_catplot(
                        data=data,
                        x="alpha",
                        y=objkey,
                        hue="r_time",
                        col="r_subject",
                        title=title + f"method: {mthd}",
                    )
                    g.fig.tight_layout()
                    g.fig.savefig(
                        cvfig_path / f"{dmsnw_name}_{mthd}.png", bbox_inches="tight"
                    )
                    plt.close("all")


# }}}

## plot_community_detection_results(RESULTS_PATH, FIGURE_DIR)
## plot_cross_validation_results(RESULTS_PATH, FIGURE_DIR)
