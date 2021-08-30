import yaml
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path


MAIN_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = MAIN_DIR / "data"


if __name__ == "__main__":
    import sys

    sys.path.append(str(MAIN_DIR))


import dypoces.utils.sutils as sutils  # noqa: E402
import dypoces.nw as nw  # noqa: E402


class GCNW:
    @staticmethod
    def get_missing_genes(gene_info):
        missing_gene_dict = defaultdict(list)
        for _, nw_name in enumerate(gene_info.columns):
            missing_gene_dict[nw_name] = gene_info.index[
                gene_info[nw_name] == 0
            ].tolist()

        lacking_nw_dict = defaultdict(list)
        for gene, row in gene_info.iterrows():
            lacking_nw_dict[gene] = gene_info.columns[row == 0]

        return missing_gene_dict, lacking_nw_dict

    @staticmethod
    def get_common_genes(lacking_nw_dict):
        common_genes, uncommon_genes = [], []
        for gene, networks in lacking_nw_dict.items():
            if len(networks) == 0:
                common_genes.append(gene)
            else:
                uncommon_genes.append(gene)
        return common_genes, uncommon_genes


class CellSpecificGCNW(GCNW):
    def __init__(self, data_dir_path, cell_name, p=90):
        self.cell_name = cell_name
        self.p = p
        self.data_dir_path = data_dir_path
        self.cell_dir_path, self.gcnw_dir_path, self.adj_dir_path = self.get_paths()
        self.adj_p_dir_path = self.adj_dir_path / f"p{self.p}"

    def get_paths(self):
        cell_dir_path = (
            self.data_dir_path / "cell-type-specific" / f"{self.cell_name}-networks"
        )
        gcnw_dir_path = cell_dir_path / "gcnw"
        adj_dir_path = cell_dir_path / "adj"
        return cell_dir_path, gcnw_dir_path, adj_dir_path

    @staticmethod
    def threshold_coexpression_nw(gcnw: np.ndarray, p=95):
        q_threshold = np.percentile(gcnw, p)
        adj_thresholded = np.where(gcnw > q_threshold, 1, 0)
        return adj_thresholded

    @staticmethod
    def save_unweighted_undirected_adj(adj: np.ndarray, adj_path):
        adj = np.int8(adj)
        adj = np.triu(adj, 1) + np.triu(adj, 1).T
        np.save(adj_path, adj)

    @staticmethod
    def parse_nw_name(name):
        nw_info = name.split("__")
        assert len(nw_info) == 3
        [donor_ID, day_name, treatment] = nw_info
        return donor_ID, day_name, treatment

    @staticmethod
    def get_gene_index_map(adj_gene_index):
        for row, col in adj_gene_index:
            assert row == col
        for idx, (row, _) in enumerate(adj_gene_index[1:]):
            assert adj_gene_index[idx][0] == row
        for idx, (_, col) in enumerate(adj_gene_index[1:]):
            assert adj_gene_index[idx][1] == col
        genes = adj_gene_index[0][0]
        gene_index_map = {i: genes[i] for i in range(len(genes))}
        return gene_index_map

    def retain_genes(self, opt):
        gene_info_path = self.gcnw_dir_path / "info" / f"{self.cell_name}-genes.csv"
        gene_info = pd.read_csv(gene_info_path)

        if opt == "common":
            _, lacking_nw_dict = self.__class__.get_missing_genes(gene_info)
            common_genes, uncommon_genes = self.__class__.get_common_genes(
                lacking_nw_dict
            )
            retained_genes = common_genes
            excluded_genes = uncommon_genes
        else:
            raise ValueError(
                f"Unkown option {opt} is given to determine retained genes."
            )

        return retained_genes, excluded_genes

    def retain_days(self, opt):
        with open(self.cell_dir_path / "days.yaml", "r") as outfile:
            days_dict = yaml.safe_load(outfile)
        if opt == "all":
            retained_days = list(days_dict.keys())
            excluded_days = []
        else:
            raise ValueError(
                f"Unkown option {opt} is given to determine retained days."
            )

        return retained_days, excluded_days

    def construct_msdyn_nw(self, opt="all", treatment="NONE"):
        retained_days, excluded_days = self.retain_days(opt)
        with open(self.cell_dir_path / "donors.yaml", "r") as outfile:
            donor_dict = yaml.safe_load(outfile)
        with open(self.cell_dir_path / "treatments.yaml", "r") as outfile:
            treatment_dict = yaml.safe_load(outfile)

        donor_indices = {}
        excluded_donors = []
        msdyn_nw = []

        for i, (donor, day_nw_dict) in enumerate(
            dict(sorted(donor_dict.items())).items()
        ):
            if set(retained_days) == set(day_nw_dict.keys()):
                lacks_gene = False
                dyn_nw = []
                for j, (day, nw_name_list) in enumerate(
                    dict(sorted(day_nw_dict.items())).items()
                ):
                    gene_at_t = False
                    for nw_name in nw_name_list:
                        if nw_name in treatment_dict[treatment]:
                            adj_path = self.adj_p_dir_path / f"{nw_name}.npy"
                            dyn_nw.append(np.load(adj_path))
                            gene_at_t = True
                        else:
                            pass
                    lacks_gene = not gene_at_t or lacks_gene
                if not lacks_gene:
                    sutils.log(
                        f"{nw_name} appears in all retained days, hence included..."
                    )
                    msdyn_nw.append(np.array(dyn_nw))
                    donor_indices[i] = donor
                else:
                    excluded_donors.append(donor)
                    sutils.log(
                        f"{nw_name} is absent in some of the retained days, hence excluded..."
                    )

        msdyn_nw = np.array(msdyn_nw)
        np.save(self.adj_p_dir_path / "msdyn_nw.npy", msdyn_nw)

        msdyn_nw_details = {"option": opt}
        msdyn_nw_details["day_indices"] = {i: t for i, t in enumerate(retained_days)}
        msdyn_nw_details["donor_indices"] = donor_indices
        msdyn_nw_details["exluded_days"] = excluded_days
        msdyn_nw_details["excluded_donors"] = excluded_donors
        msdyn_nw_details_path = self.adj_p_dir_path / "msdyn_nw_details.yaml"
        sutils.ensure_file_dir(msdyn_nw_details_path)
        with open(msdyn_nw_details_path, "w") as outfile:
            yaml.dump(msdyn_nw_details, outfile)

    def construct_networks(self, opt="common"):
        cell_gcnws_paths = [f for f in self.gcnw_dir_path.iterdir() if not f.is_dir()]
        cell_nw_dict = defaultdict(dict)
        donor_dict = defaultdict(lambda: defaultdict(list))
        day_dict = defaultdict(lambda: defaultdict(list))
        treatment_dict = defaultdict(list)

        retained_genes, excluded_genes = self.retain_genes(opt)
        excluded_nw = []
        adj_gene_index = []

        for i, gcnw_path in enumerate(sorted(cell_gcnws_paths)):
            nw_name = str(gcnw_path.stem)
            donor_ID, day_name, treatment = self.__class__.parse_nw_name(nw_name)

            nw_weighted_df = pd.read_csv(gcnw_path, sep=None, engine="python")
            nw_weighted_df.sort_index(axis=1, inplace=True)
            nw_weighted_df.sort_index(axis=0, inplace=True)
            gene_names = nw_weighted_df.columns

            cell_nw_dict[nw_name] = {
                "donor_ID": donor_ID,
                "day_name": day_name,
                "treatment": treatment,
                "genes": list(gene_names),
            }
            donor_dict[donor_ID][day_name].append(nw_name)
            day_dict[day_name][donor_ID].append(nw_name)
            treatment_dict[treatment].append(nw_name)

            if (set(retained_genes) <= set(nw_weighted_df.columns)) and (
                set(retained_genes) <= set(nw_weighted_df.index)
            ):
                nw_weighted_df = nw_weighted_df[retained_genes].loc[retained_genes]
                adj_gene_index.append(
                    (list(nw_weighted_df.index), list(nw_weighted_df.columns))
                )
                nw_weighted = nw_weighted_df.to_numpy()
                adj = self.__class__.threshold_coexpression_nw(nw_weighted, p=self.p)

                adj_path = self.adj_p_dir_path / nw_name
                sutils.ensure_file_dir(adj_path)
                self.__class__.save_unweighted_undirected_adj(adj, adj_path)
                info_txt = nw.inform_about_network(adj)
                sutils.log(f"{nw_name} contains all retained genes, hence included...")
                sutils.log(f"{nw_name} ~ {info_txt}")
            else:
                excluded_nw.append(nw_name)
                sutils.log(f"{nw_name} lack some retained genes, hence excluded...")

        adj_details = {"option": opt}
        adj_details["excluded_networks"] = excluded_nw
        adj_details["excluded_genes"] = excluded_genes
        adj_details["gene_indices"] = self.__class__.get_gene_index_map(adj_gene_index)
        with open(self.adj_p_dir_path / "adj_details.yaml", "w") as outfile:
            yaml.dump(adj_details, outfile, sort_keys=False)

        def dd_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: dd_to_dict(v) for k, v in d.items()}
            return d

        for filename, dictionary in [
            ("networks.yaml", cell_nw_dict),
            ("donors.yaml", donor_dict),
            ("days.yaml", day_dict),
            ("treatments.yaml", treatment_dict),
        ]:
            outpath = self.cell_dir_path / filename
            sutils.ensure_file_dir(outpath)
            with open(outpath, "w") as outfile:
                yaml.dump(dd_to_dict(dictionary), outfile)


if __name__ == "__main__":
    cell_name = "Epen1"
    percentile = 95
    csGCNW = CellSpecificGCNW(DATA_DIR, "Epen1", p=percentile)
    csGCNW.construct_networks()
    csGCNW.construct_msdyn_nw()
