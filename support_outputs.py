from cmath import inf
from itertools import product
import os
import logging

import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt
from more_itertools import locate
import numpy as np
import pandas as pd
from scipy.stats import chi2

from clinalyse.equations import _ProfilerSingleLocus

matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO)


class Graphs:
    def __init__(
        self, profiler, profiles: np.array, list_of_fibgrids: list, data: np.array, model: str,
    ):
        self.list_of_fibgrids = list_of_fibgrids
        self.data = data
        self.profiler = profiler
        self.profiles = profiles
        self.model = model

    # Graphing function for parameter support
    def graphing_of_parameter_support(self, path="."):
        logging.info('Computing graphs of parameter support.')
        if path:
            if not os.path.isdir(f"{path}/parameter_support"):
                os.mkdir(f"{path}/parameter_support")
        for i in range(len(self.profiles[0])):
            x_values = []
            y_values = []
            fig_i, ax_i = plt.subplots()
            ax_i.set_xlabel(f"Parameter {i+1}")
            ax_i.set_ylabel("support log likelihood")
            ax_i.set_ylim(bottom=-10)
            ax_i.set_title(
                f"Parameter {i+1} on x axis and support log likelihood on y axis"
            )
            for profile_i in self.profiles:
                max_value_for_i = np.max(profile_i[i][1])
                support_for_par_i_prof_i = [
                    (x - max_value_for_i) for x in profile_i[i][1]
                ]
                ax_i.plot(
                    self.list_of_fibgrids[i].grid[profile_i[i][0]],
                    support_for_par_i_prof_i,
                )
                x_values.append(self.list_of_fibgrids[i].grid[profile_i[i][0]])
                y_values.append(support_for_par_i_prof_i)

            fig = ax_i.figure
            if path:
                fig.savefig(f"{path}/parameter_support/graph_for_parameter_{i+1}_{self.model}.png")
                if not os.path.isdir(f"{path}/graph_values"):
                    os.mkdir(f"{path}/graph_values")
                df = pd.DataFrame(
                    y_values,
                    columns=x_values[0],
                )
                df.to_csv(
                    f"{path}/graph_values/val_parameter_support_graph_param_{i+1}_{self.model}.csv"
                )
            plt.close()
        return

    @staticmethod
    def sig_cline_y_values(x_value_range, mles, intervals):
        y_value_range = _ProfilerSingleLocus.safe_sigmoid_cline(
            (x_value_range - mles[0]) / mles[1]
        )
        try:
            y_value_range_min_sig = np.minimum(
                np.minimum(
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][0]
                    ),
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][-1]
                    ),
                ),
                np.minimum(
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][0]
                    ),
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][-1]
                    ),
                ),
            )
            y_value_range_max_sig = np.maximum(
                np.maximum(
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][0]
                    ),
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][-1]
                    ),
                ),
                np.maximum(
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][0]
                    ),
                    _ProfilerSingleLocus.safe_sigmoid_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][-1]
                    ),
                ),
            )
            return [y_value_range, y_value_range_min_sig, y_value_range_max_sig]
        except IndexError:
            y_value_range_min_sig = y_value_range * np.nan
            y_value_range_max_sig = y_value_range * np.nan
            return [y_value_range, y_value_range_min_sig, y_value_range_max_sig]

    def gossetbar_cline_y_values(self, x_value_range, mles, intervals):
        y_value_range = self.profiler.gossetbar_cline(
            (x_value_range - mles[0]) / mles[1]
        )
        try:
            y_value_range_min_sig = np.minimum(
                np.minimum(
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][0]
                    ),
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][-1]
                    ),
                ),
                np.minimum(
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][0]
                    ),
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][-1]
                    ),
                ),
            )
            y_value_range_max_sig = np.maximum(
                np.maximum(
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][0]
                    ),
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][0]) / intervals[1][-1]
                    ),
                ),
                np.maximum(
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][0]
                    ),
                    self.profiler.gossetbar_cline(
                        (x_value_range - intervals[0][-1]) / intervals[1][-1]
                    ),
                ),
            )
            return [y_value_range, y_value_range_min_sig, y_value_range_max_sig]
        except IndexError:
            y_value_range_min_sig = y_value_range * np.nan
            y_value_range_max_sig = y_value_range * np.nan
            return [y_value_range, y_value_range_min_sig, y_value_range_max_sig]

    @staticmethod
    def bar_cline_y_values(x_value_range, mles, intervals):
        product_create = product(range(2), repeat=4)
        y_value_range_bar = _ProfilerSingleLocus.folded_barrier_cline(
            ((x_value_range - mles[0]) / mles[1]), mles[2], mles[3]
        )
        y_value_range_min_bar = inf
        y_value_range_max_bar = -inf
        for i in list(product_create):
            i = [x * (-1) for x in i]
            try:
                y_value_range_min_bar = np.minimum(
                    _ProfilerSingleLocus.folded_barrier_cline(
                        ((x_value_range - intervals[0][i[0]]) / intervals[1][i[1]]),
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                    ),
                    y_value_range_min_bar,
                )
                y_value_range_max_bar = np.maximum(
                    _ProfilerSingleLocus.folded_barrier_cline(
                        (x_value_range - intervals[0][i[0]]) / intervals[1][i[1]],
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                    ),
                    y_value_range_max_bar,
                )
            except IndexError:
                y_value_range_min_bar = y_value_range_bar * np.nan
                y_value_range_max_bar = y_value_range_bar * np.nan
        return [y_value_range_bar, y_value_range_min_bar, y_value_range_max_bar]

    @staticmethod
    def asy_cline_y_values(x_value_range, mles, intervals):
        product_thingy = product(range(2), repeat=4)
        y_value_range_asy = _ProfilerSingleLocus.folded_asymmetric_cline(
            ((x_value_range - mles[0]) / mles[1]), mles[2], mles[3]
        )
        y_value_range_min_asy = inf
        y_value_range_max_asy = -inf
        for i in list(product_thingy):
            i = [x * (-1) for x in i]
            try:
                y_value_range_min_asy = np.minimum(
                    _ProfilerSingleLocus.folded_asymmetric_cline(
                        ((x_value_range - intervals[0][i[0]]) / intervals[1][i[1]]),
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                    ),
                    y_value_range_min_asy,
                )
                y_value_range_max_asy = np.maximum(
                    _ProfilerSingleLocus.folded_asymmetric_cline(
                        (x_value_range - intervals[0][i[0]]) / intervals[1][i[1]],
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                    ),
                    y_value_range_max_asy,
                )
            except IndexError:
                y_value_range_min_asy = y_value_range_asy * np.nan
                y_value_range_max_asy = y_value_range_asy * np.nan
        return [y_value_range_asy, y_value_range_min_asy, y_value_range_max_asy]

    @staticmethod
    def asy_bar_cline_y_values(x_value_range, mles, intervals):
        product_thingy = product(range(2), repeat=5)
        y_value_range_bar = _ProfilerSingleLocus.folded_asy_bar_cline(
            ((x_value_range - mles[0]) / mles[1]), mles[2], mles[3], mles[4]
        )
        y_value_range_min_bar = inf
        y_value_range_max_bar = -inf
        for i in list(product_thingy):
            i = [x * (-1) for x in i]
            try:
                y_value_range_min_bar = np.minimum(
                    _ProfilerSingleLocus.folded_asy_bar_cline(
                        ((x_value_range - intervals[0][i[0]]) / intervals[1][i[1]]),
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                        intervals[4][i[4]]
                    ),
                    y_value_range_min_bar,
                )
                y_value_range_max_bar = np.maximum(
                    _ProfilerSingleLocus.folded_asy_bar_cline(
                        (x_value_range - intervals[0][i[0]]) / intervals[1][i[1]],
                        intervals[2][i[2]],
                        intervals[3][i[3]],
                        intervals[4][i[4]]
                    ),
                    y_value_range_max_bar,
                )
            except IndexError:
                y_value_range_min_bar = y_value_range_bar * np.nan
                y_value_range_max_bar = y_value_range_bar * np.nan
        return [y_value_range_bar, y_value_range_min_bar, y_value_range_max_bar]

    def cline_graph(
        self,
        locus_number: int,
        x_start: float,
        x_end: float,
        x_grain: float,
        rect=None,
        path=".",
    ):
        x_value_range = np.arange(x_start, x_end, x_grain, dtype=float)
        mles = []
        intervals = []
        logging.info(f'Creating cline graphs for locus {locus_number+1}')
        if path:
            if not os.path.isdir(f"{path}/{self.model}_cline"):
                os.mkdir(f"{path}/{self.model}_cline")
        for i in range(len(self.profiles[0])):
            max_value_for_i = np.max(self.profiles[locus_number][i][1])
            support_for_par_i_prof_i = [
                (x - max_value_for_i) for x in self.profiles[locus_number][i][1]
            ]
            value_of_mle = self.list_of_fibgrids[i].grid[
                Support.find_indices_mle(support_for_par_i_prof_i, 0)
            ]
            mles.append(value_of_mle)
            interval_of_llus_2 = self.list_of_fibgrids[i].grid[
                Support.llus_intervals(support_for_par_i_prof_i, -2)
            ]
            intervals.append(interval_of_llus_2)

        if self.model == "sigmoid":
            y_values = Graphs.sig_cline_y_values(x_value_range, mles, intervals)
        elif self.model == "gossetbar":
            y_values = self.gossetbar_cline_y_values(x_value_range, mles, intervals)
        elif self.model == "barrier":
            y_values = Graphs.bar_cline_y_values(x_value_range, mles, intervals)
        elif self.model == "asymmetric":
            y_values = Graphs.asy_cline_y_values(x_value_range, mles, intervals)
        else:
            y_values = Graphs.asy_bar_cline_y_values(x_value_range, mles, intervals)
        plt.title(
            f"{self.model} cline fit with a 2llunit support for a locus {self.data.names_of_loci[locus_number]}"
        )
        plt.plot(x_value_range, y_values[0], label="Sup0")
        plt.plot(x_value_range, y_values[1], label="x-Gridded Mins over Sup-2")
        plt.plot(x_value_range, y_values[2], label="x-Gridded Maxes over Sup-2")
        plt.fill_between(
            x_value_range,
            y_values[1],
            y_values[2],
            color="lightgray",
        )
        if np.isnan(y_values[1]).any():
            plt.text(
                x_value_range[0] + 10,
                y_values[0][0],
                "WARNING:The search spaces for parameters not wide enough, only the MLE cline fit shown:",
                size="x-small",
            )
        plt.legend()
        if rect:
            rect = mpatches.Rectangle(
                (rect[0], rect[1]),
                rect[2],
                rect[3],
                fill=False,
                color="red",
                linewidth=2,
            )
            plt.gca().add_patch(rect)
        if path:
            values_of_cline = {
                "x_values": x_value_range,
                "y_values": y_values[0],
                "y_value_range_min": y_values[1],
                "y_value_range_max": y_values[2],
            }
            df = pd.DataFrame.from_dict(values_of_cline)
            df.to_csv(
                f"{path}/{self.model}_cline/graph_values_for_locus_{locus_number+1}.csv",
                index=False,
            )
            plt.savefig(
                f"{path}/{self.model}_cline/{self.model}_cline_for_locus_{self.data.names_of_loci[locus_number]}.png"
            )
        plt.close()


class Support:
    def __init__(
        self,
        profiles: np.array,
        list_of_fibgrids: list,
        data: np.array,
        model: str,
        path=".",

    ):
        self.list_of_fibgrids = list_of_fibgrids
        self.profiles = profiles
        self.data = data
        self.support_for_all_sigmoid = None
        self.support_for_all_gossetbar = None
        self.support_for_all_barrier = None
        self.support_for_all_asymmetric = None
        self.support_for_all_asymmetric_barrier = None
        self.path = path
        self.model = model

    # Estimated support calculations
    @staticmethod
    def find_indices_mle(list_to_check, item_to_find):
        indices = locate(list_to_check, lambda x: x == item_to_find)
        return list(indices)

    @staticmethod
    def llus_intervals(support, support_unit_threshold):
        intervals = []
        interval_start_idx = None
        last_seen_above_threshold_idx = None
        in_interval = False
        for i, v in enumerate(support):
            if v > support_unit_threshold:
                if not in_interval:
                    interval_start_idx = i
                    in_interval = True
                last_seen_above_threshold_idx = i
            else:
                if in_interval:
                    intervals.append(interval_start_idx)
                    intervals.append(last_seen_above_threshold_idx)
                    in_interval = False
        return intervals

    def estimate_support(self):
        logging.info('Estimating parameter support.')
        support_for_all_sigmoid = []
        support_for_all_gossetbar = []
        support_for_all_barrier = []
        support_for_all_asymmetric = []
        support_for_all_asymmetric_barrier = []
        if self.path:
            if not os.path.isdir(f"{self.path}/estimated_support"):
                os.mkdir(f"{self.path}/estimated_support")
        for i in range(len(self.profiles[0])):
            support_for_i = []
            mle_of_i = []
            llus_2_i = []
            llus_3_i = []
            for profile_i in self.profiles:
                max_value_for_i = np.max(profile_i[i][1])
                support_for_par_i_prof_i = [
                    (x - max_value_for_i) for x in profile_i[i][1]
                ]
                support_for_i.append(support_for_par_i_prof_i)
                value_of_mle = self.list_of_fibgrids[i].grid[
                    Support.find_indices_mle(support_for_par_i_prof_i, 0)
                ]
                mle_of_i.append(value_of_mle)
                interval_of_llus_2 = self.list_of_fibgrids[i].grid[
                    Support.llus_intervals(support_for_par_i_prof_i, -2)
                ]
                llus_2_i.append(interval_of_llus_2)
                interval_of_llus_3 = self.list_of_fibgrids[i].grid[
                    Support.llus_intervals(support_for_par_i_prof_i, -3)
                ]
                llus_3_i.append(interval_of_llus_3)
            # csv for a parameter
            if self.model == "sigmoid":
                support_for_all_sigmoid.append(support_for_i)
            if self.model == "gossetbar":
                support_for_all_gossetbar.append(support_for_i)
            if self.model == "barrier":
                support_for_all_barrier.append(support_for_i)
            if self.model == "asymmetric":
                support_for_all_asymmetric.append(support_for_i)
            if self.model == "asymmetric_barrier":
                support_for_all_asymmetric_barrier.append(support_for_i)
            list_of_i = {
                "Locus": self.data.names_of_loci,
                "Parameter": (i + 1),
                "MLE": mle_of_i,
                "2 Log Likelihood Unit Support": llus_2_i,
                "3 Log Likelihood Unit Support": llus_3_i,
            }
            df = pd.DataFrame.from_dict(list_of_i)
            df["MLE"] = df["MLE"].str[0]
            if self.path:
                df.to_csv(
                    f"{self.path}/estimated_support/Estimated support_parameter_{i+1}_{self.model}.csv",
                    index=False,
                )
        self.support_for_all_sigmoid = support_for_all_sigmoid
        self.support_for_all_gossetbar = support_for_all_gossetbar
        self.support_for_all_barrier = support_for_all_barrier
        self.support_for_all_asymmetric_barrier = support_for_all_asymmetric_barrier
        self.support_for_all_asymmetric = support_for_all_asymmetric


class Hypotheses:
    # User tests:
    # H1: Locus x is different from locus y with parameter z.
    # H2: Locus x is a stepped cline.
    # H1:
    @staticmethod
    def loci_different(
        locus_1_idx: int, locus_2_idx: int, support: list, par1=bool, par2=bool
    ):
        support_for_param_x = None
        if len(support) == 2:
            if par1:
                support_for_param_x = support[0]
            if par2:
                support_for_param_x = support[1]
        first_locus = support_for_param_x[locus_1_idx]
        second_locus = support_for_param_x[locus_2_idx]
        combined_loci = [
            first_locus[i] + second_locus[i] for i in range(len(first_locus))
        ]
        delta_llu = np.abs(np.max(combined_loci))
        df = 1
        p_value = 1 - chi2.cdf(2 * delta_llu, df)
        return print(f"For the hypothesis 1 of different loci p-value is: {p_value}")

    # if less than alpha (eg 0.05) they are different

    # H2:
    @staticmethod
    def locus_stepped(
        locus_idx: int,
        support_for_one: list,
        support_for_two: list,
        par1=bool,
        par2=bool,
        compare_all=False,
    ):
        support_two = None
        support_one = None
        if par1:
            support_one = support_for_one[0][locus_idx]
            support_two = support_for_two[0][locus_idx]
        if par2:
            support_one = support_for_one[1][locus_idx]
            support_two = support_for_two[1][locus_idx]

        combined_support = [
            support_one[i] + support_two[i] for i in range(len(support_one))
        ]
        delta_llu = np.abs(np.max(combined_support))
        df = 2
        p_value = 1 - chi2.cdf(2 * delta_llu, df)
        if not compare_all:
            print(f"For the hypothesis 2 of stepped locus p-value is: {p_value}")
        return p_value

    # if less than alpha (eg 0.05) it is a candidate for a stepped cline
