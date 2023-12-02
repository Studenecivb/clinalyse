from cmath import inf
import csv
import math
import multiprocessing
import os
import zipfile
import logging
import scipy.stats as st
import scipy.special as ss

import numpy as np
import pandas as pd

from clinalyse.fibonacci import _Fibmax

logging.basicConfig(level=logging.INFO)


class Profiler:
    def __init__(self, data: np.array, list_of_fibgrids: list, model: str, path=".", shape=None):
        self.data = data
        self.list_of_fibgrids = list_of_fibgrids
        self.fm = _Fibmax(self.list_of_fibgrids)
        self.profiles = None
        self.path = path
        self.model = model
        self.shape = shape

    def calculate_profiles(self, data: np.array, number_of_processes=4):
        logging.info('Profiles calculations starting.')
        pool = multiprocessing.Pool(number_of_processes)
        if self.data.test is None:
            self.profiles = pool.map(
                _ProfilerSingleLocus.get_1d_profiles,
                [
                    _ProfilerSingleLocus(
                        self.data, self.list_of_fibgrids, self.model, i, self.path, self.shape
                    )
                    for i in range(len(data.data_labelled_ready) - 1)
                    # for i in range(1)
                ],
            )
        else:
            self.profiles = pool.map(
                _ProfilerSingleLocus.get_1d_profiles,
                [
                    _ProfilerSingleLocus(
                        self.data, self.list_of_fibgrids, self.model, i, self.path, self.shape
                    )
                    for i in range(len(data.test))
                ],
            )

    # We want to save the profiles outputs into a csv
    def profiles_save_into_csv(self, path="."):
        logging.info('Saving profiles.')
        if not os.path.isdir(f"{path}/profiles"):
            os.mkdir(f"{path}/profiles")
        for i in range(len(self.profiles[0])):
            parameter_values = []
            likelihood_values = []
            for profile_i in self.profiles:
                parameter_values_i = (
                    self.list_of_fibgrids[i].grid[profile_i[i][0]].round(2)
                )
                likelihood_values_i = profile_i[i][1]
                parameter_values.append(parameter_values_i)
                likelihood_values.append(likelihood_values_i)
            # saving parameter values
            df = pd.DataFrame(
                likelihood_values,
                columns=parameter_values[0],
                index=self.data.names_of_loci,
            )
            df.to_csv(
                f"{path}/profiles/profiles_parameter_{i+1}_{self.model}.csv", index=True
            )
        return


class _ProfilerSingleLocus:
    max_mag = 9
    # safe_p_lower_bound = safe_sigmoid_cline(-2 * max_mag)
    # safe_p_upper_bound = safe_sigmoid_cline(2 * max_mag)
    safe_p_lower_bound = 2.319522830243569e-16
    safe_p_upper_bound = 0.9999999999999998

    def __init__(
        self, data: np.array, list_of_fibgrids: list, model,  locus_idx: int, path=".", shape=None,
    ):
        self.geo_at_locus_i = np.concatenate(
            np.hsplit(self._grab_data(data.data_labelled_ready, locus_idx), 2)[1], axis=0
        )
        self.geno_at_locus_i = np.concatenate(
            np.hsplit(self._grab_data(data.data_labelled_ready, locus_idx), 2)[0], axis=0
        )
        # print(f'geno: {self.geno_at_locus_i}')
        self.ploidy_at_i = np.concatenate(
            np.hsplit(self._grab_data(data.ploidy_ready, locus_idx), 2)[0], axis=0
        )
        # print(f'ploidy: {self.ploidy_at_i}')
        self.list_of_fibgrids = list_of_fibgrids
        self.fm = _Fibmax(self.list_of_fibgrids)
        self.data = data
        self.locus_idx = locus_idx
        self.path = path
        self.model = model
        self.shape = shape
        self.prep = self._preparation_gosset()

    def _preparation_gosset(self):
        if self.shape:
            result = self._studentcdfgrad()
            return [self.shape, result]
        else:
            return None

    @staticmethod
    def _grab_data(data: np.array, locus_idx: int):
        gege = np.dstack(
            (
                data[locus_idx].astype(float),
                data[-1].astype(float),
            )
        )[0]
        masker = gege[:, 0]
        return gege[~np.isnan(masker)]

    # helper functions:
    @staticmethod
    def safe_exp(x):
        return np.exp(np.minimum(np.maximum(x, - 4 * _ProfilerSingleLocus.max_mag), 4 * _ProfilerSingleLocus.max_mag))

    @staticmethod
    def safe_tanh(x):
        return np.tanh(np.minimum(np.maximum(x, -2 * _ProfilerSingleLocus.max_mag), 2 * _ProfilerSingleLocus.max_mag))

    @staticmethod
    def safe_locate_n_scale(x, c, w):
        x_c = x - c
        if w != 0:
            result = x_c / w
        else:
            return_array = np.zeros(len(x))
            return_array = np.where(x_c < 0, -_ProfilerSingleLocus.max_mag, return_array)
            return_array = np.where(x_c > 0, _ProfilerSingleLocus.max_mag, return_array)
            result = return_array
        return result

    @staticmethod
    def _efficient_bin_log_likelihood(
        p_hypothesis: np.ndarray, give_n_trials: np.ndarray, given_n_success: np.ndarray
    ):
        # given_n_trials_array = np.full(p_hypothesis.shape, give_n_trials)
        result = (give_n_trials - given_n_success) * np.log(
            (np.ones(p_hypothesis.shape) - p_hypothesis)
        ) + given_n_success * np.log(p_hypothesis)
        return result

    @staticmethod
    def _negative_squared_distance(p_hypothesis: np.ndarray, p_observed: np.ndarray):
        result = -np.power(p_hypothesis - p_observed, 2)
        return result

    # First model - sigmoid
    @staticmethod
    def safe_sigmoid_cline(x):
        result = 1 / (1 + _ProfilerSingleLocus.safe_exp(-4 * x))
        return result

    def _sigmoid_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        _ProfilerSingleLocus.safe_sigmoid_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.geo_at_locus_i, cw[0], cw[1]
                            )
                        ),
                        self.ploidy_at_i,
                        self.geno_at_locus_i,
                    )
                )
            return real_likelihood_equation
        else:
            def real_leastsquared_equation(cw):
                return sum(
                    self.data.test[0][2]
                    * _ProfilerSingleLocus._negative_squared_distance(
                        _ProfilerSingleLocus.safe_sigmoid_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            )
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # Additional model - Gossetbar and all its little
    def _studentcdfgrad(self):
        result = ss.gamma((self.shape + 1)/2)/(math.sqrt(math.pi * self.shape) * ss.gamma(self.shape / 2))
        return result

    def gossetbar_cline(self, x):
        result = st.t.cdf(x, self.prep[0], scale=self.prep[1])
        return result

    def _gossetbar_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        self.gossetbar_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.geo_at_locus_i, cw[0], cw[1]
                            )
                        ),
                        self.ploidy_at_i,
                        self.geno_at_locus_i,
                    )
                )
            return real_likelihood_equation
        else:
            def real_leastsquared_equation(cw):
                return sum(
                    self.data.test[0][2]
                    * _ProfilerSingleLocus._negative_squared_distance(
                        self.gossetbar_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            )
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # Second model - barrier
    @staticmethod
    def safe_beta_condition(beta, gamma):
        if np.absolute(beta) <= math.e:
            return 0
        else:
            return math.e / np.log(
                (-np.sign(gamma) * beta + math.e / 2)
                / (-np.sign(gamma) * beta - math.e / 2)
            )

    @staticmethod
    # Second model - barrier cline
    def folded_barrier_cline(x, beta, gamma):
        if beta == 0 or gamma == 0:
            if gamma == 0:
                beta = -beta
            else:
                pass
            return 1 / (
                1 + _ProfilerSingleLocus.safe_exp(2 * (-2 * x + beta * -np.sign(x)))
            )
        else:
            if (0 <= gamma < _ProfilerSingleLocus.safe_beta_condition(beta, gamma)) or (
                _ProfilerSingleLocus.safe_beta_condition(beta, gamma) < gamma < 0
            ):
                x = -x
            else:
                pass

            if gamma <= 0:
                beta = -beta
            else:
                pass
            return 1 / (
                1
                + _ProfilerSingleLocus.safe_exp(
                    -2 * (2 * x + beta * _ProfilerSingleLocus.safe_tanh(2 * x / gamma))
                )
            )

    def _barrier_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        _ProfilerSingleLocus.folded_barrier_cline(
                            (
                                _ProfilerSingleLocus.safe_locate_n_scale(
                                    self.geo_at_locus_i, cw[0], cw[1]
                                )
                            ),
                            cw[2],
                            cw[3],
                        ),
                        self.ploidy_at_i,
                        self.geno_at_locus_i,
                    )
                )
            return real_likelihood_equation
        else:
            def real_leastsquared_equation(cw):
                return sum(
                    self.data.test[0][2]
                    * _ProfilerSingleLocus._negative_squared_distance(
                        _ProfilerSingleLocus.folded_barrier_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            ),
                            cw[2],
                            cw[3],
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # Third model - asymmetric
    @staticmethod
    def safe_tail(p):
        return np.minimum(np.maximum(p, _ProfilerSingleLocus.safe_p_lower_bound), _ProfilerSingleLocus.safe_p_upper_bound)

    @staticmethod
    def folded_asymmetric_cline(x, alpha, gamma):
        if alpha >= 2:
            alpha = -alpha
        else:
            pass
        if gamma == 0:
            return 1 / (1 + _ProfilerSingleLocus.safe_exp(-2 * x * (2 - alpha)))
        else:
            return _ProfilerSingleLocus.safe_tail(1 / (
                1
                + _ProfilerSingleLocus.safe_exp(-2 * x * (2 - alpha))
                * np.power(
                    (2 / (1 + (_ProfilerSingleLocus.safe_exp(4 * x / gamma)))),
                    (gamma * alpha),
                )
            ))

    def _asy_cline_equation(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        _ProfilerSingleLocus.folded_asymmetric_cline(
                            (
                                _ProfilerSingleLocus.safe_locate_n_scale(
                                    self.geo_at_locus_i, cw[0], cw[1]
                                )
                            ),
                            cw[2],
                            cw[3],
                        ),
                        self.ploidy_at_i,
                        self.geno_at_locus_i,
                    )
                )
            return real_likelihood_equation
        else:
            def real_leastsquared_equation(cw):
                return sum(
                    self.data.test[0][2]
                    * _ProfilerSingleLocus._negative_squared_distance(
                        _ProfilerSingleLocus.folded_asymmetric_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            ), cw[2],
                            cw[3]
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # Fourth model - asymmetric barrier
    @staticmethod
    def folded_asy_bar_cline(x, alpha, beta, gamma):
        if beta == 0 and alpha >= 2:
            alpha = -alpha
        else:
            pass

        if gamma == 0:
            beta = -beta
            z = 4 * x + 2 * beta * np.sign(x)
            return 1 / (1 + _ProfilerSingleLocus.safe_exp((z * (alpha - 2)) / 2))
        else:
            if (0 <= gamma < _ProfilerSingleLocus.safe_beta_condition(beta, gamma)) or (
                _ProfilerSingleLocus.safe_beta_condition(beta, gamma) < gamma < 0
            ):
                x = -x
            else:
                pass
            if gamma <= 0:
                beta = -beta
            else:
                pass
            z = 4 * x + 2 * beta * _ProfilerSingleLocus.safe_tanh((2 * x) / gamma)
            return _ProfilerSingleLocus.safe_tail(1 / (
                1
                + _ProfilerSingleLocus.safe_exp((z * (alpha - 2)) / 2)
                * np.power(
                    (2 / (1 + _ProfilerSingleLocus.safe_exp(z / gamma))),
                    (gamma * alpha),
                )
            ))

    def _asy_bar_cline_equation(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        _ProfilerSingleLocus.folded_asy_bar_cline(
                            (
                                _ProfilerSingleLocus.safe_locate_n_scale(
                                    self.geo_at_locus_i, cw[0], cw[1]
                                )
                            ),
                            cw[2],
                            cw[3],
                            cw[4]
                        ),
                        self.ploidy_at_i,
                        self.geno_at_locus_i,
                    )
                )

            return real_likelihood_equation
        else:
            def real_leastsquared_equation(cw):
                return sum(
                    self.data.test[0][2]
                    * _ProfilerSingleLocus._negative_squared_distance(
                        _ProfilerSingleLocus.folded_asy_bar_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            ), cw[2],
                            cw[3],
                            cw[4]
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    def _get_evals(self, function_to_max):
        logging.info(f'Calculating evaluations for locus {self.locus_idx+1}')
        evals = []
        n_axes = len(self.list_of_fibgrids)
        if self.path:
            if self.model == "sigmoid":
                if not os.path.isdir(f"{self.path}/sigmoid_C_evals"):
                    os.mkdir(f"{self.path}/sigmoid_C_evals")
                f = open(
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.locus_idx+1}.csv",
                    "w",
                )
                with f:
                    header = ["c-fibgridpos", "w-fibgridpos", f"values"]
                    writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    for a in range(n_axes):
                        for v_i in range(len(self.list_of_fibgrids[a].grid)):
                            evals_i, best = self.fm.fibmax(
                                function_to_max(self), fix_axis=(a, v_i)
                            )
                            evals.append(evals_i)
                            for x in evals_i:
                                writer.writerow(
                                    {
                                        "c-fibgridpos": x[0][0],
                                        "w-fibgridpos": x[0][1],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.locus_idx+1}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.locus_idx+1}.csv"
                    )
                os.remove(
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.locus_idx+1}.csv"
                )
            if self.model == "gossetbar":
                if not os.path.isdir(f"{self.path}/gossetbar_C_evals"):
                    os.mkdir(f"{self.path}/gossetbar_C_evals")
                f = open(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.locus_idx+1}.csv",
                    "w",
                )
                with f:
                    header = ["c-fibgridpos", "w-fibgridpos", f"values"]
                    writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    for a in range(n_axes):
                        for v_i in range(len(self.list_of_fibgrids[a].grid)):
                            evals_i, best = self.fm.fibmax(
                                function_to_max(self), fix_axis=(a, v_i)
                            )
                            evals.append(evals_i)
                            for x in evals_i:
                                writer.writerow(
                                    {
                                        "c-fibgridpos": x[0][0],
                                        "w-fibgridpos": x[0][1],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.locus_idx+1}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.locus_idx+1}.csv"
                    )
                os.remove(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.locus_idx+1}.csv"
                )
            if self.model == "barrier":
                if not os.path.isdir(f"{self.path}/barrier_C_evals"):
                    os.mkdir(f"{self.path}/barrier_C_evals")
                f = open(
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.locus_idx+1}.csv",
                    "w",
                )
                with f:
                    header = [
                        "c-fibgridpos",
                        "w-fibgridpos",
                        "beta-fibgridpos",
                        "cw-fibgridpos",
                        f"values",
                    ]
                    writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    for a in range(n_axes):
                        for v_i in range(len(self.list_of_fibgrids[a].grid)):
                            evals_i, best = self.fm.fibmax(
                                function_to_max(self), fix_axis=(a, v_i)
                            )
                            evals.append(evals_i)
                            for x in evals_i:
                                writer.writerow(
                                    {
                                        "c-fibgridpos": x[0][0],
                                        "w-fibgridpos": x[0][1],
                                        "beta-fibgridpos": x[0][2],
                                        "cw-fibgridpos": x[0][3],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.locus_idx+1}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/barrier_C_evals/bar_C_evals_{self.locus_idx+1}.csv"
                    )
                os.remove(
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.locus_idx+1}.csv"
                )
            if self.model == "asymmetric":
                if not os.path.isdir(f"{self.path}/asy_C_evals"):
                    os.mkdir(f"{self.path}/asy_C_evals")
                f = open(
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.locus_idx+1}.csv",
                    "w",
                )
                with f:
                    header = [
                        "c-fibgridpos",
                        "w-fibgridpos",
                        "alpha-fibgridpos",
                        "cw-fibgridpos",
                        f"values",
                    ]
                    writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    for a in range(n_axes):
                        for v_i in range(len(self.list_of_fibgrids[a].grid)):
                            evals_i, best = self.fm.fibmax(
                                function_to_max(self), fix_axis=(a, v_i)
                            )
                            evals.append(evals_i)
                            for x in evals_i:
                                writer.writerow(
                                    {
                                        "c-fibgridpos": x[0][0],
                                        "w-fibgridpos": x[0][1],
                                        "alpha-fibgridpos": x[0][2],
                                        "cw-fibgridpos": x[0][3],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.locus_idx+1}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/asy_C_evals/asy_C_evals_{self.locus_idx+1}.csv"
                    )
                os.remove(
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.locus_idx+1}.csv"
                )
            if self.model == "asymmetric_barrier":
                if not os.path.isdir(f"{self.path}/asy_bar_C_evals"):
                    os.mkdir(f"{self.path}/asy_bar_C_evals")
                f = open(
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.locus_idx+1}.csv",
                    "w",
                )
                with f:
                    header = [
                        "c-fibgridpos",
                        "w-fibgridpos",
                        "alpha-fibgridpos",
                        "beta-fibgridpos",
                        "cw-fibgridpos",
                        f"values",
                    ]
                    writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    for a in range(n_axes):
                        for v_i in range(len(self.list_of_fibgrids[a].grid)):
                            evals_i, best = self.fm.fibmax(
                                function_to_max(self), fix_axis=(a, v_i)
                            )
                            evals.append(evals_i)
                            for x in evals_i:
                                writer.writerow(
                                    {
                                        "c-fibgridpos": x[0][0],
                                        "w-fibgridpos": x[0][1],
                                        "alpha-fibgridpos": x[0][2],
                                        "beta-fibgridpos": x[0][3],
                                        "cw-fibgridpos": x[0][4],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.locus_idx+1}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.locus_idx+1}.csv"
                    )
                os.remove(
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.locus_idx+1}.csv"
                )
            return evals
        else:
            for a in range(n_axes):
                for v_i in range(len(self.list_of_fibgrids[a].grid)):
                    evals_i, best = self.fm.fibmax(
                        function_to_max(self), fix_axis=(a, v_i)
                    )
                    evals.append(evals_i)
            return evals

    def get_1d_profiles(self):
        profiles = []
        n_axes = len(self.list_of_fibgrids)
        function_to_max = None
        if self.model == "sigmoid":
            function_to_max = _ProfilerSingleLocus._sigmoid_cline_equations
        if self.model == "gossetbar":
            function_to_max = _ProfilerSingleLocus._gossetbar_cline_equations
        if self.model == "barrier":
            function_to_max = _ProfilerSingleLocus._barrier_cline_equations
        if self.model == "asymmetric":
            function_to_max = _ProfilerSingleLocus._asy_cline_equation
        if self.model == "asymmetric_barrier":
            function_to_max = _ProfilerSingleLocus._asy_bar_cline_equation

        for a in range(n_axes):
            profiles.append(
                (
                    [0] * (len(self.list_of_fibgrids[a].grid)),
                    [-inf] * (len(self.list_of_fibgrids[a].grid)),
                )
            )

        evals = self._get_evals(function_to_max)

        for a in range(n_axes):
            for v_i in range(len(self.list_of_fibgrids[a].grid)):
                profiles[a][0][v_i] = v_i

            for evals_i in evals:
                for e in evals_i:
                    for a_inner in range(n_axes):
                        profiles[a_inner][1][e[0][a_inner]] = np.maximum(
                            profiles[a_inner][1][e[0][a_inner]], e[1]
                        )
        return profiles
