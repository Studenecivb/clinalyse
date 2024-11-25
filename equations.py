from cmath import inf
import csv
import math
import multiprocessing
import os
import zipfile
import logging
import timeit

import scipy.special
import scipy.stats as st
from scipy.special import betainc, gamma, erfc, hyp1f1

import numpy as np
import pandas as pd

from clinalyse.fibonacci import _Fibmax

logging.basicConfig(level=logging.INFO)


class Profiler:
    def __init__(self, data: np.array, list_of_fibgrids: list, model: str, names_of_loci=None, path="."):
        self.data = data
        self.names_of_loci = names_of_loci
        self.list_of_fibgrids = list_of_fibgrids
        self.fm = _Fibmax(self.list_of_fibgrids)
        self.profiles = None
        self.path = path
        self.model = model

    def calculate_profiles(self, data: np.array, number_of_processes=4):
        logging.info('Profiles calculations starting.')
        pool = multiprocessing.Pool(number_of_processes)
        if self.data.test is None:
            self.profiles = pool.map(
                _ProfilerSingleLocus.get_1d_profiles,
                [
                    _ProfilerSingleLocus(
                        self.data, self.list_of_fibgrids, self.model, i, self.names_of_loci, self.path)
                    for i in range(len(data.data_labelled_ready) - 1)
                    # for i in range(1)
                ],
            )
        else:
            self.profiles = pool.map(
                _ProfilerSingleLocus.get_1d_profiles,
                [
                    _ProfilerSingleLocus(
                        self.data, self.list_of_fibgrids, self.model, i, self.path
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
        self, data: np.array, list_of_fibgrids: list, model,  locus_idx: int, names_of_loci, path="."
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
        self.names_of_loci = names_of_loci

        if self.model == "gossetbar":
            preparator_g = PreparationGosset(fibgrid_shape=self.list_of_fibgrids[2].grid)
            preparator_g.preparation_gosset()
            self.prep = preparator_g.map_shape_hack_shape

        if self.model == 'gossetbar_asy':
            preparator_g = PreparationGosset(fibgrid_shape=self.list_of_fibgrids[2].grid, fibgrid_asy=self.list_of_fibgrids[3].grid)
            preparator_g.preparation_gosset_asy()
            self.prep = preparator_g.map_shape_hack_asy

        if self.model == 'unit_hl':
            preparator_g = PreparationGosset(fibgrid_shape=self.list_of_fibgrids[5].grid, fibgrid_asy=self.list_of_fibgrids[2].grid)
            preparator_g.preparation_gosset_asy()
            self.prep = preparator_g.map_shape_hack_asy

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
        result = (give_n_trials - given_n_success) * np.log(
            (np.ones(p_hypothesis.shape) - p_hypothesis)
        ) + given_n_success * np.log(p_hypothesis)
        return result

    # @staticmethod
    # def _efficient_bin_log_likelihood(
    #         p_hypothesis: np.ndarray, give_n_trials: np.ndarray, given_n_success: np.ndarray
    # ):
    #     result = (give_n_trials - given_n_success) * np.log(
    #         (np.maximum((np.ones(p_hypothesis.shape) - p_hypothesis), 0.000000000001))
    #     ) + given_n_success * np.log(np.maximum(p_hypothesis, 0.000000000001))
    #     return result

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
    def gossetbar_cline(self, x, shape):
        result = st.t.cdf(x, shape, scale=self.prep[shape])
        return result

    def _gossetbar_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        self.gossetbar_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.geo_at_locus_i, cw[0], cw[1]
                            ), cw[2]
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
                            ), cw[2]
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # MOST COMPLEX MODEL - UNIT HL cline
    # U{l, h}(x) = UN(X + h(UT(X / l) − UT(0)))
    # X = x(l/(h + l))

    def unit_h_l_cline(self, x, asymmetry, height, length, shape):
        big_x = x * (length / (height + length))
        result = self.safe_sigmoid_cline(big_x + height * self.gossetbar_cline_asy(big_x, shape, asymmetry) - self.gossetbar_cline_asy(0, shape, asymmetry))
        return result

    def unit_hl_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        self.unit_h_l_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.geo_at_locus_i, cw[0], cw[1]
                            ), cw[2], cw[3], cw[4], cw[5]
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
                        self.unit_h_l_cline(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            ), cw[2], cw[3], cw[4], cw[5]
                        ),
                        self.data.test[0][1],
                    )
                )
            return real_leastsquared_equation

    # Another Gosset model

    def gossetbar_cline_asy(self, x, shape, asymmetry):
        result = NonCentralStudent.nctas243(x, shape, asymmetry, 1/self.prep[(shape, asymmetry)][0], -self.prep[(shape, asymmetry)][1]/self.prep[(shape, asymmetry)][0])
        return result

    # @staticmethod
    # def gossetbar_cline_asy_preppy(x, shape, asymmetry, prep1, prep2):
    #     result = NonCentralStudent.nctas243(x, shape, asymmetry, 1/prep1, -prep2/prep1)
    #     # result = st.nct.cdf(x, shape, nc=asymmetry, scale=1/self.prep[(shape, asymmetry)][0], loc=-self.prep[(shape, asymmetry)][1]/self.prep[(shape, asymmetry)][0])
    #     result = np.asarray(result, dtype=np.float64)
    #     return result

    def _gossetbar_asy_cline_equations(self):
        if self.data.test is None:
            def real_likelihood_equation(cw):
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        self.gossetbar_cline_asy(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.geo_at_locus_i, cw[0], cw[1]
                            ), cw[2], cw[3]
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
                        self.gossetbar_cline_asy(
                            _ProfilerSingleLocus.safe_locate_n_scale(
                                self.data.test[0][0], cw[0], cw[1]
                            ), cw[2], cw[3]
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
                # print(f'I am width param value:{cw[1]}')
                return sum(
                    _ProfilerSingleLocus._efficient_bin_log_likelihood(
                        _ProfilerSingleLocus.folded_barrier_cline(
                            (
                                _ProfilerSingleLocus.safe_locate_n_scale(
                                    self.geo_at_locus_i, cw[0], 1
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
        logging.info(f'Calculating evaluations for locus {self.names_of_loci[self.locus_idx]}')
        evals = []
        n_axes = len(self.list_of_fibgrids)
        if self.path:
            if self.model == "sigmoid":
                if not os.path.isdir(f"{self.path}/sigmoid_C_evals"):
                    os.mkdir(f"{self.path}/sigmoid_C_evals")
                    os.listdir(self.path)
                f = open(
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
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
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/sigmoid_C_evals/sig_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "gossetbar":
                if not os.path.isdir(f"{self.path}/gossetbar_C_evals"):
                    os.mkdir(f"{self.path}/gossetbar_C_evals")
                f = open(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
                    "w",
                )
                with f:
                    header = ["c-fibgridpos", "w-fibgridpos", "shape-fibgirdpos", f"values"]
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
                                        "c-fibgridpos": self.list_of_fibgrids[0].grid[x[0][0]],
                                        "w-fibgridpos": self.list_of_fibgrids[1].grid[x[0][1]],
                                        "shape-fibgirdpos": self.list_of_fibgrids[2].grid[x[0][2]],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/gossetbar_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "gossetbar_asy":
                if not os.path.isdir(f"{self.path}/gossetbar_asy_C_evals"):
                    os.mkdir(f"{self.path}/gossetbar_asy_C_evals")
                f = open(
                    f"{self.path}/gossetbar_asy_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
                    "w",
                )
                with f:
                    header = ["c-fibgridpos", "w-fibgridpos", "shape-fibgirdpos", "asymmetry-fibgridpos", f"values"]
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
                                        "shape-fibgirdpos": x[0][2],
                                        "asymmetry-fibgridpos": x[0][3],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/gossetbar_asy_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/gossetbar_asy_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/gossetbar_asy_C_evals/gos_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "barrier":
                if not os.path.isdir(f"{self.path}/barrier_C_evals"):
                    os.mkdir(f"{self.path}/barrier_C_evals")
                f = open(
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
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
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/barrier_C_evals/bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/barrier_C_evals/bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "asymmetric":
                if not os.path.isdir(f"{self.path}/asy_C_evals"):
                    os.mkdir(f"{self.path}/asy_C_evals")
                f = open(
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
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
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/asy_C_evals/asy_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/asy_C_evals/asy_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "asymmetric_barrier":
                if not os.path.isdir(f"{self.path}/asy_bar_C_evals"):
                    os.mkdir(f"{self.path}/asy_bar_C_evals")
                f = open(
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
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
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/asy_bar_C_evals/asy_bar_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                )
            if self.model == "unit_hl":
                if not os.path.isdir(f"{self.path}/unit_hl_C_evals"):
                    os.mkdir(f"{self.path}/unit_hl_C_evals")
                f = open(
                    f"{self.path}/unit_hl_C_evals/unit_hl_C_evals_{self.names_of_loci[self.locus_idx]}.csv",
                    "w",
                )
                with f:
                    header = [
                        "c-fibgridpos",
                        "w-fibgridpos",
                        "alpha-fibgridpos",
                        "height-fibgridpos",
                        "length-fibgridpos",
                        "shape-fibgridpos",
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
                                        "height-fibgridpos": x[0][3],
                                        "length-fibgridpos": x[0][4],
                                        "shape-fibgridpos": x[0][5],
                                        "values": x[1],
                                    }
                                )
                f.close()
                with zipfile.ZipFile(
                    f"{self.path}/unit_hl_C_evals/unit_hl_C_evals_{self.names_of_loci[self.locus_idx]}.zip",
                    "w",
                ) as f:
                    f.write(
                        f"{self.path}/unit_hl_C_evals/unit_hl_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
                    )
                os.remove(
                    f"{self.path}/unit_hl_C_evals/unit_hl_C_evals_{self.names_of_loci[self.locus_idx]}.csv"
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
        if self.model == "gossetbar_asy":
            function_to_max = _ProfilerSingleLocus._gossetbar_asy_cline_equations
        if self.model == "barrier":
            function_to_max = _ProfilerSingleLocus._barrier_cline_equations
        if self.model == "asymmetric":
            function_to_max = _ProfilerSingleLocus._asy_cline_equation
        if self.model == "asymmetric_barrier":
            function_to_max = _ProfilerSingleLocus._asy_bar_cline_equation
        if self.model == "unit_hl":
            function_to_max = _ProfilerSingleLocus.unit_hl_cline_equations

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
                        profiles[a_inner][1][e[0][a_inner]] = max(profiles[a_inner][1][e[0][a_inner]], e[1])
        return profiles


class PreparationGosset:
    def __init__(self, fibgrid_shape, fibgrid_asy=None):
        self.fibgrid_shape = fibgrid_shape
        self.fibgrid_asy = fibgrid_asy
        self.fibgrid_shape_res = []
        self.prep_shape = None
        self.map_shape_hack_shape = {}
        self.prep_asy = None
        self.map_shape_hack_asy = {}

    def preparation_gosset(self):
        for i in self.fibgrid_shape:
            result = self._studentcdfgrad(i)
            self.fibgrid_shape_res.append(result)
            self.map_shape_hack_shape[i] = result
        self.prep_shape = [self.fibgrid_shape, self.fibgrid_shape_res]
        return

    def preparation_gosset_asy(self):
        for z in self.fibgrid_shape:
            for y in self.fibgrid_asy:
                g = [gamma((i+z)/2) for i in range(4)]
                gsq = [g[i] ** 2 for i in range(4)]
                asq = y ** 2
                offset = y * math.sqrt(z / 2) * g[2] / g[3]
                nc_student_cdf_gradient_numer = 2 ** -z * math.exp(asq / 2) * math.pi * z ** -(1 + z / 2) * (z + (asq * z * gsq[2]) / (2 * gsq[3])) ** ((1 + z) / 2)
                nc_student_cdf_gradient_demon = g[1] * PreparationGosset.hermite(-1-z, -asq * g[2]/math.sqrt(2 * asq * gsq[2] + 4 * gsq[3]))
                result = nc_student_cdf_gradient_numer/nc_student_cdf_gradient_demon
                self.map_shape_hack_asy[(z, y)] = [result, offset]
        return

    @staticmethod
    def hermite(v, z):
        result = 2 ** v * math.sqrt(math.pi) * (1 / gamma((1 - v) / 2) * hyp1f1(-v / 2, 1 / 2, z ** 2) - 2 * z / gamma(-v / 2) * hyp1f1((1 - v) / 2, 3 / 2, z ** 2))
        return result

    @staticmethod
    def _studentcdfgrad(x):
        result = gamma((x + 1)/2)/(math.sqrt(math.pi * x) * gamma(x / 2))
        return result


class NonCentralStudent:
    def __int__(self, x, v, a, prep, accuracy=pow(10, -3)):
        self.x = x
        self.v = v
        self.a = a
        self.accuracy = accuracy
        self.prep = prep

    @staticmethod
    def helper_nct(i, v, a, accuracy):
        if i >= 0:
            return NonCentralStudent.nctas243_half(i, v, a, accuracy)
        elif i < 0:
            return 1 - NonCentralStudent.nctas243_half(i, v, (-a), accuracy)

    @staticmethod
    def nctas243(x, v, a, prep1, prep2, accuracy=pow(10, -3)):
        if a == 0:
            result = st.t.cdf(x, v, scale=prep1, loc=prep2)
        else:
            xprep = (x - prep2) / prep1
            xprep = np.array(xprep)
            vf = np.vectorize(NonCentralStudent.helper_nct)
            result = vf(xprep, v, a, accuracy)
        return np.minimum(result, 0.999999999999)

    # @staticmethod
    # def nctas243_old(x, v, a, prep1, prep2, accuracy=pow(10, -3)):
    #     result = []
    #     if a == 0:
    #         result.append(st.nct.cdf(x, v, nc=a, scale=prep1, loc=prep2))
    #     else:
    #         xprep = (x - prep2) / prep1
    #         xprep = np.array(xprep)
    #
    #         for i in xprep:
    #             if i >= 0:
    #                 calculation = NonCentralStudent.nctas243_half(i, v, a, accuracy)
    #             elif i < 0:
    #                 calculation = (1 - NonCentralStudent.nctas243_half(i, v, (-a), accuracy))
    #             result.append(calculation)
    #     return result

    @staticmethod
    def nctas243_half(x, v, a, accuracy=pow(10, -3)):
        total_sum = 0
        converged = False
        if x == 0 and v == 0:
            y = 1
        else:
            y = (pow(x, 2)/(pow(x, 2) + v))
        vo2 = v/2
        j = 0
        a1 = j + 1/2
        a2 = j + 1
        a3 = j + 3/2
        iya1 = betainc(a1, vo2, y)
        iya2 = betainc(a2, vo2, y)
        ga1p1 = gamma((a1+1))
        ga2 = gamma(a2)
        ga2p1 = gamma((a2 + 1))
        ga1pvo2 = gamma((a1 + vo2))
        ga2pvo2 = gamma((a2+vo2))
        gvo2 = gamma(vo2)
        ga3 = gamma(a3)

        while not converged:
            old_sum = total_sum
            total_sum += 0.5 * math.exp(-(pow(a, 2)/2)) * pow((pow(a, 2)/2), j) * ((1/ga2)*iya1+(a/(math.sqrt(2)*ga3))*iya2)
            converged = abs((old_sum - total_sum)/total_sum) <= accuracy
            if not converged:
                iya1 = iya1 - (ga1pvo2/(ga1p1*gvo2))*(pow(y, a1))*(pow((1-y), vo2))
                iya2 = iya2 - (ga2pvo2 / (ga2p1 * gvo2)) * (pow(y, a2)) * (pow((1 - y), vo2))
                ga1p1 = (a1 + 1)*ga1p1
                ga2 = a2*ga2
                ga2p1 = (a2+1)*ga2p1
                ga1pvo2 = (a1 + vo2)*ga1pvo2
                ga2pvo2 = (a2 + vo2)*ga2pvo2
                ga3 = a3*ga3
                j += 1
                a1 += 1
                a2 += 1
                a3 += 1
        return 0.5*erfc(a/math.sqrt(2))+total_sum

