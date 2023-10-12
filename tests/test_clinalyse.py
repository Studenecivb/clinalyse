import unittest
import zipfile

import numpy as np
import numpy.core.defchararray as np_f
import csv
from csv_diff import load_csv, compare

import clinalyse

fg_c = clinalyse.FibGrid(-35, +35, 1)
fg_w = clinalyse.FibGrid(0, 141, 1)
# fg_b = clinalyse.FibGrid(-2.2, 3.2, 0.015)
# fg_a = clinalyse.FibGrid(-2.2, 2.2, 0.01)
# fg_gamma = clinalyse.FibGrid(-0.5, 3, 0.015)

input_data = []
with open("test_geno.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        input_data.append(row)
cleaned_data = np_f.replace(np.transpose(input_data[1:]), "NO_K_DATA_ID", "nan")
cleaned_data = np_f.replace(cleaned_data, "kIDbutNoK", "nan")
cleaned_data = np_f.replace(cleaned_data, "NA", "nan")
names_of_loci = input_data[0][1:]

ploidy_data = []
with open("test_ploidy.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        ploidy_data.append(row)
file.close()
clean_ploidy = np.transpose(ploidy_data[1:])

d = clinalyse.InputData(cleaned_data, names_of_loci, clean_ploidy, geno_pos=1)
d.load_data()


class Test(unittest.TestCase):
    def test_evals(self):
        print('Evals checks')
        profiler_s = clinalyse.Profiler(d, [fg_c, fg_w], model="sigmoid", path="./results")
        profiler_s.calculate_profiles(d, number_of_processes=1)
        with zipfile.ZipFile("./results/sigmoid_C_evals/sig_C_evals_1.zip", "r") as zip_ref:
            zip_ref.extractall("./compare_res")
        with zipfile.ZipFile("./expected/sigmoid_C_evals/sig_C_evals_1.zip", "r") as zip_ref:
            zip_ref.extractall("./compare_exp")
        diff = compare(
            load_csv(open("./compare_exp/clinalyse/tests/expected/sigmoid_C_evals/sig_C_evals_1.csv")),
            load_csv(open("./compare_res/clinalyse/tests/results/sigmoid_C_evals/sig_C_evals_1.csv"))
        )
        self.assertTrue(diff)
        print('Evals check done')

    def test_profiles(self):
        print('Starting profiles check.')
        profiles_exp = []
        with open("./expected/profiles/profiles_parameter_1_sigmoid.csv", "r") as fil1:
            csvreader1 = csv.reader(fil1, delimiter=",")
            for row1 in csvreader1:
                profiles_exp.append(row1)
        fil1.close()
        with open("./expected/profiles/profiles_parameter_2_sigmoid.csv", "r") as fil1:
            csvreader1 = csv.reader(fil1, delimiter=",")
            for row1 in csvreader1:
                profiles_exp.append(row1)
        fil1.close()
        profiler_s = clinalyse.Profiler(d, [fg_c, fg_w], model="sigmoid", path="./results")
        profiler_s.calculate_profiles(d, number_of_processes=1)
        profiles_res = profiler_s.profiles
        profiles_exp_1 = [float(x) for x in profiles_exp[1][1:]]
        profiles_exp_2 = [float(x) for x in profiles_exp[3][1:]]
        self.assertEqual(profiles_res[0][0][1], profiles_exp_1)
        self.assertEqual(profiles_res[0][1][1], profiles_exp_2)
        print('Profiles check done.')

    def test_cline(self):
        print('Graph checks')
        profiler_s = clinalyse.Profiler(d, [fg_c, fg_w], model="sigmoid", path="./results")
        profiler_s.calculate_profiles(d, number_of_processes=1)
        grapher_s = clinalyse.Graphs(profiler_s.profiles, [fg_c, fg_w], d, model="sigmoid")
        grapher_s.graphing_of_parameter_support(path="./results")
        grapher_s.cline_graph(0, -150, 150, 0.1, path="./results")
        diff = compare(
            load_csv(open("./expected/sigmoid_cline/graph_values_for_locus_1.csv")),
            load_csv(open("./results/sigmoid_cline/graph_values_for_locus_1.csv"))
        )
        self.assertTrue(diff)
        print('Graph check done')

if __name__ == "__main__":
    unittest.main()


