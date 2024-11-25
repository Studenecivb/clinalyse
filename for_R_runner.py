import csv

import numpy as np
import numpy.core.defchararray as np_f

import clinalyse


def create_fibgrids(start, end, step):
    return clinalyse.FibGrid(start, end, step)


def sigmoid(input_data_path, ploidy_data_path, my_path, fg_c, fg_w, names_of_loci_pos, geno_pos):
    if __name__ == "__main__":
        names_of_loci_pos = int(names_of_loci_pos)
        geno_pos = int(geno_pos)
        print(names_of_loci_pos, geno_pos)
        input_data = []
        with open(input_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                input_data.append(row)
        cleaned_data = np_f.replace(np.transpose(input_data[1:]), "NO_K_DATA_ID", "nan")
        cleaned_data = np_f.replace(cleaned_data, "kIDbutNoK", "nan")
        cleaned_data = np_f.replace(cleaned_data, "NA", "nan")
        names_of_loci = input_data[0][names_of_loci_pos:-1]

        ploidy_data = []
        with open(ploidy_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                ploidy_data.append(row)
        clean_ploidy = np.transpose(ploidy_data[1:])

        d = clinalyse.InputData(cleaned_data, names_of_loci, clean_ploidy, geno_pos=geno_pos)
        d.load_data()

        profiler_s = clinalyse.Profiler(d, [fg_c, fg_w], model="sigmoid", path=my_path, names_of_loci=names_of_loci)
        profiler_s.calculate_profiles(d, number_of_processes=4)
        profiler_s.profiles_save_into_csv(path=my_path)
        grapher_s = clinalyse.Graphs(profiler=profiler_s, profiles=profiler_s.profiles, list_of_fibgrids=[fg_c, fg_w], data=d, model="sigmoid")
        grapher_s.graphing_of_parameter_support(path=my_path)
        for i in range(len(profiler_s.profiles)):
            grapher_s.cline_graph(i, -150, 150, 0.1, path=my_path)
        supporter_s = clinalyse.Support(
            profiles=profiler_s.profiles, list_of_fibgrids=[fg_c, fg_w], data=d, model="sigmoid", path=my_path)
        supporter_s.estimate_support()


def barrier(input_data_path, ploidy_data_path, my_path, fg_c, fg_w, fg_b, fg_gamma, names_of_loci_pos, geno_pos):
    if __name__ == "__main__":
        names_of_loci_pos = int(names_of_loci_pos)
        geno_pos = int(geno_pos)
        print(names_of_loci_pos, geno_pos)
        input_data = []
        with open(input_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                input_data.append(row)
        cleaned_data = np_f.replace(np.transpose(input_data[1:]), "NO_K_DATA_ID", "nan")
        cleaned_data = np_f.replace(cleaned_data, "kIDbutNoK", "nan")
        cleaned_data = np_f.replace(cleaned_data, "NA", "nan")
        names_of_loci = input_data[0][names_of_loci_pos:-1]

        ploidy_data = []
        with open(ploidy_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                ploidy_data.append(row)
        clean_ploidy = np.transpose(ploidy_data[1:])

        d = clinalyse.InputData(cleaned_data, names_of_loci, clean_ploidy, geno_pos=geno_pos)
        d.load_data()

        profiler_b = clinalyse.Profiler(d, [fg_c, fg_w, fg_b, fg_gamma], model="barrier", path=my_path, names_of_loci=names_of_loci)
        profiler_b.calculate_profiles(d, number_of_processes=4)
        profiler_b.profiles_save_into_csv(path=my_path)
        grapher_b = clinalyse.Graphs(profiler=profiler_b, profiles=profiler_b.profiles, list_of_fibgrids=[fg_c, fg_w, fg_b, fg_gamma], data=d, model="barrier")
        grapher_b.graphing_of_parameter_support(path=my_path)
        for i in range(len(profiler_b.profiles)):
            grapher_b.cline_graph(i, -150, 150, 0.1, path=my_path)
        supporter_b = clinalyse.Support(
            profiles=profiler_b.profiles, list_of_fibgrids=[fg_c, fg_w, fg_b, fg_gamma], data=d, model="barrier", path=my_path)
        supporter_b.estimate_support()


def asymmetric(input_data_path, ploidy_data_path, my_path, fg_c, fg_w, fg_a, fg_gamma, names_of_loci_pos, geno_pos):
    if __name__ == "__main__":
        names_of_loci_pos = int(names_of_loci_pos)
        geno_pos = int(geno_pos)
        print(names_of_loci_pos, geno_pos)
        input_data = []
        with open(input_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                input_data.append(row)
        cleaned_data = np_f.replace(np.transpose(input_data[1:]), "NO_K_DATA_ID", "nan")
        cleaned_data = np_f.replace(cleaned_data, "kIDbutNoK", "nan")
        cleaned_data = np_f.replace(cleaned_data, "NA", "nan")
        names_of_loci = input_data[0][names_of_loci_pos:-1]

        ploidy_data = []
        with open(ploidy_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                ploidy_data.append(row)
        clean_ploidy = np.transpose(ploidy_data[1:])

        d = clinalyse.InputData(cleaned_data, names_of_loci, clean_ploidy, geno_pos=geno_pos)
        d.load_data()

        profiler_asy = clinalyse.Profiler(d, [fg_c, fg_w, fg_a, fg_gamma], model="asymmetric", path=my_path)
        profiler_asy.calculate_profiles(d)
        profiler_asy.profiles_save_into_csv(path=my_path)
        grapher_asy = clinalyse.Graphs(profiler_asy, profiler_asy.profiles, [fg_c, fg_w, fg_a, fg_gamma], d, model="asymmetric")
        grapher_asy.graphing_of_parameter_support(path=my_path)
        for i in range(len(profiler_asy.profiles)):
            grapher_asy.cline_graph(i, -100, 100, 0.1, path=my_path)
        supporter_asy = clinalyse.Support(
            profiler_asy.profiles,
            [fg_c, fg_w, fg_a, fg_gamma],
            d,
            model="asymmetric",
            path=my_path,)
        supporter_asy.estimate_support()


def asymmetric_barrier(input_data_path, ploidy_data_path, my_path, fg_c, fg_w, fg_a, fg_b, fg_gamma, names_of_loci_pos, geno_pos):
    if __name__ == "__main__":
        names_of_loci_pos = int(names_of_loci_pos)
        geno_pos = int(geno_pos)
        print(names_of_loci_pos, geno_pos)
        input_data = []
        with open(input_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                input_data.append(row)
        cleaned_data = np_f.replace(np.transpose(input_data[1:]), "NO_K_DATA_ID", "nan")
        cleaned_data = np_f.replace(cleaned_data, "kIDbutNoK", "nan")
        cleaned_data = np_f.replace(cleaned_data, "NA", "nan")
        names_of_loci = input_data[0][names_of_loci_pos:-1]

        ploidy_data = []
        with open(ploidy_data_path, "r") as file:
            csvreader = csv.reader(file, delimiter=",")
            for row in csvreader:
                ploidy_data.append(row)
        clean_ploidy = np.transpose(ploidy_data[1:])

        d = clinalyse.InputData(cleaned_data, names_of_loci, clean_ploidy, geno_pos=geno_pos)
        d.load_data()

        profiler_asy_bar = clinalyse.Profiler(
            d, [fg_c, fg_w, fg_a, fg_b, fg_gamma], model="asymmetric_barrier", path=my_path)
        profiler_asy_bar.calculate_profiles(d, number_of_processes=4)
        profiler_asy_bar.profiles_save_into_csv(path=my_path)
        grapher_asy_bar = clinalyse.Graphs(profiler_asy_bar, profiler_asy_bar.profiles,
            [fg_c, fg_w, fg_a, fg_b, fg_gamma],
            d,
            model="asymmetric_barrier",)
        grapher_asy_bar.graphing_of_parameter_support(path=my_path)
        for i in range(len(profiler_asy_bar.profiles)):
            grapher_asy_bar.cline_graph(i, -100, 100, 0.1, path=my_path)
        supporter_asy_bar = clinalyse.Support(
            profiler_asy_bar.profiles,
            [fg_c, fg_w, fg_a, fg_b, fg_gamma],
            d,
            model="asymmetric_barrier",
            path=my_path)
        supporter_asy_bar.estimate_support()




