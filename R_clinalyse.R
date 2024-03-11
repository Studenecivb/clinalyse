install.packages('reticulate')
library(reticulate)
os <- import("os")
os$listdir(".")


py_install('clinalyse')
clinalyse <- import("clinalyse")

source_python('for_R_runner.py')

#data1 <- read_csv('input_data.csv',show_col_types = FALSE)
#data_ploidy <- read_csv('ploidy.csv',show_col_types = FALSE)
#id_list <-seq(from = 13, by = 1, length.out = 92)
#names_of_loci <- colnames(data1)[id_list]

my_path <- "./R_trial"
input_data_path <-"C:/NINA/MUNI/BIOMB/Master/master_thesis/data/input_data.csv"
ploidy_data_path <- "C:/NINA/MUNI/BIOMB/Master/master_thesis/data/ploidy.csv"

#replace_vals <- c('NO_K_DATA_ID', 'kIDbutNoK')
#replacement_values <- c('nan', 'nan')

#data1$Kdata <- replace(data1$Kdata, data1$Kdata %in% replace_vals, "nan") 

#def sigmoid(input_data, ploidy_data, my_path, fg_c, fg_w, names_of_loci, geno_pos):

fg_c <- create_fibgrids(-50, 50, 1)
fg_w <- create_fibgrids(0, 100, 1)
fg_b <- create_fibgrids(-2.2, 3.2, 0.015)
fg_a <- create_fibgrids(-2.2, 2.2, 0.01)
fg_gamma <- create_fibgrids(-0.5, 3, 0.015)

sigmoid(input_data_path, ploidy_data_path, my_path, fg_c, fg_w, names_of_loci_pos=12, geno_pos=12)

