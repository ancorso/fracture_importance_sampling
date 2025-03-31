using XLSX
using DataFrames
using Plots
using DataStructures
using ExpectationMaximization
using Distributions
using Random
using CSV
using Clustering

include("utils.jl")
Random.seed!(1)

## Iteration 0
filepath_0 = "data/GTF_data.xlsx"
index_start_0 = 1
index_stop_0 = 100
output_path_0 = "outputs/iter0/"
mkpath(output_path_0)
parameters_0, results_0, max_lengths_0, weights_0 = load_samples(filepath_0, index_start=index_start_0, index_stop=index_stop_0, plot_folder=output_path_0, include_worst_case=true)
get_new_samples(parameters_0, results_0, max_lengths_0, weights_0; index_start=index_start_0, index_stop=index_stop_0, n_samples=100, plot_folder=output_path_0, q=0.7)
get_cdf(max_lengths_0, weights_0; n_bootstrap=100, plot_folder=output_path_0)

## Iteration 1
filepath_1 = "data/GTF_data_101_200.xlsx"
index_start_1 = 101
index_stop_1 = 200
output_path_1 = "outputs/iter1/"
mkpath(output_path_1)
parameters_1, results_1, max_lengths_1, weights_1 = load_samples(filepath_1, index_start=index_start_1, index_stop=index_stop_1, plot_folder=output_path_1, weights_file=output_path_0 * "new_samples.csv")

all_parameters = vcat(parameters_0, parameters_1)
all_max_lengths = vcat(max_lengths_0, max_lengths_1)
all_results = hcat(results_0[1:end-1, :], results_1)
all_weights = vcat(weights_0, weights_1)
get_new_samples(all_parameters, all_results, all_max_lengths, all_weights; index_start=index_start_0, index_stop=index_stop_1, n_samples=100, plot_folder=output_path_1, q=0.7)
get_cdf(all_max_lengths, all_weights; n_bootstrap=100, plot_folder=output_path_1)
