using XLSX
using DataFrames
using Plots
using DataStructures
using ExpectationMaximization
using Distributions
using Random
using CSV
using Clustering

Random.seed!(1)

outputfolder = "outputs/new_samples_101-200/"
mkpath(outputfolder)

## Load all of the data
filepath = "data/GTF_data.xlsx"
parameters = DataFrame(XLSX.readtable(filepath, "Parameters", "B:L", first_row=3))
ranges = OrderedDict(
    "K" => (50, 500),
    "TEC" => (4.07e-6, 8.5e-6),
    "PR" => (0.2, 0.4),
    "YM" => (10600000, 14400000),
    "Biot" => (0.5, 0.8),
    "K1C" => (433.125, 1299.5),
    "Shmin" => (357, 465),
    "Tres" => (85.5, 99.5),
    "Pres" => (225.1, 250.2),
    "KvKh" => (0.5, 1),
    "FcK" => (-2, 0),
)
prior = product_distribution([Uniform(v[1], v[2]) for v in values(ranges)])

# Assert that the order of everything is the same
for (k1, k2) in zip(names(parameters), keys(ranges))
    @assert k1 == k2
end

results = DataFrame(XLSX.readtable(filepath, "Results", first_row=3))
worst_case_params = DataFrame(XLSX.readtable(filepath, "Worst case scenario", "B:L", first_row=3))
worst_case_results = DataFrame(XLSX.readtable(filepath, "Worst case scenario", "O:P", first_row=2))

## Plot the results
function plot_results(;plot_worst_case=false)
    p = plot(ylabel="Crack Half Length (m)", xlabel="Time (days)")
    for i=1:Int(size(results,2) / 2)
        plot!(results[!, "Time_case_$i"], results[!, "Half_length_case$i"], color=:black, alpha=0.2, label="")
    end
    if plot_worst_case
        plot!(worst_case_results[!, "Time"], worst_case_results[!, "Half length"], color=:red, label="Worst Case")
    end 
    return p
end
plot_results(plot_worst_case=true)
savefig(outputfolder * "prior_results.png")

## Plot the parameters
function plot_parameters(parameters; plot_worst_case=false, plots=OrderedDict(k => plot() for k in keys(ranges)), kwargs...)
    for k in keys(ranges)
        p = plots[k]
        histogram!(p, parameters[!, k], xlabel=k, label="", bins = range(ranges[k]..., length=20), normalize=true; kwargs...)
        if plot_worst_case
            vline!(p, worst_case_params[!, k], label="", linewidth=5)
        end
    end
    
    p = plot(values(plots)..., size=(1500, 1200))
    p, plots
end
params_plot = plot_parameters(parameters, plot_worst_case=true)[1]
savefig(outputfolder * "prior_parameters.png")

## Filter the top quantile and produce a dataset
# Get the maximum half length for each case
max_lengths = Float64[]
for i=1:Int(size(results,2) / 2)
    push!(max_lengths, maximum(results[!, "Half_length_case$i"]))
end

# Find the 75th percentile threshold
q = 0.7
length_threshold = quantile(max_lengths, q)

# Get indices of cases above threshold
top_cases = findall(x -> x >= length_threshold, max_lengths)

# Extract parameters for those cases
top_parameters = parameters[top_cases, :]

# plot the quantile on the results plots
plot_results()
hline!([length_threshold], color=:red, label="$q quantile")
savefig(outputfolder * "quantile_results.png")

p1, plots = plot_parameters(parameters, alpha=0.3, label="Prior")
p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
p2
savefig(outputfolder * "quantile_parameters.png")

## Fit a distribution to the dataset
d = fit_mle(MvNormal, collect(Matrix{Float64}(top_parameters)'))

################# The code below fits a mixture model, but often hits posdef issues
# # Cluster the top_parameters
# N_clusters = 2
# cluster_features = collect(Matrix{Float64}(top_parameters)')
# cluster_result = kmeans(cluster_features, N_clusters); # run K-means for the 3 clusters

# # Fit MvNormals to each of the clusters
# # Initialize array to store fitted distributions
# cluster_dists = MvNormal[]

# # For each cluster, fit a multivariate normal distribution
# for i in 1:N_clusters
#     # Get indices of points in this cluster
#     cluster_indices = findall(x -> x == i, cluster_result.assignments)
    
#     # Get the data points for this cluster
#     cluster_data = cluster_features[:, cluster_indices]
    
#     # Fit MvNormal to the cluster data
#     cluster_dist = fit_mle(MvNormal, cluster_data)
#     push!(cluster_dists, cluster_dist)
# end
# guess = MixtureModel(cluster_dists, fill(1/N_clusters, N_clusters))
# d = fit_mle(guess, collect(Matrix{Float64}(top_parameters)'))

# Plot some samples 
raw_samples = rand(d, 100000)
raw_samples_df = DataFrame(raw_samples', names(parameters))
p1, plots = plot_parameters(parameters, alpha=0.3, label="Prior")
p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
p3, plots = plot_parameters(raw_samples_df; plots, alpha=0.3, label="New Raw Samples", bins=20)
p3
savefig(outputfolder * "new_raw_samples_parameters.png")

## Generate some new samples via rejection sampling
function reject(sample, prior)
    pdf(prior, sample) == 0
end

# draw a sample from `raw_dist` but rejected everyone pdf(prior, sample) == 0
function rejection_sample(raw_dist, prior; max_tries=100000)
    count = 0
    while count < max_tries
        sample = rand(raw_dist)
        if reject(sample, prior)
            count += 1
            continue
        end
        return sample
    end
end

rejection_samples = [rejection_sample(d, prior) for _ in 1:100]
rejection_samples_df = DataFrame(hcat(rejection_samples...)', names(parameters))
p1, plots = plot_parameters(parameters, alpha=0.3, label="Prior")
p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
p3, plots = plot_parameters(rejection_samples_df; plots, alpha=0.3, label="New Samples (rejected)")
p3
savefig(outputfolder * "new_rejection_samples_parameters.png")

## Compute the normalization constant and weights of the samples
# Normalization constant of the rejection-sampled distribution
N = mean([!reject(d, prior) for d in eachcol(raw_samples)])

# Returns the logpdf of the sample under the prior (uniform distribution )
function logpdf_rejection_dist(raw_dist, prior, sample; N)
    if reject(sample, prior)
        return -Inf
    end

    return logpdf(raw_dist, sample) - log(N)
end

function weight(sample, raw_dist, prior; N)
    logq = logpdf_rejection_dist(raw_dist, prior, sample; N)
    logp = logpdf(prior, sample)
    return exp(logp - logq)
end

weights = [weight(sample, d, prior; N) for sample in rejection_samples]
histogram(weights, xlabel="Weights", label="")
savefig(outputfolder * "new_weights.png")

## write everything to disk
results = deepcopy(rejection_samples_df)
results.weights = weights
CSV.write(outputfolder * "samples_101-200.csv", results)
