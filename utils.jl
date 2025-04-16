const prior_ranges = OrderedDict(
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
const PRIOR = product_distribution([Uniform(v[1], v[2]) for v in values(prior_ranges)])


function plot_results(results; worst_case_results=nothing, index_start, index_stop)
    p = plot(ylabel="Crack Half Length (m)", xlabel="Time (days)")
    for i = index_start:index_stop
        plot!(results[!, "Time_case_$i"], results[!, "Half_length_case$i"], color=:black, alpha=0.2, label="")
    end
    if !isnothing(worst_case_results)
        plot!(worst_case_results[!, "Time"], worst_case_results[!, "Half length"], color=:red, label="Worst Case")
    end
    return p
end

function plot_parameters(parameters; worst_case_params=nothing, plots=OrderedDict(k => plot() for k in keys(prior_ranges)), kwargs...)
    for k in keys(prior_ranges)
        p = plots[k]
        histogram!(p, parameters[!, k], xlabel=k, label="", bins=range(prior_ranges[k]..., length=20), normalize=true; kwargs...)
        if !isnothing(worst_case_params)
            vline!(p, worst_case_params[!, k], label="", linewidth=5)
        end
    end

    p = plot(values(plots)..., size=(1500, 1200))
    p, plots
end

function load_samples(filepath; index_start, index_stop, plot_folder=nothing, include_worst_case=false, weights_file=nothing)
    parameters = DataFrame(XLSX.readtable(filepath, "Parameters", "B:L", first_row=3))
    # Assert that the order of everything is the same
    for (k1, k2) in zip(names(parameters), keys(prior_ranges))
        @assert k1 == k2
    end

    # Read the xlsx file for the simulation results
    results = DataFrame(XLSX.readtable(filepath, "Results", first_row=3))

    # Get the maximum half length for each case
    max_lengths = Float64[]
    for i = index_start:index_stop
        push!(max_lengths, maximum(results[!, "Half_length_case$i"]))
    end

    # Check if the results have a weights column
    if isnothing(weights_file)
        weights = ones(length(max_lengths))
    else
        df = CSV.read(weights_file, DataFrame)
        # check that the parameters are the same
        for name in names(parameters)
            @assert all(df[!, name] .≈ parameters[!, name])
        end
        weights = df[!, :weights]
        weights = clamp.(weights, 0, 1)
    end


    worst_case_params = nothing
    worst_case_results = nothing
    if include_worst_case
        # Load the worst case scenario
        worst_case_params = DataFrame(XLSX.readtable(filepath, "Worst case scenario", "B:L", first_row=3))
        worst_case_results = DataFrame(XLSX.readtable(filepath, "Worst case scenario", "O:P", first_row=2))
    end

    if !isnothing(plot_folder)
        plot_results(results, worst_case_results=worst_case_results; index_start, index_stop)
        savefig(plot_folder * "results.png")

        plot_parameters(parameters, worst_case_params=worst_case_params)[1]
        savefig(plot_folder * "parameters.png")
    end

    return parameters, results, max_lengths, weights
end


function bootstrap_cdf(max_lengths, weights)
    bootstrap_indices = rand(1:length(max_lengths), length(max_lengths))
    bootstrap_max_lengths = max_lengths[bootstrap_indices]
    bootstrap_weights = weights[bootstrap_indices]

    sorted_indices = sortperm(bootstrap_max_lengths, rev=true)
    bootstrap_max_lengths = bootstrap_max_lengths[sorted_indices]
    bootstrap_weights = bootstrap_weights[sorted_indices] ./ sum(bootstrap_weights)

    bootstrap_cdf_vec = 1 .- cumsum(bootstrap_weights)
    return bootstrap_max_lengths, bootstrap_cdf_vec
end


function get_cdf(max_lengths, weights; n_bootstrap=nothing, plot_folder=nothing)
    sorted_indices = sortperm(max_lengths, rev=true)
    max_lengths = max_lengths[sorted_indices]
    weights = weights[sorted_indices]
    cdf_vec = 1 .- cumsum(weights ./ sum(weights))

    if !isnothing(n_bootstrap)
        bootstrap_max_lengths_vec = []
        bootstrap_cdf_vec_vec = []
        for i in 1:n_bootstrap
            bootstrap_max_lengths, bootstrap_cdf_vec = bootstrap_cdf(max_lengths, weights)
            push!(bootstrap_max_lengths_vec, bootstrap_max_lengths)
            push!(bootstrap_cdf_vec_vec, bootstrap_cdf_vec)
        end

        if !isnothing(plot_folder)
            p = plot(xlabel="Length", ylabel="CDF", dpi=300)
            for i in 1:n_bootstrap
                plot!(p, bootstrap_max_lengths_vec[i], bootstrap_cdf_vec_vec[i], alpha=0.02, label="", color=:black)
            end
            plot!(p, max_lengths, cdf_vec, label="", color=:red, linewidth=3)
            savefig(plot_folder * "bootstrap_cdf.png")
        end

        return max_lengths, cdf_vec, bootstrap_max_lengths_vec, bootstrap_cdf_vec_vec
    end

    if !isnothing(plot_folder)
        plot(max_lengths, cdf_vec, label="", color=:red, linewidth=3, xlabel="Length", ylabel="CDF")
        savefig(plot_folder * "bootstrap_cdf.png")
    end

    return max_lengths_sorted, cdf_vec
end

function get_new_samples(parameters, results, max_lengths, weights; index_start, index_stop, n_samples=100, plot_folder=nothing, q=0.7)
    # Get indices of cases above threshold
    length_threshold = quantile(max_lengths, q)
    top_cases = findall(x -> x >= length_threshold, max_lengths)

    # Extract parameters for those cases
    top_parameters = parameters[top_cases, :]
    top_weights = weights[top_cases]

    # plot the quantile on the results plots
    if !isnothing(plot_folder)
        plot_results(results; index_start, index_stop)
        hline!([length_threshold], color=:red, label="$q quantile")
        savefig(plot_folder * "quantile_results.png")

        p1, plots = plot_parameters(parameters, alpha=0.3, label="All Samples")
        p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
        p2
        savefig(plot_folder * "quantile_parameters.png")
    end

    ## Fit a distribution to the dataset
    d = fit_mle(MvNormal, collect(Matrix{Float64}(top_parameters)'), top_weights)

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
    if !isnothing(plot_folder)
        raw_samples = rand(d, 100000)
        raw_samples_df = DataFrame(raw_samples', names(parameters))
        p1, plots = plot_parameters(parameters, alpha=0.3, label="All Samples")
        p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
        p3, plots = plot_parameters(raw_samples_df; plots, alpha=0.3, label="New Raw Samples", bins=20)
        p3
        savefig(plot_folder * "new_raw_samples_parameters.png")
    end

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

    rejection_samples = [rejection_sample(d, PRIOR) for _ in 1:n_samples]
    rejection_samples_df = DataFrame(hcat(rejection_samples...)', names(parameters))
    if !isnothing(plot_folder)
        p1, plots = plot_parameters(parameters, alpha=0.3, label="All Samples")
        p2, plots = plot_parameters(top_parameters; plots, alpha=0.3, label="$q quantile")
        p3, plots = plot_parameters(rejection_samples_df; plots, alpha=0.3, label="New Samples (rejected)")
        p3
        savefig(plot_folder * "new_rejection_samples_parameters.png")
    end

    ## Compute the normalization constant and weights of the samples
    # Normalization constant of the rejection-sampled distribution
    N = mean([!reject(d, PRIOR) for d in eachcol(raw_samples)])

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

    weights = [weight(sample, d, PRIOR; N) for sample in rejection_samples]
    if !isnothing(plot_folder)
        histogram(weights, xlabel="Weights", label="")
        savefig(plot_folder * "new_weights.png")
    end

    ## write everything to disk
    results = deepcopy(rejection_samples_df)
    results.weights = weights
    CSV.write(plot_folder * "new_samples.csv", results)
end
