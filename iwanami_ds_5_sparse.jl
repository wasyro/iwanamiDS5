using ScikitLearn
using Gadfly, Cairo
using PyCall

@sk_import datasets: load_boston
@sk_import linear_model: (Lasso, lasso_path, LassoCV)

function normalize!(X::Array)
    for i in 1:size(X)[2]
        X[:, i] -= mean(X[:, i])
        X[:, i] /= std(X[:, i])
    end
    return nothing
end

boston = load_boston()
X, y = boston["data"], boston["target"]
feature_names = convert(Array, boston["feature_names"])
normalize!(X)

clf = Lasso(alpha=0.1)
clf[:fit](X, y)
println(clf[:coef_])
println(clf[:intercept_])

alphas, coefs, _ = lasso_path(X, y, fit_intercept=false)

log_alphas = log10.(alphas)

function plot_solution_path(log_alphas::Array, coefs::Array, labels::Array)
    layers = []
    colors = ["dimgray", "brown1", "darksalmon", "blue", "red",
              "gold3", "orange", "purple", "pink", "lightgreen",
              "palevioletred2", "cyan4", "khaki"]
    p = plot(Guide.xlabel("-log10(alpha)"), Guide.ylabel("coefficients"),
             Guide.manual_color_key("feature", labels, colors),
             Theme(background_color="white"))
    for i in 1:size(coefs)[1]
        push!(p.layers, layer(x=-log_alphas, y=coefs[i, :], Geom.line,
                              Theme(default_color=colors[i]))[1])
    end
    return p
end

p = plot_solution_path(log_alphas, coefs, feature_names)
draw(PNG("solution_path_result.png", 1280px, 720px), p)

k = 10
log_alphas = linspace(-5, 1, 100)
alphas = 10 .^ log_alphas
model = LassoCV(cv=k, alphas=alphas)[:fit](X,y)

function mean_std_column(ary::Array)
    means = Array{Float64}(0)
    stds = Array{Float64}(0)
    for i in 1:size(ary)[1]
        push!(means, mean(ary[i, :]))
        push!(stds, std(ary[i, :]))
    end
    return means, stds
end

function plot_errorbar(model::PyCall.PyObject, k::Int64)
        path_means, path_stds = mean_std_column(model[:mse_path_])
        p = plot(x=log10.(model[:alphas_]), y=path_means,
                 ymin=path_means-path_stds/sqrt(k), ymax=path_means+path_stds/sqrt(k),
                 Geom.point, Geom.errorbar,
                 Theme(background_color="white"))
        return p
end

p = plot_errorbar(model, k)
draw(PNG("CV_result.png", 1280px, 720px), p)
