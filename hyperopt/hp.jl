using CSV
using Plots
using DataFrames

p = CSV.read("./hyperopt-1layer-feat-selected-50bw.csv")
p = dropmissing(p)
names(p)

p[:params] = map(x -> parse(Float64, x), p[:params])
p[:params_1] = map(x -> parse(Int, x), p[:params_1])
p[:params_2] = map(x -> parse(Float64, x), p[:params_2])
p[:params_3] = map(x -> parse(Float64, x), p[:params_3])

dp = scatter(p.value, p.params, label="dropout")
neurons = scatter(p.value, p.params_1, label="# neurons")
lr = scatter(p.value, p.params_2, label="LR")
wd = scatter(p.value, p.params_3, label="WD")
plot(dp, neurons, lr, wd, layout=(2,2), title = ["($i)" for j = 1:1, i=1:4], titleloc = :right, titlefont = font(8))

p

dropmissing(p)

parse(Array{Int},p.params_1)
