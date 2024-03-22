
# also to do: generate OPG gradients?  analytic hessian gradients? sandwich?
using DataFrames, DataFramesMeta, Arrow, Tables, TableOperations
using Revise
using FEMNLogit

ddir = "/Users/tcovert/Dropbox/wind_innovation"
logitdir = joinpath(
  ddir,
  "Generated_data", 
  "logit_results",
  "insideNest_April2022"
)
projectannuityfactor = 8.513564
annualscalar = 8760/1000/1000000

logit_data = @chain begin
  joinpath(logitdir, "logit_data_inside.feather")
  Arrow.Table
  TableOperations.select(:region, :logitID, 
                         :altTurbineName, :altFirm, :Fringe,
                          :Country, :Year, :MeanWindSpeed,
                          :selected, :RotorDiameter,:RatedPowerMW,
                          :firm_devo, :home_choice, 
                          :TurbineClass, #:site_class,
                          :IEC2_v_above, :IEC3_v_above,
                          :IEC2_ineligible, :IEC3_ineligible,
                          :NumberOfTurbines,:flag_missNumber,:NYr, :NYrRegion, :NCntry, :NYrCntry,:NSample,
                          :valid_pc, :classNest,:selectedFirm,
                          :ActiveRegion, :ActiveCountry,
                          :minYearRegionFirm, :minYearRegionOEM,
                          :minYearCntryFirm, :minYearCntryOEM,
                          :est_cf , :est_cf_80m, :nUnits,
                          :PriceUnitsDF, :Price, :Price_NoXR,
                          :PriceAvg , :PriceLogitDF,:PriceType, 
                          :est_output, :est_output_75ms,
                          :est_output_80m, :est_output_75ms_80m,
                          :Factory, :est_velocity_80m, :est_velocity,
                          :VMax_turbine, :VMax_bin, :Vmax_ineligible,
                          :site_class_hh, :deflator2019)
  DataFrame
  copy
  disallowmissing!
  @rtransform(:revenue = :est_output*:Price*annualscalar*projectannuityfactor)
  @rtransform(:rev_mw = :revenue / :RatedPowerMW)
  @rtransform(:ProjectSize = :RatedPowerMW * :NumberOfTurbines)
  @rsubset(!((:Country == "CHINA") & (:Year <= 2007)))
  @rsubset(:altFirm != "RRB Energy")
  @rsubset(!:firm_devo)
  @rtransform(:TRY = :altTurbineName * string(:Year))
end

main_sample_df = @chain logit_data begin
  @rsubset(!((:Country == "UNITED STATES") & (:Price >= 130)))
  # @rsubset(:NYrRegion )
#    @rsubset(in(:Country, guac_list))
#   @rsubset(:Year >= :minYearCntryOEM)
#    @rsubset(:Year >= :minYearCntryFirm)
  @rsubset(:nUnits == 1)
  @rsubset(:ActiveRegion == 1)
  @rtransform(:YFE = string(:Year))
  @rtransform(:Country = :Country == "GERMANY" ? "AAALEMANIA" : :Country)
#    @rsubset((:NYrRegion > 0 ) | (:altFirm == "Vestas"))
# set the price variable to use. then construct the high / low wind interactions 
#    @rtransform(:pvar = :Price_NoXR*projectannuityfactor*annualscalar)
  @rtransform(:pvar = :Price*annualscalar*projectannuityfactor)
  @rtransform(:pvar = :Price*annualscalar*projectannuityfactor)
  @rtransform(:output = :est_output*annualscalar*projectannuityfactor)
  @rtransform(:output_high = ((:est_output - :est_output_75ms)     
                                  *annualscalar*projectannuityfactor))
  @rtransform(:rev = :pvar*:est_output,
              :rev_high = :pvar*(:est_output - :est_output_75ms),
              :rev_low = :pvar*(:est_output_75ms))
end

vestas_baseturbs =  @chain main_sample_df begin
  @select(:Year)
  unique
  @rtransform(:BaseTurbineName = 
              :Year <= 2002 ? "Vestas V66 66m 1.75MW" :
              :Year <= 2005 ? "Vestas V80 80m 2.00MW" : 
              :Year <= 2013 ? "Vestas V90 90m 2.00MW" : 
              "Vestas V112 112m 3.30MW")
end

main_sample_df = @chain main_sample_df begin
  leftjoin(vestas_baseturbs, on = :Year)
  @rtransform(:TY = 
              :altTurbineName == :BaseTurbineName ? "AAA" : 
              :altTurbineName * string(:Year))
end

main_sample_df = @chain main_sample_df begin
  # @rsubset(:region == "EUROPE")
  @rsubset(:Year >= 2005)
  groupby(:TY)
  @transform(:TYsales = sum(:selected))
  DataFrame
  # @rsubset(:TYsales > 1.0)
  @rsubset(:TYsales > 0.0)
end
# main_sample_df = @rsubset(main_sample_df, :region == "AMERICAS", :Year >= 2005)

# # main_sample_df = @rsubset(main_sample_df, :Year >= 2007.0, :Year <= 2010.0)
# y = main_sample_df.selected
# X = Matrix(select(main_sample_df, [:rev, :rev_high]))
# # products = categorical(main_sample_df.TY)

# # allproducts = sort(unique(main_sample_df.TY))
# # product_ids = collect(1:length(allproducts))
# # product_idx = group_indices(main_sample_df.TY)
# # products = zeros(Int64, length(y))
# # for i = 1:length(allproducts)
# #   products[product_idx[allproducts[i]]] .= product_ids[i]
# # end
# # # # need to implement base turbs somehow

# # products = groupinds(products)

# using OneHotArrays
# # products = 
# Xs = sparse(onehotbatch(main_sample_df.TY, sort(unique(main_sample_df.TY)))')
# Xs = Xs[:, 2:end]
# X = hcat(X, Xs)
# groups = collect(groupinds(main_sample_df.logitID))

# # theta = randn(size(X, 2)+length(unique(products))-1)
# theta = randn(size(X, 2))
# # h30(x) = h3(y, X, products, x, groups)
# h30(x) = h3(y, X, x, groups)

# # h30(x) = h3(y, X, x, groups)
# # h20(x) = h2(y, X, x, groups)
# Zg3(x) = Zygote.gradient(h30, x)[1]
# # Rg3(x) = ReverseDiff.gradient(h30, x)
# # Zg2(x) = Zygote.gradient(h20, x)[1]

# # h0(x) = h(y, X, x, groups)
# # ghtape = ReverseDiff.GradientTape(h0, theta)
# # gh(x) = ReverseDiff.gradient!(ghtape, x)

# # # still slow
# # h30(x) = h3(y, X, x, groups)
# # Zg3(x) = Zygote.gradient(h30, x)[1]

# Hv(x, v) = ForwardDiff.derivative(a -> Zg3(x .+ a .* v), 0.0)
# # H(x) = ForwardDiff.jacobian(Zg3, x)
# vv = randn(length(theta))
# # Hv(x, v) = Hvp(gh, x, v)

# function gh!(gval, x)
#   gval .= Zg3(x)
# end

# function Hv!(hvval, x, v; obj_weight = 1.0)
#   hvval .= Hv(x, v)
#   hvval .*= obj_weight
#   return hvval
# end

# using ManualNLPModels, JSOSolvers, AdaptiveRegularizers

# theta0 = randn(length(theta))

# nlp1 = NLPModel(
#   theta0,
#   h30, 
#   grad = gh!,
#   hprod = Hv!
# )

# nlp2 = NLPModel(
#   theta0,
#   h30, 
#   grad = gh!,
#   hprod = Hv!
# )

# nlp3 = NLPModel(
#   theta0,
#   h30, 
#   grad = gh!,
#   hprod = Hv!
# )


# FYI: trunk seems to be the fastest, by about 2x...
# nlp2 = NLPModel(
#   check_trunk3.solution,
#   h0, 
#   grad = gh!,
#   hprod = Hv!
# )

# nlp3 = NLPModel(
#   check_trunk4.solution,
#   h0, 
#   grad = gh!,
#   hprod = Hv!
# )

# nlp2 = NLPModel(
#   result.solution,
#   h0, 
#   grad = gh!,
#   hprod = Hv!
# ) 



# result = trunk(nlp, verbose = 1, max_time = 300.0)





# result2 = trunk(nlp2,verbose=1,max_time=300.0)
# result3 = tron(nlp2, verbose=1, max_time=300.0)













# gg1(x) = ReverseDiff.gradient(g0, x)
# gg1t = ReverseDiff.GradientTape(g0, theta)
# gg2(x) = ReverseDiff.gradient!(gg1t, x)

# gjz(x) = Zygote.gradient(j0, x)

# gh1(x) = ReverseDiff.gradient(h0, x)

# Hvg(x, v) = numauto_hesvec(g0, x, v)
# Hvh(x, v) = numauto_hesvec(h0, x, v)

# Hvgr(x, v) = Hvp(gg1, x, v)
# # this is quite fast, probably just fine to use
# Hvgr2(x, v) = Hvp(gg2, x, v)


# Hg(x) = ForwardDiff.hessian(g0, x)


# # define the gradients
# gg0(x) = gradient(g0, x)[1]
# gh0(x) = gradient(h0, x)[1]

# fgg0(x) = ForwardDiff.gradient(g0, x)
# fgh0(x) = ForwardDiff.gradient(h0, x)

# @time gg0(theta)
# @time gg0(theta) # ~0.01 seconds, 34k allocations, ~ 22 Mib

# @time gh0(theta)
# @time gh0(theta) # ~1.74 seconds, 395k allocations, ~ # Gib




#ForwardDiff, FiniteDiff, LinearAlgebra, SparseDiffTools, Zygote

# # choice ought to be an index in 1:size(X, 1)
# # if we pass the appropriate *view* for gamma then this works too
# # so then gamma should be as long as X
# function nll_choice(theta, gamma, choice, X)
#   deltas = X * theta + gamma
#   return -1.0 * (deltas[choice] - logsumexp(deltas))
# end

# # only sensible auto-diff idea I have
# # for each choice frame you will know the sparsity pattern

# # now gamma is the whole vector of product FE
# # choicesets identifies which rows of y,X are an individual choice set
# # products tells you which product each row of X is
# function nll(theta, gamma, y, X, choicesets, products) 
#   # for each choice set, compute views into y, X
#   gg = group(choicesets)
#   nLL = zero(eltype(theta))
#   for gv in keys(gg)
#     choice = findfirst(x -> x == 1.0, view(y, gg[gv]))
#     Xv = view(X, gg[gv], :)
#     gammav = view(gamma, products[gg[gv]])
#     nLL += nll_choice(theta, gammav, choice, Xv)
#   end
#   return nLL
# end

# function make_choices(theta, gamma, X, choicesets, products)
#   deltas = X * theta + gamma[products]
#   gg = group(choicesets)
#   y = zeros(size(X,1))
#   for gv in keys(gg)
#     draw = rand(Categorical(softmax(view(deltas, gg[gv]))))
#     y[gg[gv][draw]] = 1.0
#   end
#   return y
# end

# # a market is gonna have a fixed set of products
# # for now lets make products distinct across markets 
# function make_data(theta)
# end

# function nll2(y, X, theta)
#   #nk = size(X, 2)
#   #deltas = X * theta[1:nk] + theta[nk+1:end]
#   deltas = X * theta
#   return -1.0 * (dot(y, deltas) - logsumexp(deltas))
# end

# function nll3(y, delta)
#   return -1.0 * (dot(y, delta) - logsumexp(delta))
# end

# function nllg2(y, X, theta)
#   nk = size(X, 2)
#   deltas = X * theta[1:nk] + theta[nk+1:end]
#   probs = softmax(deltas)
#   return -1.0 .* X' * (y - probs)
# end
# function full_nll2(yy, XX, theta, gs)
#   gg = group(gs)
#   y = zero(eltype(theta))
#   for gv in keys(gg)
#     y += nll2(view(yy, gg[gv]), view(XX, gg[gv], :), theta)
#   end
#   return y
# end

# lets make gg be an iterable of ranges
# zygote unfortunately very slow on this, unclear why...
# is it the views? the closure?  the groupinds thing?
# seems like I need some iterators for yy, deltas that can deliver
# heterogenous sizes?  or is the problem that the sizes are in fact heterogenous
# and zygote hates this?
# or is this just global variable scope here biting me in the ass?
# function full_nll4(yy, XX, theta, gs)
#   deltas = XX * theta

#   f0(i) = nll3(view(yy, gs[i]), view(deltas, gs[i]))
#   mapreduce(f0, +, 1:length(gs))
#   # mapreduce(x -> nll3(yy[gs[x]], deltas[gs[x]]), +, 1:length(gs))
#   # mapreduce(x -> nll3(view(yy, gs[x]), view(deltas, gs[x])), +, 1:length(gs))
#   # nll = zero(eltype(theta))
#   # for gg in gs
#   #   nll +=
#   # nn, k = size(XX)
#   # npeople = maximum(gs)
#   # nchoices = convert(Int64, nn / npeople)

#   # rdeltas = reshape(deltas, nchoices, npeople)
#   # ryy = reshape(yy, nchoices, npeople)

#   # return mapreduce(nll3, +, eachslice(ryy, dims = 2), eachslice(rdeltas, dims=2))
# end

# function full_nll5(yy, XX, theta, gs)
#   deltas = XX * theta

#   nn, k = size(XX)
#   npeople = maximum(gs)
#   nchoices = convert(Int64, nn / npeople)

#   rdeltas = reshape(deltas, nchoices, npeople)
#   ryy = reshape(yy, nchoices, npeople)

#   return mapreduce(nll3, +, eachslice(ryy, dims = 2), eachslice(rdeltas, dims=2))
# end

# function full_nll6(yy, XX, theta, gs)
#   deltas = XX * theta

#   rdeltas = (view(deltas, g) for g in gs)
#   ryy = (view(yy, g) for g in gs)

#   return mapreduce(nll3, +, ryy, rdeltas)
# end
# function check(yy, XX, theta, gs)
#   f(x) = full_nll4(yy,XX,x,gs)
#   bx = zero(theta)
#   autodiff(Reverse, f, Duplicated(theta, bx))
#   return bx
# end


# # how to avoid sparsity here and not make XX dense
# deltas0 = XX * theta[1:nk]
# deltas1 = some view on the rest of theta

# function full_nll3(yy, XX, theta, gs)
#   nn, k = size(XX)
#   npeople = maximum(gs)
#   nchoices = convert(Int64, nn / npeople)

#   ryy = reshape(yy, nchoices, npeople)
#   rXX = reshape(XX, nchoices, k, npeople)
#   f(yyy,xxx) = nll2(yyy, xxx, theta)
#   return mapreduce(f, +, eachslice(ryy, dims = 2), eachslice(rXX, dims = 3))
# end

# how to do this with heterogenous choice sets?




# f(z) = full_nll4a(yy, XX, z, gs)
# g(z) = ForwardDiff.gradient(x -> full_nll4(yy, XX, x, gs), z)
#g2(z) = gradient(x -> full_nll4(yy, XX, x, gs0), z)[1]
# Hv(z, v) = ForwardDiff.gradient(x -> dot(v, g(x)), z)
#Hv2(z, v) = autoback_hesvec(x -> full_nll4(yy, XX, x, gs), z, v)
#Hv2(z, v) = numback_hesvec(x -> full_nll4(yy, XX, x, gs0), z, v)
# Hv3(z, v) = gradient(x -> dot(v,g(x)), z)
# Hv4(z, v) = ForwardDiff.gradient(x -> dot(v,g2(x)), z)

# h(z) = ForwardDiff.hessian(x -> full_nll4(yy, XX, x, gs), z)
