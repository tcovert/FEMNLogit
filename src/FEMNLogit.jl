module FEMNLogit

using LogExpFunctions, LinearAlgebra#, Distributions
using SparseArrays, Zygote, MLUtils#, CategoricalArrays
using ForwardDiff, OneHotArrays
using Zygote: @adjoint
using ManualNLPModels, JSOSolvers
using DataFrames, CategoricalArrays, SplitApplyCombine
using StatsModels

export estimate_logit, @formula

########## define some core functions
# this is a separate function for an otherwise obvious step so we can sidestep
# some autodiff issues that arise if X is sparse, and we will always have a
# sparse X when we have choice fixed effects
function mvprod(X, theta)
  return X * theta
end

# this just tells zygote not to bother trying to differentiate X.  this is 
# crucial if X is sparse (and we want to use the hessian-vector product trick)
@adjoint mvprod(X, theta) = mvprod(X, theta), dbar -> (nothing, X' * dbar)

# standard logit choice-level nll
function choicenll(y, delta)
  return -1.0 * (dot(y, delta) - logsumexp(delta))
end

# inputs to this:
# y: a vector of 1 or 0, 1 indicating that this choice was made
# X: a (potentially sparse) matrix of the things that go into delta
# theta: parameters on X that generate delta
# gs: an array of arrays, where each sub-array is the row indices corresponding
# to an individual choice situation
function nll(y, X, theta, gs)
  deltas = mvprod(X, theta)
  mlg = map(length, gs)
  return sum(map(choicenll, chunk(y, size = mlg), chunk(deltas, size = mlg)))
end

# something like this?
function hnll(y, X, Z, theta, gamma, gs)
  deltas = mvprod(X, theta) .* exp.(mvprod(Z, gamma))
  mlg = map(length, gs)
  return sum(map(choicenll, chunk(y, size = mlg), chunk(deltas, size = mlg)))
end

# sets up the nlp and solves it
function logit_solve(obj, theta0)
  grad(theta) = Zygote.gradient(obj, theta)[1]
  Hv(theta, v) = ForwardDiff.derivative(a -> grad(theta .+ a .* v), 0.0)

  function grad!(gval, x)
    gval .= grad(x)
  end
  
  function Hv!(hvval, x, v; obj_weight = 1.0)
    hvval .= Hv(x, v)
    hvval .*= obj_weight
    return hvval
  end

  nlp = NLPModel(theta0, obj, grad = grad!, hprod = Hv!)

  return trunk(nlp, verbose = 1, max_time = 600.0)
end

function inner_vanilla_logit(y, X, gs)
  obj(theta) = nll(y, X, theta, gs)
  theta0 = randn(size(X, 2))

  return logit_solve(obj, theta0)
end

function inner_scale_logit(y, X, Z, gs)
  K = size(X, 2)
  L = size(Z, 2)
  obj(theta) = hnll(y, X, Z, theta[1:K], theta[K+1:end], gs)
  theta0 = randn(K + L)

  return logit_solve(obj, theta0)
end

# main user-facing function
function estimate_logit(fm, data; scalefm = nothing)
  # all of this stuff has to happen for any model
  af = apply_schema(fm, schema(fm, data))
  y, Xd = modelcols(af, data)

  Xs = sparse(onehotbatch(data.TY, sort(unique(data.TY)))')
  Xs = Xs[:, 2:end]

  X = hcat(Xd, Xs)
  gs = collect(groupinds(data.logitID))

  choicefe_names = sort(unique(data.TY))[2:end]
  parameter_names = String.(coefnames(af)[2])

  if !isnothing(scalefm)
    af_scale = apply_schema(scalefm, schema(scalefm, data))
    nothing, Z = modelcols(af_scale, data)
    solution = inner_scale_logit(y, X, Z, gs)
    scale_parameter_names = String.(coefnames(af_scale)[2])
    scale_parameter_names = "Scale_" .* scale_parameter_names
    parameter_names = vcat(parameter_names, scale_parameter_names)
  else
    solution = inner_vanilla_logit(y, X, gs)
  end

  theta_star = solution.solution

  return DataFrame(
    parameter = vcat(parameter_names, choicefe_names),
    pointest = theta_star
  )
end

end
