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

# main user-facing function
function estimate_logit(fm, data)
  af = apply_schema(fm, schema(fm, data))
  y, Xd = modelcols(af, data)

  parameter_names = coefnames(af)[2]

  Xs = sparse(onehotbatch(data.TY, sort(unique(data.TY)))')
  Xs = Xs[:, 2:end]

  choicefe_names = sort(unique(data.TY))[2:end]

  X = hcat(Xd, Xs)
  gs = collect(groupinds(data.logitID))

  theta0 = randn(size(X, 2))

  obj(theta) = nll(y, X, theta, gs)
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

  result = trunk(nlp, verbose = 1, max_time = 600.0)
  theta_star = result.solution

  return DataFrame(
    parameter = vcat(String.(parameter_names), choicefe_names),
    pointest = theta_star
  )
end

end
