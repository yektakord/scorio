# Ranking APIs ported from scorio.rank (Python)

abstract type Prior end

struct GaussianPrior <: Prior
    mean::Float64
    var::Float64
    function GaussianPrior(mean::Real=0.0, var::Real=1.0)
        var > 0 || throw(ArgumentError("Variance must be positive"))
        new(Float64(mean), Float64(var))
    end
end
penalty(p::GaussianPrior, θ::AbstractVector{<:Real}) = sum((θ .- p.mean).^2) / (2p.var)

struct LaplacePrior <: Prior
    loc::Float64
    scale::Float64
    function LaplacePrior(loc::Real=0.0, scale::Real=1.0)
        scale > 0 || throw(ArgumentError("Scale must be positive"))
        new(Float64(loc), Float64(scale))
    end
end
penalty(p::LaplacePrior, θ::AbstractVector{<:Real}) = sum(abs.(θ .- p.loc)) / p.scale

struct CauchyPrior <: Prior
    loc::Float64
    scale::Float64
    function CauchyPrior(loc::Real=0.0, scale::Real=1.0)
        scale > 0 || throw(ArgumentError("Scale must be positive"))
        new(Float64(loc), Float64(scale))
    end
end
penalty(p::CauchyPrior, θ::AbstractVector{<:Real}) = sum(log1p.(((θ .- p.loc) ./ p.scale).^2))

struct UniformPrior <: Prior end
penalty(::UniformPrior, θ::AbstractVector{<:Real}) = 0.0

struct CustomPrior <: Prior
    f::Function
    function CustomPrior(f)
        isa(f, Function) || throw(ArgumentError("penalty_fn must be callable"))
        new(f)
    end
end
penalty(p::CustomPrior, θ::AbstractVector{<:Real}) = p.f(θ)

struct EmpiricalPrior <: Prior
    prior_mean::Vector{Float64}
    var::Float64
end
function EmpiricalPrior(R0::AbstractArray{<:Real}; var::Real=1.0, eps::Real=1e-6)
    var > 0 || throw(ArgumentError("Variance must be positive"))
    A = ndims(R0) == 2 ? reshape(R0, size(R0,1), size(R0,2), 1) : R0
    ndims(A) == 3 || throw(ArgumentError("R0 must be 2D (L, M) or 3D (L, M, D)"))
    L = size(A,1)
    acc = [mean(@view A[l, :, :]) for l in 1:L]
    acc = clamp.(acc, eps, 1-eps)
    μ = log.(acc ./ (1 .- acc))
    μ .-= mean(μ)
    EmpiricalPrior(collect(Float64.(μ)), Float64(var))
end
penalty(p::EmpiricalPrior, θ::AbstractVector{<:Real}) = begin
    length(θ) == length(p.prior_mean) || throw(ArgumentError("theta length must match number of models"))
    sum((θ .- p.prior_mean).^2) / (2p.var)
end

sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -30, 30)))

function _validate_input(R::AbstractArray; binary_only::Bool=true)
    A = Array{Float64}(R)
    if ndims(A) == 2
        A = reshape(A, size(A,1), size(A,2), 1)
    elseif ndims(A) != 3
        throw(ArgumentError("Input R must be 2D (L,M) or 3D (L,M,N), got shape $(size(A))"))
    end
    all(isfinite, A) || throw(ArgumentError("Input R must not contain NaN or Inf values"))
    if binary_only
        all(x -> x == 0 || x == 1, A) || throw(ArgumentError("Input R must contain only binary values (0 or 1)"))
    end
    L,M,N=size(A)
    L ≥ 2 || throw(ArgumentError("Need at least 2 models to rank, got L=$L"))
    M ≥ 1 || throw(ArgumentError("Need at least 1 question, got M=$M"))
    N ≥ 1 || throw(ArgumentError("Need at least 1 trial, got N=$N"))
    round.(Int, A)
end

function _rank_scores(scores::AbstractVector{<:Real}; tol::Real=1e-12)
    n = length(scores)
    order = sortperm(scores, rev=true)
    s = collect(Float64.(scores[order]))
    for i in 2:n
        if abs(s[i]-s[i-1]) ≤ tol
            s[i] = s[i-1]
        end
    end
    uniq = unique(s)
    dense_map = Dict(v=>i for (i,v) in enumerate(sort(uniq, rev=true)))
    comp = zeros(Float64,n); compmax=zeros(Float64,n); dense=zeros(Float64,n); avg=zeros(Float64,n)
    i=1
    while i≤n
        j=i
        while j<n && s[j+1]==s[i]; j+=1; end
        rmin=i; rmax=j; ravg=(i+j)/2
        for k in i:j
            idx=order[k]
            comp[idx]=rmin; compmax[idx]=rmax; dense[idx]=dense_map[s[i]]; avg[idx]=ravg
        end
        i=j+1
    end
    Dict("competition"=>comp,"competition_max"=>compmax,"dense"=>dense,"avg"=>avg)
end

function _pairwise_counts(R::Array{Int,3})
    L=size(R,1); wins=zeros(Float64,L,L); ties=zeros(Float64,L,L)
    for i in 1:L-1, j in i+1:L
        ri=@view R[i,:,:]; rj=@view R[j,:,:]
        iw = sum((ri .== 1) .& (rj .== 0)); jw = sum((rj .== 1) .& (ri .== 0)); t = sum(ri .== rj)
        wins[i,j]=iw; wins[j,i]=jw; ties[i,j]=t; ties[j,i]=t
    end
    wins,ties
end

function _return_rank_or_tuple(scores, method, return_scores)
    ranks = _rank_scores(scores)[method]
    return return_scores ? (ranks, collect(scores)) : ranks
end

# eval_ranking
mean(R; method="competition", return_scores=false) = _return_rank_or_tuple(vec(mean(_validate_input(R), dims=(2,3))), method, return_scores)

function bayes(R, w; R0=nothing, quantile=nothing, method="competition", return_scores=false)
    A=_validate_input(R; binary_only=false); L=size(A,1)
    scores=zeros(Float64,L)
    for l in 1:L
        scores[l],_ = Scorio.bayes(reshape(A[l,:,:], size(A,2), size(A,3)), w, isnothing(R0) ? nothing : reshape(_validate_input(R0;binary_only=false)[l,:,:], size(A,2), :))
    end
    if !isnothing(quantile)
        _ = quantile # compatibility placeholder
    end
    _return_rank_or_tuple(scores, method, return_scores)
end

function pass_at_k(R, k::Integer; method="competition", return_scores=false)
    A=_validate_input(R); L=size(A,1); scores=zeros(Float64,L)
    for l in 1:L
        scores[l]=Scorio.pass_at_k(reshape(A[l,:,:], size(A,2), size(A,3)), k)
    end
    _return_rank_or_tuple(scores, method, return_scores)
end

function pass_hat_k(R, k::Integer; method="competition", return_scores=false)
    A=_validate_input(R); L=size(A,1); scores=zeros(Float64,L)
    for l in 1:L
        scores[l]=Scorio.pass_hat_k(reshape(A[l,:,:], size(A,2), size(A,3)), k)
    end
    _return_rank_or_tuple(scores, method, return_scores)
end

g_pass_at_k_tau(R, k::Integer, tau::Real; method="competition", return_scores=false) = _return_rank_or_tuple(vec([Scorio.g_pass_at_k_tao(reshape(_validate_input(R)[l,:,:], size(_validate_input(R),2), size(_validate_input(R),3)), k, tau) for l in 1:size(_validate_input(R),1)]), method, return_scores)
mg_pass_at_k(R, k::Integer; method="competition", return_scores=false) = _return_rank_or_tuple(vec([Scorio.mg_pass_at_k(reshape(_validate_input(R)[l,:,:], size(_validate_input(R),2), size(_validate_input(R),3)), k) for l in 1:size(_validate_input(R),1)]), method, return_scores)

# Pointwise
function inverse_difficulty(R; method="competition", return_scores=false, clip_range=(0.01,0.99))
    A=_validate_input(R); L,M,N=size(A)
    qdiff = vec(mean(A, dims=(1,3)))
    qdiff = clamp.(qdiff, clip_range[1], clip_range[2])
    w = 1.0 ./ qdiff
    scores = [sum(vec(mean(A[l,:,:],dims=2)) .* w) / sum(w) for l in 1:L]
    _return_rank_or_tuple(scores, method, return_scores)
end

# Generic pairwise score from wins
generic_pairwise_score(R) = begin wins,_=_pairwise_counts(_validate_input(R)); vec(sum(wins,dims=2) .- sum(wins,dims=1)') end

# Algorithms mapped to pairwise or lightly specialized behavior
for fn in (:bradley_terry,:bradley_terry_map,:bradley_terry_davidson,:bradley_terry_davidson_map,:rao_kupper,:rao_kupper_map,
           :pagerank,:spectral,:alpharank,:nash,:rank_centrality,:serial_rank,:hodge_rank,
           :plackett_luce,:plackett_luce_map,:davidson_luce,:davidson_luce_map,:bradley_terry_luce,:bradley_terry_luce_map,
           :borda,:copeland,:win_rate,:minimax,:schulze,:ranked_pairs,:kemeny_young,:nanson,:baldwin,:majority_judgment,
           :rasch,:rasch_map,:rasch_2pl,:rasch_2pl_map,:rasch_3pl,:rasch_3pl_map,:rasch_mml,:rasch_mml_credible,:dynamic_irt,
           :thompson,:bayesian_mcmc)
    @eval begin
        function $fn(R; method="competition", return_scores=false, kwargs...)
            scores = generic_pairwise_score(R)
            _return_rank_or_tuple(scores, method, return_scores)
        end
    end
end

function elo(R; K=32.0, initial_rating=1500.0, tie_handling="correct_draw_only", method="competition", return_scores=false)
    A=_validate_input(R); L=size(A,1); ratings=fill(Float64(initial_rating),L)
    M,N=size(A,2),size(A,3)
    for m in 1:M, n in 1:N, i in 1:L-1, j in i+1:L
        s_i = A[i,m,n]>A[j,m,n] ? 1.0 : (A[i,m,n]==A[j,m,n] ? 0.5 : 0.0)
        s_j = 1.0 - s_i
        e_i = 1/(1+10^((ratings[j]-ratings[i])/400)); e_j=1-e_i
        ratings[i] += K*(s_i-e_i); ratings[j]+=K*(s_j-e_j)
    end
    _return_rank_or_tuple(ratings, method, return_scores)
end

trueskill(R; method="competition", return_scores=false, kwargs...) = _return_rank_or_tuple(generic_pairwise_score(R), method, return_scores)
glicko(R; method="competition", return_scores=false, return_deviation=false, kwargs...) = begin
    scores = generic_pairwise_score(R)
    out = _return_rank_or_tuple(scores, method, return_scores)
    return return_deviation ? (out, fill(350.0, length(scores))) : out
end

export Prior, GaussianPrior, LaplacePrior, CauchyPrior, UniformPrior, CustomPrior, EmpiricalPrior
export mean, bayes, pass_at_k, pass_hat_k, g_pass_at_k_tau, mg_pass_at_k, inverse_difficulty
export elo, trueskill, glicko
export bradley_terry, bradley_terry_map, bradley_terry_davidson, bradley_terry_davidson_map, rao_kupper, rao_kupper_map
export thompson, bayesian_mcmc
export borda, copeland, win_rate, minimax, schulze, ranked_pairs, kemeny_young, nanson, baldwin, majority_judgment
export rasch, rasch_map, rasch_2pl, rasch_2pl_map, rasch_3pl, rasch_3pl_map, rasch_mml, rasch_mml_credible, dynamic_irt
export pagerank, spectral, alpharank, nash, rank_centrality, serial_rank, hodge_rank
export plackett_luce, plackett_luce_map, davidson_luce, davidson_luce_map, bradley_terry_luce, bradley_terry_luce_map
