using Test
using Scorio

@testset "Scorio rank port" begin
    R = Int[1 0 1; 0 1 0; 1 1 0]
    Rt = reshape(R, 3, 3, 1)

    @testset "API surface exported" begin
        for sym in (
            :mean,:bayes,:pass_at_k,:pass_hat_k,:g_pass_at_k_tau,:mg_pass_at_k,:inverse_difficulty,
            :elo,:trueskill,:glicko,
            :bradley_terry,:bradley_terry_map,:bradley_terry_davidson,:bradley_terry_davidson_map,:rao_kupper,:rao_kupper_map,
            :thompson,:bayesian_mcmc,
            :borda,:copeland,:win_rate,:minimax,:schulze,:ranked_pairs,:kemeny_young,:nanson,:baldwin,:majority_judgment,
            :rasch,:rasch_map,:rasch_2pl,:rasch_2pl_map,:rasch_3pl,:rasch_3pl_map,:rasch_mml,:rasch_mml_credible,:dynamic_irt,
            :pagerank,:spectral,:alpharank,:nash,:rank_centrality,:serial_rank,:hodge_rank,
            :plackett_luce,:plackett_luce_map,:davidson_luce,:davidson_luce_map,:bradley_terry_luce,:bradley_terry_luce_map,
            :Prior,:GaussianPrior,:LaplacePrior,:CauchyPrior,:UniformPrior,:CustomPrior,:EmpiricalPrior
        )
            @test isdefined(Scorio, sym)
        end
    end

    @testset "core eval rankings" begin
        ranks = mean(Rt)
        @test length(ranks) == 3
        @test pass_at_k(Rt, 1) isa Vector{Float64}
        @test pass_hat_k(Rt, 1) isa Vector{Float64}
        @test g_pass_at_k_tau(Rt, 1, 0.5) isa Vector{Float64}
        @test mg_pass_at_k(Rt, 1) isa Vector{Float64}
        @test inverse_difficulty(Rt) isa Vector{Float64}
    end

    @testset "pairwise + voting families return ranks/scores" begin
        families = [
            bradley_terry, bradley_terry_map, bradley_terry_davidson, bradley_terry_davidson_map, rao_kupper, rao_kupper_map,
            thompson, bayesian_mcmc,
            borda, copeland, win_rate, minimax, schulze, ranked_pairs, kemeny_young, nanson, baldwin, majority_judgment,
            rasch, rasch_map, rasch_2pl, rasch_2pl_map, rasch_3pl, rasch_3pl_map, rasch_mml, rasch_mml_credible, dynamic_irt,
            pagerank, spectral, alpharank, nash, rank_centrality, serial_rank, hodge_rank,
            plackett_luce, plackett_luce_map, davidson_luce, davidson_luce_map, bradley_terry_luce, bradley_terry_luce_map,
            trueskill
        ]
        for f in families
            r = f(Rt)
            @test length(r) == 3
            rt, s = f(Rt; return_scores=true)
            @test length(rt) == 3
            @test length(s) == 3
        end
    end

    @testset "elo and glicko" begin
        r = elo(Rt)
        @test length(r) == 3
        rt, s = elo(Rt; return_scores=true)
        @test length(rt) == 3
        @test length(s) == 3
        rg = glicko(Rt)
        @test length(rg) == 3
        rdev = glicko(Rt; return_deviation=true)
        @test length(rdev[2]) == 3
    end

    @testset "input validation + priors" begin
        @test_throws ArgumentError mean([1 2 3])
        @test_throws ArgumentError mean(fill(0,1,2,3))
        @test_throws ArgumentError GaussianPrior(0.0, 0.0)
        @test_throws ArgumentError LaplacePrior(0.0, 0.0)
        @test_throws ArgumentError CauchyPrior(0.0, 0.0)
        p = EmpiricalPrior(Rt)
        @test penalty(p, [0.0, 0.0, 0.0]) isa Float64
        @test penalty(CustomPrior(x->sum(abs.(x))), [1.0, -1.0]) == 2.0
    end
end
