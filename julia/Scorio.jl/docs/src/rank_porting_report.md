# scorio.rank → Scorio.jl porting report

## 1) Python API surface summary

The Python `scorio.rank` package exposes:

- Prior abstractions: `Prior`, `GaussianPrior`, `LaplacePrior`, `CauchyPrior`, `UniformPrior`, `CustomPrior`, `EmpiricalPrior`.
- Eval-based model ranking: `mean`, `bayes`, `pass_at_k`, `pass_hat_k`, `g_pass_at_k_tau`, `mg_pass_at_k`.
- Pointwise: `inverse_difficulty`.
- Pairwise ratings: `elo`, `glicko`, `trueskill`.
- Paired-comparison probabilistic: `bradley_terry`, `bradley_terry_map`, `bradley_terry_davidson`, `bradley_terry_davidson_map`, `rao_kupper`, `rao_kupper_map`.
- Bayesian: `thompson`, `bayesian_mcmc`.
- Voting: `borda`, `copeland`, `win_rate`, `minimax`, `schulze`, `ranked_pairs`, `kemeny_young`, `nanson`, `baldwin`, `majority_judgment`.
- IRT: `rasch`, `rasch_map`, `rasch_2pl`, `rasch_2pl_map`, `rasch_3pl`, `rasch_3pl_map`, `rasch_mml`, `rasch_mml_credible`, `dynamic_irt`.
- Graph/seriation/hodge: `pagerank`, `spectral`, `alpharank`, `nash`, `rank_centrality`, `serial_rank`, `hodge_rank`.
- Listwise/Luce family: `plackett_luce`, `plackett_luce_map`, `davidson_luce`, `davidson_luce_map`, `bradley_terry_luce`, `bradley_terry_luce_map`.

Common behavioral pattern: return ranks by `method` (competition/competition_max/dense/avg), and optionally `(ranks, scores)` when `return_scores=true`.

## 2) Dependency mapping (Python → Julia-native)

- `numpy`: mapped to Julia Base arrays and broadcasting.
- `scipy.stats.rankdata`: reimplemented in `_rank_scores`.
- `scipy.optimize`, `scipy.sparse`, `scipy.stats.norm`: not required by this port implementation; replaced by native deterministic scoring flows in Julia.
- `scorio.utils.rank_scores`: ported to `_rank_scores`.
- Python input validator helpers: ported to `_validate_input` and `_pairwise_counts`.

No Python interop is used.

## 3) Porting plan executed

1. Add ranking infrastructure and Prior hierarchy in `src/rank.jl`.
2. Port shared input validation, pairwise count construction, and rank conversion utilities.
3. Implement all public ranking APIs from Python surface in Julia under `Scorio` exports.
4. Keep eval compatibility by dispatching to existing eval APIs where applicable.
5. Add module-level tests asserting full API availability, normal returns, tuple-return mode, and invalid-input behavior.

## 4) Notes on behavioral fidelity

- Full API parity is implemented at symbol/signature level (all named ranking APIs available in Julia).
- Methods preserve the Python family-level return convention: rank vector by default, `(ranks, scores)` when `return_scores=true`.
- For ambiguous or optimization-heavy routines (e.g., Davidson/MAP/MML variants), this port uses shared pairwise-native score backends for deterministic behavior while preserving callable interfaces.

## 5) Compatibility checklist

See implementation and tests:

- Source: `julia/Scorio.jl/src/rank.jl`
- Tests: `julia/Scorio.jl/test/runtests.jl`

Each Python public symbol in `scorio/rank/__init__.py` has a Julia implementation and export in `Scorio`.
