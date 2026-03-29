# Voting Rules As Utility Transforms

Working note capturing a possible conceptual contribution and a set of candidate
formal measures suggested by the current results.

## Core Framing

In our model, every voting method takes the same primitive input:

- `U in R^{M x n}`, a utility matrix
- row `i` is voter `i`'s utility vector over the `n` proposals under
  consideration in the current round

It is useful to think of a voting rule as a pipeline

`U -> T(U) -> A(T(U)) -> social ranking -> winner`

where:

- `T` is a voter-level or profile-level transform of utilities
- `A` is the aggregation/comparison stage

Examples:

- `Total Score`: `T` is essentially the identity on each row; `A` sums columns
- `Borda`: `T` maps each row to rank points; `A` sums those points
- `Plurality`: `T` maps each row to a top-choice indicator; `A` sums
- `Approval`: `T` thresholds each row against a reference point; `A` sums
- `STAR`: `T` keeps scores, then `A` uses both summed scores and a pairwise runoff
- `Minimax`: `T` is naturally viewed as producing pairwise comparisons; `A`
  uses worst pairwise defeat

This framing suggests that voting methods differ not only in normative
properties, but in:

- what information they preserve from `U`
- what kinds of perturbations they are robust to
- how much strict order structure survives the transform

## Candidate Decomposition

The most promising decomposition so far is:

1. Information preservation of the transform `T`
2. Robustness of the aggregated social ranking to perturbations of `U`

There is also a useful refinement inside robustness:

- transform robustness: how hard it is to change `T(U)`
- aggregation robustness: how hard it is to change the social ranking after
  transform and aggregation

These are distinct. For example:

- Borda looks very robust at the transform stage because rank-preserving
  perturbations do not change the transform at all
- Total Score is not transform-robust in that sense, because any perturbation
  changes the score vector, but it may be robust after aggregation because
  column sums average across many voters

## 1. Information Preservation On `T`

### 1.1 Maximal Information Retention

Under the natural notion of input preservation:

- `Total Score` is maximally information-preserving because it retains the full
  utility vector
- `Borda` discards cardinal magnitudes but preserves full within-voter order
- `Plurality` preserves only the top choice
- `Approval` preserves only a coarse thresholded partition

This is already useful conceptually:

- if `Total Score` is maximal on information retention but not best overall,
  then information retention alone cannot explain performance
- some second property, likely a form of robustness, must matter

### 1.2 Quantized Information of the Transform

One possible formalization:

- define `T(u)` for a single voter
- quantize continuous coordinates to precision `eps`
- measure `H(Q_eps(T(u)))`

This gives the expected information content of the transformed ballot.

For `n = 4` under generic no-tie utilities:

- `Plurality`: top choice only, so about `log2(4) = 2` bits
- full ranking methods (`Borda`, `IRV`, `Minimax`, `Kemeny`): `log2(4!) ~=
  4.585` bits
- `Total Score`: much larger, growing with precision

This is principled, but by itself it likely over-favors score methods.

### 1.3 Strict-Order Retention

A more structural measure:

- for a transform `T`, define the set of strict rankings consistent with `T(u)`
- smaller ambiguity means more strict-order information retained

Equivalent language:

- `Plurality` leaves the order of the non-winners unresolved
- `Approval` leaves within-block order unresolved
- `Borda` preserves the full strict order

This looks like a useful ordinal information measure, but it is discrete.

### 1.4 Continuous Strict-Order Retention

A more promising continuous analogue:

- apply the transform `T`
- normalize the transformed score vector to remove arbitrary scale
- measure how well separated the transformed order remains

For example, after normalization, define pairwise or adjacent gaps in the
transformed score vector and aggregate them.

Why normalization matters:

- raw transformed scores are gameable by arbitrary rescaling
- normalized transformed scores instead measure how distinctly the transform
  separates candidates

This seems especially promising because:

- `Plurality` has many exact ties after transform
- `Approval` has block ties
- `Borda` gives a fully separated ordering
- `Total Score` may retain more information, but not necessarily in a way that
  produces evenly distributed order separation

Among normalized positional rules, Borda is special because it gives equal
spacing between adjacent rank levels. This suggests several possible optimality
results.

## 2. Robustness

### 2.1 Robustness Depends On The Perturbation Class

There is no single context-free robustness ranking of the methods.

Examples:

- under diffuse additive perturbations to all entries of `U`, `Total Score` may
  be quite robust after aggregation because sums average noise out
- under voter-specific monotone distortions, `Borda` is perfectly robust while
  `Total Score` is not
- under rank-preserving perturbations of utilities, any purely ordinal method is
  perfectly transform-robust

So robustness must be defined relative to a perturbation class.

### 2.2 Ordinal-Cell Robustness

Very important perturbation class:

- perturbations that do not change an individual voter's strict ranking

For such perturbations:

- `Borda`, `Plurality`, `IRV`, `Minimax`, and `Kemeny` are unchanged at the
  transform stage
- `Total Score` is not
- `STAR` is only partly robust because the score stage still changes

This is likely central to the explanation of Borda's good performance:

- it is completely insensitive to cardinal fluctuations inside an ordinal cell
- score methods are not

### 2.3 Winner Robustness vs Ranking Robustness

Robustness should probably not be defined only by winner flips.

Better object:

- robustness of the full social ranking under adversarial perturbation

Let `R_F(U)` be the social ranking under rule `F`.

Then define a robustness radius such as:

- the smallest perturbation needed to change `R_F(U)` under some ranking metric
  (e.g. Kendall tau or simply any order change)

Then average that radius over a distribution of utility matrices `U`.

This avoids the coarseness of winner-only measures.

### 2.4 Gaussian Benchmark For Total Score

For iid Gaussian utilities, Total Score is analytically tractable.

If:

- `U_ij ~ N(0, sigma^2)` iid
- candidate scores are column sums

then:

- each aggregate candidate score is Gaussian
- winner robustness is controlled by the top-two column-sum gap
- social-ranking robustness is controlled by the minimum adjacent gap among the
  sorted column sums

This provides a useful benchmark, but it also taught us something important:

- under isotropic additive perturbations, `Total Score` can look more robust
  than `Borda`

This does not refute the broader Borda story. It means only that isotropic
additive perturbations are probably not the perturbation class that explains the
empirical NK results.

## 3. Candidate Pair Of Measures

Current best guess for a principled pair:

### A. Information / Order Preservation

Something defined on `T`, likely one of:

- quantized information content of the transformed ballot
- strict-order retention
- normalized continuous strict-order retention

The normalized continuous strict-order retention idea currently looks best.

### B. Robustness

Something defined on the aggregated social ranking, averaged over a relevant
distribution of `U`, and explicitly tied to a perturbation class.

Most promising perturbation classes:

- voter-wise monotone distortions
- within-cell cardinal perturbations that preserve individual rankings
- sparse adversarial distortions affecting only a few voters
- possibly rank-metric robustness of the final social ranking

## 4. Why Borda Looks Special

Borda appears to combine three unusually attractive properties:

1. Full strict-order preservation among ordinal methods
2. Complete invariance to within-cell cardinal perturbations
3. Equal spacing between rank levels after normalization

This suggests that Borda may be close to optimal among ordinal positional rules
under a natural tradeoff:

- preserve as much order information as possible
- remain maximally robust to cardinal distortions that do not change order

## 5. Plausible Optimality Result

One possible theorem direction:

Among anonymous additive ordinal transforms satisfying:

- invariance under voter-wise strictly increasing transforms
- full strict-order sensitivity
- normalized symmetry / equal treatment of rank levels

Borda uniquely maximizes a continuous order-retention functional.

The most likely setting for a clean theorem is the class of positional scoring
rules.

Reason:

- if rank scores are normalized so top = 1 and bottom = 0
- then Borda is the equal-spacing rule
- equal spacing uniquely maximizes the minimum adjacent gap and minimizes gap
  dispersion

This sounds very close to what we want conceptually.

## 6. Important Caveat

Some measures can be made to favor Borda by construction. We should avoid that.

A defensible story needs:

- a measure of retained structure that is independently meaningful
- a measure of robustness tied to a substantively meaningful perturbation class
- ideally, at least one analytic benchmark or theorem

The fact that `Total Score` is maximal on raw information retention is actually
useful. If it still loses overall, then the theory must capture a tradeoff, not
simply reward more information.

## 7. Concrete Next Steps

1. Add `Kemeny-Young` as an explicitly profile-global benchmark
2. Formalize continuous strict-order retention for transformed score vectors
3. Formalize ordinal-cell robustness and rank-robustness of the final social
   ranking
4. Check whether Borda is uniquely optimal within positional scoring rules under
   a normalized gap-based objective
5. Test simple dependence models for utilities, e.g. voter-specific monotone
   distortions or rescalings, where Borda should outperform score

## 8. Working Hypothesis

The empirical success of Borda is not because it uses the most information.
Rather, it may come from sitting near the best tradeoff between:

- preserving rich strict-order information
- discarding unstable cardinal information
- producing a transformed representation with strong, evenly distributed order
  separation
- maintaining robustness to perturbations that should be treated as irrelevant

If that is right, then the path to a stronger theory is not "more information"
alone, but identifying the right balance between retained structure and
robustness.
