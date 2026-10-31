"""Guarded IS/PSIS prior reweighting over a list of changed prior factors.

Scope (see PRIOR_INVENTORY.md): same likelihood, same model structure, same sampled
variables. Only the prior distribution of one or more *sampled* variables changes.

    log w_s = sum_{q in factors} sum_{non-draw axes} [log p1_q(phi_q^s) - log p0_q(phi_q^s)]

Each factor's two distributions must be evaluated on the same draws of the same sampled
variable. Unchanged factors and the likelihood cancel and must not be passed in.

Notes on `az.psislw` (ArviZ 0.19, checked in its source):
  * it modifies its input array in place (`x -= np.max(x)` in `_psislw`), so a copy is passed;
  * it returns k = inf whenever fewer than 5 draws fall in the tail (constant or few-valued
    log weights, tiny S), i.e. "could not fit", not evidence of a heavy tail.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import arviz as az
import jax
import numpy as np
from scipy.special import logsumexp

import tensorflow_probability.substrates.jax as tfp_jax

tfd = tfp_jax.distributions

# Constant log ratios (identical priors) are detected with this absolute spread tolerance.
CONSTANT_RATIO_ATOL = 1e-10


@dataclasses.dataclass(frozen=True)
class PriorFactor:
    """One changed prior on one sampled variable.

    Attributes:
      name: Key of the variable in the `draws` mapping (e.g. "roi_m").
      reference: Prior used in the reference fit, as the model actually used it.
      target: Prior to evaluate. Same support (or a subset), same sampled variable.
    """

    name: str
    reference: tfd.Distribution
    target: tfd.Distribution


@dataclasses.dataclass(frozen=True)
class ReweightResult:
    log_ratio: np.ndarray  # (S,) raw joint log ratio
    raw_weights: np.ndarray  # (S,) normalized
    psis_weights: np.ndarray | None  # (S,) normalized; None when PSIS was not applied
    pareto_k: float  # nan when PSIS was not applied
    raw_ess: float
    psis_ess: float
    ess_fraction: float  # psis_ess / S
    n_draws: int
    n_zero_weight: int  # draws with target density exactly 0 (log ratio -inf)
    status: str  # "identical" | "bounded" | "good" | "warning" | "fail"
    k_threshold: float  # threshold used for `status`, from the n_draws - n_zero_weight draws PSIS saw
    factor_log_ratios: Mapping[str, np.ndarray]  # per-factor (S,) contributions


def _unwrap(dist):
    while hasattr(dist, "distribution") and dist.distribution is not dist:
        dist = dist.distribution
    return dist


def good_k_threshold(n_draws: int) -> float:
    """min(1 - 1/log10(S), 0.7): the PSIS "good" bound used by `az.loo` (stats.py) and
    attributed there to Vehtari et al. (2024). Equivalent to S > 10^(1/(1-k)) for k < 0.7."""
    return float(min(1.0 - 1.0 / np.log10(max(n_draws, 2)), 0.7))


def relative_efficiency(chain_draw_arrays: Sequence[np.ndarray]) -> float:
    """Relative MCMC efficiency (ESS/S) for `reweight(reff=...)`, following `az.loo`.

    `az.loo` takes the mean, over all posterior variables, of the "mean"-method ESS
    divided by the number of draws. Pass the reference-fit arrays shaped
    (chain, draw, ...), typically those of the sampled variables whose priors change.
    The result is capped at 1.0 (ESS above S is not used to shrink the PSIS tail).
    """
    ess, n_total = [], None
    for a in chain_draw_arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim < 2:
            raise ValueError("Expected arrays shaped (chain, draw, ...).")
        n = a.shape[0] * a.shape[1]
        if n_total not in (None, n):
            raise ValueError("Arrays have different numbers of draws.")
        n_total = n
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        ess.extend(
            float(az.ess(flat[:, :, i], method="mean")) for i in range(flat.shape[-1])
        )
    if not ess:
        raise ValueError("No arrays supplied.")
    return float(min(np.mean(ess) / n_total, 1.0))


def _factor_log_ratio(factor: PriorFactor, x: np.ndarray) -> np.ndarray:
    for label, dist in (("reference", factor.reference), ("target", factor.target)):
        if isinstance(_unwrap(dist), tfd.Deterministic):
            raise ValueError(
                f"Factor '{factor.name}': {label} prior is Deterministic (a point mass); "
                "it has no density and cannot enter an importance ratio."
            )
    if not np.all(np.isfinite(x)):
        raise ValueError(f"Factor '{factor.name}': draws contain non-finite values.")
    lp0 = np.asarray(factor.reference.log_prob(x), dtype=np.float64)
    lp1 = np.asarray(factor.target.log_prob(x), dtype=np.float64)
    if lp0.shape != lp1.shape or lp0.shape[:1] != x.shape[:1]:
        raise ValueError(
            f"Factor '{factor.name}': log_prob shapes differ or lost the draw axis "
            f"(reference {lp0.shape}, target {lp1.shape}, draws {x.shape})."
        )
    if np.any(np.isnan(lp0)) or np.any(np.isneginf(lp0)) or np.any(np.isposinf(lp0)):
        raise ValueError(
            f"Factor '{factor.name}': reference log density is not finite on the draws "
            "(draws outside the reference support, or an invalid reference prior)."
        )
    if np.any(np.isnan(lp1)) or np.any(np.isposinf(lp1)):
        raise ValueError(
            f"Factor '{factor.name}': target log density is NaN or +inf on the draws "
            "(invalid target parameters, e.g. non-positive scale)."
        )
    diff = lp1 - lp0  # -inf allowed: target density is 0 there
    return diff.reshape(diff.shape[0], -1).sum(axis=1)


def reweight(
    draws: Mapping[str, np.ndarray],
    factors: Sequence[PriorFactor],
    reff: float = 1.0,
) -> ReweightResult:
    """Raw IS and PSIS weights for the combined changed prior factors.

    Args:
      draws: Flattened posterior draws, name -> array of shape (S, ...). The leading
        axis is the draw axis; chain and draw must already be merged.
      factors: Changed prior factors, each used exactly once.
      reff: Relative MCMC efficiency passed to `az.psislw`. 1.0 treats draws as
        independent; use the relative MCMC ESS of the reference fit when known.
    """
    if not jax.config.jax_enable_x64:
        # Meridian enables x64 on import; without it TFP evaluates log densities in float32.
        raise RuntimeError(
            "JAX x64 is disabled; log density ratios would be computed in float32. "
            "Import meridian (or enable jax_enable_x64) before using this module."
        )
    if not factors:
        raise ValueError("At least one changed prior factor is required.")
    names = [f.name for f in factors]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate factor names {names}; each factor may appear once.")
    missing = [n for n in names if n not in draws]
    if missing:
        raise KeyError(f"No draws supplied for factors: {missing}")
    sizes = {n: np.asarray(draws[n]).shape[0] for n in names}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Factors have different numbers of draws: {sizes}")
    n_draws = next(iter(sizes.values()))
    if n_draws < 2:
        raise ValueError("Need at least 2 draws.")

    per_factor = {
        f.name: _factor_log_ratio(f, np.asarray(draws[f.name], dtype=np.float64))
        for f in factors
    }
    log_ratio = np.sum(list(per_factor.values()), axis=0)

    if np.all(np.isneginf(log_ratio)):
        raise ValueError("Target prior has zero density on every reference draw.")
    finite = np.isfinite(log_ratio)
    n_zero = int(n_draws - finite.sum())

    raw_lw = log_ratio - logsumexp(log_ratio)
    raw_w = np.exp(raw_lw)
    raw_ess = float(1.0 / np.sum(raw_w**2))

    lr_f = log_ratio[finite]
    n_f = int(lr_f.size)
    k_thr = good_k_threshold(n_f)

    if np.ptp(lr_f) <= CONSTANT_RATIO_ATOL:
        # Weights are constant on the support: identical priors (n_zero == 0) or a pure
        # truncation (n_zero > 0, weights in {0, c}). Bounded weights, nothing to smooth,
        # and az.psislw would return k = inf, which is "no tail to fit", not a failure.
        return ReweightResult(
            log_ratio=log_ratio, raw_weights=raw_w, psis_weights=None,
            pareto_k=float("nan"), raw_ess=raw_ess, psis_ess=raw_ess,
            ess_fraction=raw_ess / n_draws, n_draws=n_draws, n_zero_weight=n_zero,
            status="identical" if n_zero == 0 else "bounded",
            k_threshold=k_thr, factor_log_ratios=per_factor,
        )

    # PSIS is fitted on the draws with positive target density only. Zero-weight draws
    # (-inf) would otherwise count towards S and turn the whole positive set into the
    # "tail". This subsetting is our adaptation, not part of the published method.
    psis_f, k = az.psislw(lr_f.copy(), reff=reff)  # copy: psislw mutates its input
    psis_f = np.asarray(psis_f, dtype=np.float64)
    if np.any(np.isnan(psis_f)) or np.any(np.isposinf(psis_f)):
        raise ValueError("PSIS produced NaN or +inf log weights.")
    psis_lw = np.full(n_draws, -np.inf)
    psis_lw[finite] = psis_f
    psis_w = np.exp(psis_lw - logsumexp(psis_lw))
    psis_ess = float(1.0 / np.sum(psis_w**2))
    k = float(np.asarray(k))

    if not np.isfinite(k) or k >= 1.0:
        status = "fail"
    elif k > k_thr:
        status = "warning"
    else:
        status = "good"

    return ReweightResult(
        log_ratio=log_ratio, raw_weights=raw_w, psis_weights=psis_w, pareto_k=k,
        raw_ess=raw_ess, psis_ess=psis_ess, ess_fraction=psis_ess / n_draws,
        n_draws=n_draws, n_zero_weight=n_zero, status=status, k_threshold=k_thr,
        factor_log_ratios=per_factor,
    )
