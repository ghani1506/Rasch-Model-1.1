"""
Rasch analysis engine — dichotomous, Partial Credit and Rating Scale models.

The data type is detected per question and the model is chosen automatically:
  * Dichotomous Rasch  — every question is right/wrong (MCQ letters or 0/1)
  * Rating Scale Model — every question uses the same multi-category scale (e.g. Likert 1-5)
  * Partial Credit Model — questions have different numbers of score categories
                           (e.g. MCQ + structured questions out of 2, 3, 4 ... marks)

Estimation: Joint Maximum Likelihood (JML / UCON, as in Winsteps) with (L-1)/L bias
correction, missing data handled, extreme scores estimated with a 0.3 adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXTREME_ADJ = 0.3
LETTERS = list("ABCDEFGH")
MISSING_TOKENS = {"", "NAN", "NONE", "-", ".", "NA", "N/A"}

MODEL_DICHOTOMOUS = "Dichotomous Rasch"
MODEL_PCM = "Partial Credit Model"
MODEL_RSM = "Rating Scale Model"


# ===========================================================================
# 1. Data preparation & automatic model selection
# ===========================================================================
@dataclass
class PreparedData:
    scored: pd.DataFrame                 # 0..m_i integer scores (float, NaN = missing)
    max_score: np.ndarray                # m_i per item
    col_types: dict                      # item -> "mcq" | "dichotomous" | "polytomous"
    letters: pd.DataFrame | None         # cleaned letters for MCQ columns
    key: dict                            # item -> correct letter (MCQ only)
    auto_model: str
    allowed_models: list
    notes: list = field(default_factory=list)


def _clean(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.where(~s.isin(MISSING_TOKENS), np.nan)


def column_type(series: pd.Series) -> str | None:
    """'mcq' (letters), 'numeric' (integer scores) or None (not an item column)."""
    s = _clean(series).dropna()
    s = s[s != "KEY"]
    if len(s) == 0:
        return None
    if s.isin(LETTERS).mean() > 0.9:
        return "mcq"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95:
        num = num.dropna()
        if (num == num.round()).all() and num.min() >= 0 and num.max() <= 20:
            return "numeric"
    return None


def prepare_data(items: pd.DataFrame, key_row: dict | None = None) -> PreparedData:
    """
    items: raw response columns (strings). key_row: optional {item: value} from the KEY row —
    a letter for MCQ items, optionally the maximum mark for numeric items.
    """
    key_row = key_row or {}
    notes, col_types, key = [], {}, {}
    letters = {}
    raw_num = {}
    for c in items.columns:
        t = column_type(items[c])
        if t == "mcq":
            letters[c] = _clean(items[c])
            k = str(key_row.get(c, "")).strip().upper()
            if k not in LETTERS:
                raise ValueError(f"Question {c} has letter answers but no valid answer key.")
            key[c] = k
        elif t == "numeric":
            raw_num[c] = pd.to_numeric(_clean(items[c]), errors="coerce")
        else:
            raise ValueError(f"Column {c} is neither MCQ letters nor whole-number scores.")

    # Likert-type scales starting at 1 (no zeros anywhere, every numeric item min is 1) -> shift to 0
    if raw_num:
        mins = [v.min() for v in raw_num.values()]
        if all(m >= 1 for m in mins) and min(mins) == 1 and max(v.max() for v in raw_num.values()) > 1 \
                and not letters:
            raw_num = {c: v - 1 for c, v in raw_num.items()}
            notes.append("Scores start at 1 (rating scale) — recoded so the lowest category is 0.")

    scored = pd.DataFrame(index=items.index)
    max_score = []
    for c in items.columns:
        if c in letters:
            l = letters[c]
            scored[c] = np.where(l.isna(), np.nan, (l == key[c]).astype(float))
            col_types[c] = "mcq"
            max_score.append(1)
            continue
        v = raw_num[c].copy()
        declared = pd.to_numeric(pd.Series([key_row.get(c, "")]), errors="coerce").iloc[0]
        top = int(max(v.max(), declared if not np.isnan(declared) else 0))
        observed = sorted(int(x) for x in v.dropna().unique())
        full = list(range(0, top + 1))
        unobserved = [k for k in full if k not in observed]
        if unobserved:
            # collapse unused categories (they carry no information and cannot be estimated)
            mapping = {o: i for i, o in enumerate(observed)}
            v = v.map(mapping)
            notes.append(f"{c}: score(s) {', '.join(map(str, unobserved))} never awarded — "
                         f"categories renumbered 0–{len(observed) - 1}.")
            top = len(observed) - 1
        scored[c] = v.astype(float)
        col_types[c] = "dichotomous" if top == 1 else "polytomous"
        max_score.append(top)

    max_score = np.array(max_score, dtype=int)
    if (max_score < 1).any():
        bad = [c for c, m in zip(items.columns, max_score) if m < 1]
        raise ValueError(f"Question(s) {', '.join(bad)} have only one score value — remove them.")

    if (max_score == 1).all():
        auto, allowed = MODEL_DICHOTOMOUS, [MODEL_DICHOTOMOUS]
    elif len(set(max_score)) == 1:
        auto, allowed = MODEL_RSM, [MODEL_RSM, MODEL_PCM]
    else:
        auto, allowed = MODEL_PCM, [MODEL_PCM]

    return PreparedData(scored, max_score, col_types,
                        pd.DataFrame(letters) if letters else None, key, auto, allowed, notes)


def explain_model_choice(prep: PreparedData) -> str:
    n = len(prep.max_score)
    n_mcq = sum(t == "mcq" for t in prep.col_types.values())
    n_di = sum(t == "dichotomous" for t in prep.col_types.values())
    n_poly = sum(t == "polytomous" for t in prep.col_types.values())
    parts = []
    if n_mcq:
        parts.append(f"{n_mcq} MCQ (letters, scored with the key)")
    if n_di:
        parts.append(f"{n_di} right/wrong (0/1)")
    if n_poly:
        maxes = sorted(set(int(m) for m, t in zip(prep.max_score, prep.col_types.values()) if t == "polytomous"))
        parts.append(f"{n_poly} multi-mark (max {', '.join(map(str, maxes))})")
    detected = f"{n} questions: " + "; ".join(parts) + "."
    why = {
        MODEL_DICHOTOMOUS: "Every question is scored right/wrong → **Dichotomous Rasch model**.",
        MODEL_RSM: "Every question uses the same multi-category scale → **Rating Scale Model** (Andrich): "
                   "all questions share one set of category thresholds.",
        MODEL_PCM: "Questions have different numbers of score categories → **Partial Credit Model** (Masters): "
                   "each question gets its own thresholds.",
    }[prep.auto_model]
    return detected + " " + why


# ===========================================================================
# 2. Core polytomous Rasch machinery
# ===========================================================================
def category_probs(theta: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """P(X = k | theta) for k = 0..m with Andrich step difficulties `steps` (length m)."""
    theta = np.atleast_1d(theta).astype(float)
    cum = np.cumsum(theta[:, None] - steps[None, :], axis=1)
    logits = np.concatenate([np.zeros((len(theta), 1)), cum], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def _moments(theta, steps):
    P = category_probs(theta, steps)
    k = np.arange(P.shape[1])
    E = P @ k
    V = P @ (k ** 2) - E ** 2
    C = (P * (k[None, :] - E[:, None]) ** 4).sum(1)
    return P, E, V, C


def _solve_location(target, fixed_values, steps_list, person_side, init=0.0):
    """
    Newton solve for one location parameter given the opposite parameters.
    person_side=True : solve theta with item steps list (one per observed item)
    person_side=False: solve item shift d with fixed person thetas and base steps (tau)
    """
    t = init
    for _ in range(200):
        if person_side:
            E = V = 0.0
            for st in steps_list:
                _, e, v, _ = _moments(np.array([t]), st)
                E += e[0]; V += v[0]
            step = np.clip((target - E) / max(V, 1e-9), -1, 1)
            t += step
        else:
            _, e, v, _ = _moments(fixed_values, steps_list + t)
            step = np.clip((e.sum() - target) / max(v.sum(), 1e-9), -1, 1)
            t += step
        if abs(step) < 1e-7:
            break
    return t


@dataclass
class RaschResult:
    model: str
    item_names: list
    person_ids: list
    X: np.ndarray
    max_score: np.ndarray
    theta: np.ndarray
    delta: np.ndarray            # item locations (mean of steps)
    steps: list                  # per-item Andrich step difficulties (absolute)
    theta_se: np.ndarray
    delta_se: np.ndarray
    item_table: pd.DataFrame
    person_table: pd.DataFrame
    threshold_table: pd.DataFrame
    summary: dict
    iterations: int
    converged: bool
    col_types: dict = field(default_factory=dict)
    distractors: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def beta(self):  # backwards-compatible alias
        return self.delta


def fit_rasch(prep: PreparedData, person_ids=None, model: str | None = None,
              max_iter: int = 500, tol: float = 1e-4) -> RaschResult:
    model = model or prep.auto_model
    if model not in prep.allowed_models:
        raise ValueError(f"{model} is not valid for this data. Allowed: {prep.allowed_models}")
    X = prep.scored.to_numpy(dtype=float)
    N, L = X.shape
    m = prep.max_score.astype(int)
    M = ~np.isnan(X)
    Xf = np.where(M, X, 0.0)
    item_names = list(prep.scored.columns)
    person_ids = list(person_ids) if person_ids is not None else [f"P{i + 1}" for i in range(N)]
    rsm = model == MODEL_RSM

    # ---- identify extreme persons/items (iteratively) ----
    p_keep = np.ones(N, bool)
    i_keep = np.ones(L, bool)
    for _ in range(30):
        Mk = M & p_keep[:, None] & i_keep[None, :]
        r = (Xf * Mk).sum(1); rmax = (Mk * m[None, :]).sum(1)
        s = (Xf * Mk).sum(0); smax = (Mk * m[None, :]).sum(0)
        new_p = p_keep & (r > 0) & (r < rmax)
        new_i = i_keep & (s > 0) & (s < smax)
        if (new_p == p_keep).all() and (new_i == i_keep).all():
            break
        p_keep, i_keep = new_p, new_i
    Mk = M & p_keep[:, None] & i_keep[None, :]
    r = (Xf * Mk).sum(1)
    rmax = (Mk * m[None, :]).sum(1)

    # observed counts of "score >= k" per item & step (for threshold estimation)
    obs_ge = [np.array([((Xf[:, i] >= k) & Mk[:, i]).sum() for k in range(1, m[i] + 1)], float) for i in range(L)]

    # ---- starting values ----
    theta = np.zeros(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta[p_keep] = np.log(r[p_keep] / (rmax[p_keep] - r[p_keep]))
    delta = np.zeros(L)
    s = (Xf * Mk).sum(0); smax = (Mk * m[None, :]).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta[i_keep] = -np.log(s[i_keep] / (smax[i_keep] - s[i_keep]))
    delta[i_keep] -= delta[i_keep].mean()
    if rsm:
        tau_shared = np.zeros(m[0])
        tau = [tau_shared.copy() for _ in range(L)]
    else:
        tau = [np.zeros(m[i]) for i in range(L)]

    converged, it = False, 0
    for it in range(1, max_iter + 1):
        # ---- person step ----
        Esum = np.zeros(N); Vsum = np.zeros(N)
        for i in np.where(i_keep)[0]:
            _, E, V, _ = _moments(theta, delta[i] + tau[i])
            Esum += E * Mk[:, i]; Vsum += V * Mk[:, i]
        d_th = np.zeros(N)
        d_th[p_keep] = np.clip((r - Esum)[p_keep] / np.maximum(Vsum[p_keep], 1e-9), -1, 1)
        theta += d_th

        # ---- item step ----
        max_change = np.abs(d_th).max()
        if rsm:
            num_tau = np.zeros(m[0]); den_tau = np.zeros(m[0])
        for i in np.where(i_keep)[0]:
            w = Mk[:, i]
            P, E, V, _ = _moments(theta[w], delta[i] + tau[i])
            Pge = np.cumsum(P[:, ::-1], axis=1)[:, ::-1][:, 1:]       # P(X >= k), k=1..m
            exp_ge = Pge.sum(0)
            info_ge = (Pge * (1 - Pge)).sum(0)
            if rsm:
                dd = np.clip((E.sum() - Xf[w, i].sum()) / max(V.sum(), 1e-9), -1, 1)
                delta[i] += dd
                num_tau += exp_ge - obs_ge[i]; den_tau += info_ge
                max_change = max(max_change, abs(dd))
            else:
                # full Newton on this item's steps: H_kl = sum_n Cov(1[X>=k], 1[X>=l])
                mi = m[i]
                if mi == 1:
                    ds = np.clip((exp_ge - obs_ge[i]) / np.maximum(info_ge, 1e-9), -1, 1)
                else:
                    kk = np.maximum.outer(np.arange(mi), np.arange(mi))
                    H = Pge[:, kk].sum(0) - Pge.T @ Pge
                    H += np.eye(mi) * 1e-6
                    try:
                        ds = np.linalg.solve(H, exp_ge - obs_ge[i])
                    except np.linalg.LinAlgError:
                        ds = (exp_ge - obs_ge[i]) / np.maximum(info_ge, 1e-9)
                    ds = np.clip(ds, -1, 1)
                steps_new = np.clip(delta[i] + tau[i] + ds, -15, 15)
                delta[i] = steps_new.mean(); tau[i] = steps_new - delta[i]
                max_change = max(max_change, np.abs(ds).max())
        if rsm and m[0] > 1:
            dt = np.clip(num_tau / np.maximum(den_tau, 1e-9), -1, 1)
            tau_shared = tau_shared + dt
            tau_shared -= tau_shared.mean()
            tau = [tau_shared.copy() for _ in range(L)]
            max_change = max(max_change, np.abs(dt).max())
        delta[i_keep] -= delta[i_keep].mean()
        if max_change < tol:
            converged = True
            break

    # ---- JML bias correction ----
    Lk = max(i_keep.sum(), 2)
    corr = (Lk - 1) / Lk
    theta[p_keep] *= corr
    delta[i_keep] *= corr
    tau = [t * corr for t in tau]

    # ---- extreme items: location with zero/shared thresholds ----
    for j in np.where(~i_keep)[0]:
        w = M[:, j] & p_keep
        if w.sum() == 0:
            delta[j] = np.nan; continue
        if not rsm:
            tau[j] = np.zeros(m[j])
        sc = Xf[w, j].sum(); tot = w.sum() * m[j]
        target = EXTREME_ADJ if sc == 0 else tot - EXTREME_ADJ
        delta[j] = _solve_location(target, theta[w], tau[j], person_side=False)

    # ---- extreme persons ----
    steps = [delta[i] + tau[i] for i in range(L)]
    valid_items = ~np.isnan(delta)
    for n in np.where(~p_keep)[0]:
        obs = np.where(M[n] & valid_items)[0]
        if len(obs) == 0:
            theta[n] = np.nan; continue
        sc = Xf[n, obs].sum(); tot = m[obs].sum()
        target = EXTREME_ADJ if sc == 0 else tot - EXTREME_ADJ
        theta[n] = _solve_location(target, None, [steps[i] for i in obs], person_side=True)

    # ---- model quantities on all observed cells ----
    tt = np.nan_to_num(theta)
    E = np.zeros((N, L)); V = np.zeros((N, L)); C = np.zeros((N, L))
    for i in range(L):
        if np.isnan(delta[i]):
            continue
        _, E[:, i], V[:, i], C[:, i] = _moments(tt, steps[i])
    Vm = np.where(M, V, 0.0); Cm = np.where(M, C, 0.0)
    R2 = np.where(M, (Xf - E) ** 2, 0.0)
    theta_se = 1 / np.sqrt(np.maximum(Vm.sum(1), 1e-9))
    delta_se = 1 / np.sqrt(np.maximum(Vm.sum(0), 1e-9))

    def fit_stats(axis, other_keep):
        Mm = M & (other_keep[None, :] if axis == 1 else other_keep[:, None])
        n = np.maximum(Mm.sum(axis), 1)
        Vs = np.where(Mm, V, 0.0); Cs = np.where(Mm, C, 0.0)
        outfit = np.where(Mm, R2 / np.maximum(V, 1e-9), 0).sum(axis) / n
        infit = np.where(Mm, R2, 0).sum(axis) / np.maximum(Vs.sum(axis), 1e-9)
        q_out = np.sqrt(np.maximum(np.where(Mm, Cs / np.maximum(V, 1e-9) ** 2, 0).sum(axis) / n ** 2 - 1 / n, 1e-9))
        q_in = np.sqrt(np.maximum((Cs - Vs ** 2).sum(axis) / np.maximum(Vs.sum(axis), 1e-9) ** 2, 1e-9))
        zout = (np.cbrt(outfit) - 1) * (3 / q_out) + q_out / 3
        zin = (np.cbrt(infit) - 1) * (3 / q_in) + q_in / 3
        return infit, outfit, zin, zout

    i_in, i_out, i_zin, i_zout = fit_stats(0, p_keep)
    p_in, p_out, p_zin, p_zout = fit_stats(1, i_keep)

    ptmeas = np.full(L, np.nan)
    for j in range(L):
        w = M[:, j] & ~np.isnan(theta)
        if w.sum() > 2 and np.std(Xf[w, j]) > 0:
            ptmeas[j] = np.corrcoef(Xf[w, j], theta[w])[0, 1]

    n_resp = M.sum(0)
    mean_score = np.where(n_resp > 0, (Xf * M).sum(0) / np.maximum(n_resp, 1), np.nan)
    facility = mean_score / m
    raw = (Xf * M).sum(1)
    raw_max = (M * m[None, :]).sum(1)

    def rel_sep(meas, se, keep):
        mm, e = meas[keep], se[keep]
        if len(mm) < 2:
            return np.nan, np.nan
        ov = np.var(mm, ddof=1); ev = np.mean(e ** 2); tv = max(ov - ev, 0)
        return (tv / ov if ov > 0 else np.nan), (np.sqrt(tv / ev) if ev > 0 else np.nan)

    p_rel, p_sep = rel_sep(theta, theta_se, p_keep & ~np.isnan(theta))
    i_rel, i_sep = rel_sep(delta, delta_se, i_keep & ~np.isnan(delta))
    with np.errstate(invalid="ignore"):
        iv = np.nanvar(X, axis=0, ddof=1); tv = np.var(np.nansum(X, axis=1), ddof=1)
        alpha = (L / (L - 1)) * (1 - np.nansum(iv) / tv) if tv > 0 else np.nan

    # thresholds table & ordering
    th_rows, disordered = [], np.zeros(L, bool)
    for i in range(L):
        st = steps[i]
        if len(st) > 1 and np.any(np.diff(st) <= 0) and i_keep[i]:
            disordered[i] = True
        counts = [int(((Xf[:, i] == k) & M[:, i]).sum()) for k in range(m[i] + 1)]
        for k in range(1, m[i] + 1):
            th_rows.append({"Item": item_names[i], "Step": f"{k - 1}→{k}", "Threshold": st[k - 1],
                            "Count in category": counts[k], "Ordered": not disordered[i]})

    item_table = pd.DataFrame({
        "Item": item_names,
        "Type": [prep.col_types[c] for c in item_names],
        "Key": [prep.key.get(c, "") for c in item_names],
        "Max": m,
        "N": n_resp,
        "Mean score": mean_score,
        "Facility": facility,
        "Measure": delta, "SE": delta_se,
        "Infit MNSQ": i_in, "Infit ZSTD": i_zin,
        "Outfit MNSQ": i_out, "Outfit ZSTD": i_zout,
        "Pt-Measure": ptmeas,
        "Disordered": disordered,
        "Extreme": ~i_keep,
    })
    person_table = pd.DataFrame({
        "ID": person_ids, "Raw score": raw, "Max possible": raw_max, "Answered": M.sum(1),
        "Measure": theta, "SE": theta_se,
        "Infit MNSQ": p_in, "Outfit MNSQ": p_out, "Outfit ZSTD": p_zout, "Extreme": ~p_keep,
    })
    summary = dict(
        model=model, n_persons=N, n_items=L,
        n_extreme_persons=int((~p_keep).sum()), n_extreme_items=int((~i_keep).sum()),
        person_mean=float(np.nanmean(theta)), person_sd=float(np.nanstd(theta, ddof=1)),
        item_mean=float(np.nanmean(delta)), item_sd=float(np.nanstd(delta, ddof=1)),
        person_reliability=float(p_rel), person_separation=float(p_sep),
        item_reliability=float(i_rel), item_separation=float(i_sep),
        alpha=float(alpha), raw_mean=float(raw.mean()), raw_sd=float(raw.std(ddof=1)),
        max_total=int(m.sum()), missing_pct=float(100 * (1 - M.mean())),
        targeting=float(np.nanmean(theta) - np.nanmean(delta)),
    )
    res = RaschResult(model, item_names, person_ids, X, m, theta, delta, steps, theta_se, delta_se,
                      item_table, person_table, pd.DataFrame(th_rows), summary, it, converged,
                      prep.col_types, notes=list(prep.notes))
    if prep.letters is not None:
        res.distractors = distractor_analysis(prep.letters, prep.key, theta)
    res.item_table["Flags"] = flag_items(res)
    res.item_table["Verdict"] = [verdict(f) for f in res.item_table["Flags"]]
    return res


# ===========================================================================
# 3. Distractor analysis (MCQ items)
# ===========================================================================
def distractor_analysis(letters: pd.DataFrame, key: dict, theta: np.ndarray) -> dict:
    out = {}
    options = sorted({v for v in pd.unique(letters.values.ravel()) if isinstance(v, str)})
    th = np.nan_to_num(theta)
    for col in letters.columns:
        k = key[col]
        vals = letters[col]
        n_total = vals.notna().sum()
        rows = []
        for o in options + ["(blank)"]:
            sel = vals.isna().to_numpy() if o == "(blank)" else (vals == o).to_numpy()
            cnt = int(sel.sum())
            if o == "(blank)" and cnt == 0:
                continue
            pb = np.corrcoef(sel.astype(float), th)[0, 1] if 0 < cnt < len(sel) else np.nan
            rows.append({"Option": o, "Key": "✓" if o == k else "", "Count": cnt,
                         "%": 100 * cnt / (len(sel) if o == "(blank)" else max(n_total, 1)),
                         "Mean ability": float(np.nanmean(theta[sel])) if cnt else np.nan,
                         "Pt-biserial": pb})
        out[col] = pd.DataFrame(rows)
    return out


# ===========================================================================
# 4. Flags & verdicts
# ===========================================================================
SERIOUS = ("Negative pt-measure", "Possible wrong key", "Underfit", "Everyone", "Nobody",
           "attracts able", "also correct", "Disordered")


def flag_items(res: RaschResult) -> list[str]:
    out = []
    for _, row in res.item_table.iterrows():
        f = []
        if row["Extreme"]:
            f.append("Everyone got full marks" if row["Facility"] >= 0.999 else "Nobody scored")
        pm = row["Pt-Measure"]
        if not np.isnan(pm) and pm < 0:
            f.append("Negative pt-measure" + (" (reverse-worded item not recoded?)" if res.model == MODEL_RSM else ""))
        elif not np.isnan(pm) and pm < 0.15:
            f.append("Weak discrimination")
        zmax = np.nanmax([row["Infit ZSTD"], row["Outfit ZSTD"]])
        zmin = np.nanmin([row["Infit ZSTD"], row["Outfit ZSTD"]])
        if not row["Extreme"] and (row["Infit MNSQ"] > 1.2 or row["Outfit MNSQ"] > 1.3) and zmax > 2:
            f.append("Underfit (noisy)")
        if not row["Extreme"] and (row["Infit MNSQ"] < 0.8 or row["Outfit MNSQ"] < 0.7) and zmin < -2:
            f.append("Overfit (too predictable)")
        if row["Disordered"]:
            f.append("Disordered thresholds (a score category is rarely the most likely — check marking scheme)")
        if row["Facility"] > 0.95 and not row["Extreme"]:
            f.append("Very easy")
        if row["Facility"] < 0.10 and not row["Extreme"]:
            f.append("Very hard")
        d = res.distractors.get(row["Item"])
        if d is not None and len(d):
            kr = d[d["Key"] == "✓"]
            k_avg = kr["Mean ability"].iloc[0] if len(kr) else np.nan
            k_pct = kr["%"].iloc[0] if len(kr) else 0
            for _, o in d[(d["Key"] == "") & (d["Option"] != "(blank)")].iterrows():
                if o["%"] >= 10 and not np.isnan(k_avg) and o["Mean ability"] > k_avg:
                    f.append(f"Possible wrong key ({o['Option']})" if o["%"] > k_pct
                             else f"Distractor {o['Option']} attracts able students")
                elif o["%"] >= 10 and not np.isnan(o["Pt-biserial"]) and o["Pt-biserial"] >= 0.10:
                    f.append(f"Distractor {o['Option']} discriminates positively (also correct?)")
                if o["%"] < 3 and row["Facility"] < 0.9:
                    f.append(f"Non-functioning distractor {o['Option']}")
        out.append("; ".join(f))
    return out


def verdict(flags: str) -> str:
    if not flags:
        return "Keep"
    return "Review / revise" if any(s in flags for s in SERIOUS) else "Minor issue"


# ===========================================================================
# 5. DIF
# ===========================================================================
def dif_analysis(res: RaschResult, groups: pd.Series, focal: str, reference: str) -> pd.DataFrame:
    """
    Separate item calibration per group with person measures and thresholds anchored.
    Contrast = location(focal) - location(reference); ETS-style classes in logits:
    A < 0.43, B 0.43-0.64, C >= 0.64 (with |t| >= 2).
    """
    X = res.X; M = ~np.isnan(X); Xf = np.where(M, X, 0)
    th = np.nan_to_num(res.theta)
    g = groups.astype(str).to_numpy()
    rows = []
    for j, name in enumerate(res.item_names):
        tau = res.steps[j] - res.delta[j] if not np.isnan(res.delta[j]) else np.zeros(res.max_score[j])
        est = {}
        for lab in (focal, reference):
            w = M[:, j] & (g == lab)
            if w.sum() < 10:
                est[lab] = (np.nan, np.nan); continue
            tot = w.sum() * res.max_score[j]
            target = min(max(Xf[w, j].sum(), EXTREME_ADJ), tot - EXTREME_ADJ)
            d = _solve_location(target, th[w], tau, person_side=False,
                                init=res.delta[j] if not np.isnan(res.delta[j]) else 0.0)
            _, _, v, _ = _moments(th[w], tau + d)
            est[lab] = (d, 1 / np.sqrt(max(v.sum(), 1e-9)))
        (bf, sf), (br, sr) = est[focal], est[reference]
        c = bf - br
        t = c / np.sqrt(sf ** 2 + sr ** 2) if not np.isnan(c) else np.nan
        a = abs(c)
        cls = "" if np.isnan(a) else ("C (large)" if a >= 0.64 and abs(t) >= 2 else
                                      "B (moderate)" if a >= 0.43 and abs(t) >= 2 else "A (negligible)")
        rows.append({"Item": name, f"Measure ({focal})": bf, f"Measure ({reference})": br,
                     "Contrast": c, "t": t, "DIF class": cls,
                     "Harder for": (focal if c > 0 else reference) if cls[:1] in ("B", "C") else ""})
    return pd.DataFrame(rows)


# ===========================================================================
# 6. Curves for plotting
# ===========================================================================
def empirical_icc(res: RaschResult, j: int, n_bins: int = 8) -> pd.DataFrame:
    x = res.X[:, j]
    w = ~np.isnan(x) & ~np.isnan(res.theta)
    th, x = res.theta[w], x[w]
    qs = np.unique(np.quantile(th, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(th, qs[1:-1]), 0, len(qs) - 2)
    rows = [{"Ability": th[idx == b].mean(), "Observed": x[idx == b].mean(), "N": int((idx == b).sum())}
            for b in range(len(qs) - 1) if (idx == b).any()]
    return pd.DataFrame(rows)


def expected_curve(res: RaschResult, j: int, lo=-5, hi=5, n=200):
    t = np.linspace(lo, hi, n)
    P, E, _, _ = _moments(t, res.steps[j])
    return t, E, P
