"""
Generate 5 practice datasets (1,000 students each) with known, planted features.
Run:  python generate_datasets.py
See data/PRACTICE_ANSWERS.md for what is hidden in each dataset (try first!).
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
OPTS = ["A", "B", "C", "D"]
N = 1000


def sig(z):
    return 1 / (1 + np.exp(-z))


def base_frame(rng, n=N, classes=8):
    ids = [f"S{i+1:04d}" for i in range(n)]
    cls = [f"4{chr(65 + (i % classes))}" for i in range(n)]
    rng.shuffle(cls)
    gender = rng.choice(["M", "F"], n)
    return pd.DataFrame({"StudentID": ids, "Class": cls, "Gender": gender})


def make_key(rng, n_items):
    return list(rng.choice(OPTS, n_items))


def wrong_options(k):
    return [o for o in OPTS if o != k]


def simulate_mcq(rng, theta, b, key, dist_w=None, special=None, slope=None):
    """
    theta: abilities (N,), b: difficulties (I,), key: list of correct letters.
    dist_w: dict item_index -> weights for the 3 wrong options (ordered as wrong_options(key)).
    special: dict item_index -> function(rng, theta_n, b_j, key_j) -> letter.
    slope: optional per-item discrimination (Rasch assumes 1).
    """
    n, I = len(theta), len(b)
    resp = np.empty((n, I), dtype=object)
    dist_w = dist_w or {}
    special = special or {}
    slope = np.ones(I) if slope is None else slope
    for j in range(I):
        k = key[j]
        wrong = wrong_options(k)
        if j in special:
            for i in range(n):
                resp[i, j] = special[j](rng, theta[i], b[j], k)
            continue
        p = sig(slope[j] * (theta - b[j]))
        correct = rng.random(n) < p
        w = dist_w.get(j, [0.40, 0.33, 0.27])
        # weaker students are a bit more likely to choose the most attractive distractor
        picks = rng.choice(wrong, n, p=w)
        resp[:, j] = np.where(correct, k, picks)
    return resp


def add_missing(rng, resp, rate):
    mask = rng.random(resp.shape) < rate
    resp = resp.copy()
    resp[mask] = ""
    return resp


def save(df_meta, resp, key, name, items_prefix="Q"):
    I = resp.shape[1]
    cols = [f"{items_prefix}{j+1}" for j in range(I)]
    data = pd.concat([df_meta.reset_index(drop=True), pd.DataFrame(resp, columns=cols)], axis=1)
    if key is not None:
        key_row = {c: "" for c in df_meta.columns}
        key_row[df_meta.columns[0]] = "KEY"
        key_row.update(dict(zip(cols, key)))
        data = pd.concat([pd.DataFrame([key_row]), data], ignore_index=True)
    data.to_csv(OUT / name, index=False)
    print(f"saved {name}: {data.shape}")


# ---------------------------------------------------------------------------
# 1. Science - a GOOD paper (benchmark)
# ---------------------------------------------------------------------------
def ds1():
    rng = np.random.default_rng(101)
    meta = base_frame(rng)
    theta = rng.normal(0.3, 1.2, N)
    I = 40
    b = np.sort(rng.uniform(-2.4, 2.6, I)); rng.shuffle(b)
    key = make_key(rng, I)
    resp = simulate_mcq(rng, theta, b, key)
    resp = add_missing(rng, resp, 0.005)
    save(meta, resp, key, "01_science_good_paper.csv")


# ---------------------------------------------------------------------------
# 2. Maths - paper far TOO EASY for the cohort
# ---------------------------------------------------------------------------
def ds2():
    rng = np.random.default_rng(202)
    meta = base_frame(rng)
    theta = rng.normal(1.9, 1.0, N)
    I = 40
    b = rng.uniform(-3.0, 0.6, I)
    b[0] = -9.0  # Q1: everybody gets it right
    key = make_key(rng, I)
    dist_w = {j: [0.88, 0.10, 0.02] for j in range(I) if j % 3 == 0}  # many weak distractors
    resp = simulate_mcq(rng, theta, b, key, dist_w=dist_w)
    resp[:, 0] = key[0]
    resp = add_missing(rng, resp, 0.003)
    resp[:, 0] = key[0]
    save(meta, resp, key, "02_maths_too_easy.csv")


# ---------------------------------------------------------------------------
# 3. English - paper with FAULTY ITEMS
# ---------------------------------------------------------------------------
def ds3():
    rng = np.random.default_rng(303)
    meta = base_frame(rng)
    theta = rng.normal(0.5, 1.1, N)
    I = 40
    b = rng.uniform(-2.0, 2.2, I)
    true_key = make_key(rng, I)
    key = list(true_key)

    # Q7 & Q15: wrong answer key in the marking scheme
    for j in (6, 14):
        alt = wrong_options(true_key[j])
        key[j] = alt[0]

    # Q12: two defensible answers - able students split between key and option X
    j12 = 11
    second = wrong_options(key[j12])[1]

    def ambiguous(rng, th, bj, k):
        if rng.random() < sig(th - bj):
            return k if rng.random() < 0.5 else second
        return rng.choice([o for o in OPTS if o not in (k, second)])

    # Q23: trap item - able students over-think and choose a distractor
    j23 = 22
    trap = wrong_options(key[j23])[2]

    def trap_item(rng, th, bj, k):
        if rng.random() < sig(1.6 * (th - 0.6)):
            return trap
        return k if rng.random() < 0.45 else rng.choice([o for o in OPTS if o not in (k, trap)])

    # Q31: nonsense item - pure guessing
    def guess(rng, th, bj, k):
        return rng.choice(OPTS)

    special = {j12: ambiguous, j23: trap_item, 30: guess}
    dist_w = {3: [0.70, 0.29, 0.01], 18: [0.62, 0.37, 0.01], 26: [0.75, 0.24, 0.01]}
    resp = simulate_mcq(rng, theta, b, true_key, dist_w=dist_w, special=special)
    resp = add_missing(rng, resp, 0.02)
    save(meta, resp, key, "03_english_faulty_items.csv")


# ---------------------------------------------------------------------------
# 4. Geography - DIF by Gender and by Stream
# ---------------------------------------------------------------------------
def ds4():
    rng = np.random.default_rng(404)
    meta = base_frame(rng)
    meta["Stream"] = rng.choice(["Science", "Arts"], N, p=[0.45, 0.55])
    # real ability difference between streams (impact, NOT bias)
    theta = rng.normal(0.2, 1.1, N) + np.where(meta["Stream"] == "Science", 0.5, 0.0)
    I = 30
    b = rng.uniform(-2.0, 2.0, I)
    key = make_key(rng, I)
    fem = (meta["Gender"] == "F").to_numpy()
    arts = (meta["Stream"] == "Arts").to_numpy()
    shift = np.zeros((N, I))
    shift[fem, 4] += 1.0     # Q5 harder for girls (e.g. context: football statistics)
    shift[~fem, 13] += 0.9   # Q14 harder for boys (e.g. context: textile industry)
    shift[fem, 21] += 0.7    # Q22 harder for girls
    shift[arts, 8] += 1.1    # Q9 harder for Arts stream (needs a formula taught in Science)
    resp = np.empty((N, I), dtype=object)
    for j in range(I):
        p = sig(theta - b[j] - shift[:, j])
        correct = rng.random(N) < p
        picks = rng.choice(wrong_options(key[j]), N, p=[0.40, 0.33, 0.27])
        resp[:, j] = np.where(correct, key[j], picks)
    resp = add_missing(rng, resp, 0.005)
    meta = meta[["StudentID", "Class", "Gender", "Stream"]]
    save(meta, resp, key, "04_geography_dif.csv")


# ---------------------------------------------------------------------------
# 5. Biology - already scored 0/1, speededness, misfitting items & students
# ---------------------------------------------------------------------------
def ds5():
    rng = np.random.default_rng(505)
    meta = base_frame(rng)
    class_effect = {f"4{c}": e for c, e in zip("ABCDEFGH", [0.8, 0.6, 0.3, 0.1, 0.0, -0.2, -0.4, -0.6])}
    theta = rng.normal(0.0, 1.0, N) + meta["Class"].map(class_effect).to_numpy()
    I = 35
    b = rng.uniform(-2.2, 2.2, I)
    slope = np.ones(I)
    slope[9] = 0.25   # Q10: very weak discrimination (noisy)
    slope[17] = 3.0   # Q18: over-discriminating (overfit)
    X = (rng.random((N, I)) < sig(slope * (theta[:, None] - b))).astype(float)

    # 25 careless / random-guessing students (person misfit)
    careless = rng.choice(N, 25, replace=False)
    X[careless] = (rng.random((25, I)) < 0.5).astype(float)

    # missing at random 1.5%
    X[rng.random((N, I)) < 0.015] = np.nan
    # speededness: weaker students run out of time on the last 6 items
    for k, j in enumerate(range(I - 6, I)):
        p_skip = sig(-(theta - (-0.2))) * (0.05 + 0.05 * k)
        X[rng.random(N) < p_skip, j] = np.nan

    df = pd.concat([meta, pd.DataFrame(X, columns=[f"Q{j+1}" for j in range(I)])], axis=1)
    df = df.astype({f"Q{j+1}": "Int64" for j in range(I)})
    df.to_csv(OUT / "05_biology_scored_01.csv", index=False)
    print("saved 05_biology_scored_01.csv:", df.shape)


def poly_response(rng, theta, steps, slope=1.0):
    """Simulate polytomous scores 0..m under the partial credit model."""
    cum = np.cumsum(slope * (theta[:, None] - np.asarray(steps)[None, :]), axis=1)
    logits = np.concatenate([np.zeros((len(theta), 1)), cum], axis=1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    u = rng.random(len(theta))[:, None]
    return (p.cumsum(1) < u).sum(1)


# ---------------------------------------------------------------------------
# 6. Physics - MCQ + structured questions (mixed marks) -> Partial Credit Model
# ---------------------------------------------------------------------------
def ds6():
    rng = np.random.default_rng(606)
    meta = base_frame(rng)
    theta = rng.normal(0.2, 1.1, N)
    # Section A: 10 MCQ
    n_mcq = 10
    b = rng.uniform(-1.8, 1.8, n_mcq)
    true_key = make_key(rng, n_mcq)
    key = list(true_key)
    key[2] = wrong_options(true_key[2])[0]          # Q3: wrong key in marking scheme
    mcq = simulate_mcq(rng, theta, b, true_key)
    # Section B: 8 structured questions (Q11-Q18)
    maxes = [2, 3, 4, 5, 3, 6, 4, 2]
    centres = [-1.0, -0.3, 0.2, 0.5, 0.9, 0.4, 1.2, 0.0]
    struct = np.zeros((N, 8), dtype=float)
    for q, (mx, c) in enumerate(zip(maxes, centres)):
        tau = np.linspace(-1.2, 1.2, mx) if mx > 1 else np.array([0.0])
        steps = c + tau
        slope = 1.0
        if q == 3:   # Q14 (out of 5): disordered thresholds - mark 2 rarely awarded
            steps = c + np.array([-1.0, 0.9, -0.1, 1.0, 1.6])
        if q == 5:   # Q16 (out of 6): inconsistent marking between markers
            slope = 0.3
        struct[:, q] = poly_response(rng, theta, steps, slope)
    # Q18 (out of 2): markers only ever award 0 or 2 (no partial credit given)
    struct[:, 7] = np.where(struct[:, 7] >= 1, 2, 0)
    resp = np.concatenate([mcq, struct.astype(int).astype(object)], axis=1)
    resp = add_missing(rng, resp, 0.01)
    full_key = key + [str(m) for m in maxes]          # KEY row: letters for MCQ, max marks for structured
    save(meta, resp, full_key, "06_physics_structured_pcm.csv")


# ---------------------------------------------------------------------------
# 7. Attitude survey - Likert 1-5 -> Rating Scale Model
# ---------------------------------------------------------------------------
def ds7():
    rng = np.random.default_rng(707)
    meta = base_frame(rng)
    theta = rng.normal(0.4, 1.2, N) + np.where(meta["Gender"] == "M", 0.2, 0.0)
    I = 15
    deltas = rng.uniform(-1.0, 1.2, I)
    tau = np.array([-1.6, -0.5, 0.4, 1.7])          # shared category thresholds
    X = np.zeros((N, I), dtype=float)
    for j in range(I):
        th = theta.copy()
        slope = 1.0
        if j in (3, 10):     # Q4, Q11: negatively worded statements NOT reverse-coded
            th = -theta
        if j == 7:           # Q8: vague statement, weak link to the trait
            slope = 0.3
        X[:, j] = poly_response(rng, th, deltas[j] + tau, slope) + 1   # 1..5
    X[rng.random((N, I)) < 0.01] = np.nan
    df = pd.concat([meta, pd.DataFrame(X, columns=[f"Q{j+1}" for j in range(I)])], axis=1)
    df = df.astype({f"Q{j+1}": "Int64" for j in range(I)})
    df.to_csv(OUT / "07_maths_attitude_survey_likert.csv", index=False)
    print("saved 07_maths_attitude_survey_likert.csv:", df.shape)


if __name__ == "__main__":
    ds1(); ds2(); ds3(); ds4(); ds5(); ds6(); ds7()
