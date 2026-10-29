"""
Rasch Exam Analyser — Streamlit app
Automatically detects the data type and chooses the model:
Dichotomous Rasch · Partial Credit Model · Rating Scale Model.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import rasch_engine as R

DATA_DIR = Path(__file__).parent / "data"
VERDICT_COLORS = {"Keep": "#1baf7a", "Minor issue": "#eda100", "Review / revise": "#e34948"}
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#888780"
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

st.set_page_config(page_title="Rasch Exam Analyser", page_icon="📏", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_file(name: str, raw: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw), dtype=str)
    else:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


@st.cache_data(show_spinner="Running Rasch analysis…")
def run_analysis(items: pd.DataFrame, ids: tuple, key_row: dict, model_choice: str):
    prep = R.prepare_data(items, key_row)
    model = None if model_choice.startswith("Auto") else model_choice
    return prep, R.fit_rasch(prep, list(ids), model=model)


def reverse_code(items: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = items.copy()
    for c in cols:
        v = pd.to_numeric(out[c].replace("", np.nan), errors="coerce")
        lo, hi = v.min(), v.max()
        out[c] = (lo + hi - v).map(lambda x: "" if pd.isna(x) else str(int(x)))
    return out


def fmt_num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def excel_report(res: R.RaschResult, meta: pd.DataFrame | None, explanation: str) -> bytes:
    buf = io.BytesIO()
    s = res.summary
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({
            "Statistic": ["Model", "Model choice", "Students", "Questions", "Person reliability", "Person separation",
                          "Cronbach's alpha", "Item reliability", "Person mean (logits)", "Item mean (logits)",
                          "Targeting", "Mean raw score", "Maximum total", "Missing %", "Extreme persons",
                          "Extreme items", "Notes"],
            "Value": [s["model"], explanation.replace("**", ""), s["n_persons"], s["n_items"],
                      s["person_reliability"], s["person_separation"], s["alpha"], s["item_reliability"],
                      s["person_mean"], s["item_mean"], s["targeting"], s["raw_mean"], s["max_total"],
                      s["missing_pct"], s["n_extreme_persons"], s["n_extreme_items"], " | ".join(res.notes)],
        }).to_excel(xw, sheet_name="Summary", index=False)
        res.item_table.round(3).to_excel(xw, sheet_name="Items", index=False)
        if (res.max_score > 1).any():
            res.threshold_table.round(3).to_excel(xw, sheet_name="Thresholds", index=False)
        pt = res.person_table.round(3)
        if meta is not None:
            pt = pd.concat([pt, meta.reset_index(drop=True)], axis=1)
        pt.to_excel(xw, sheet_name="Students", index=False)
        if res.distractors:
            rows = []
            for item, d in res.distractors.items():
                dd = d.copy(); dd.insert(0, "Item", item); rows.append(dd)
            pd.concat(rows).round(3).to_excel(xw, sheet_name="Distractors", index=False)
    return buf.getvalue()


def interpret_summary(s: dict) -> list[tuple[str, str]]:
    msgs = []
    r = s["person_reliability"]
    if r >= 0.8:
        msgs.append(("success", f"Person reliability {r:.2f} — results are dependable for grading and ranking."))
    elif r >= 0.7:
        msgs.append(("warning", f"Person reliability {r:.2f} — acceptable for classroom tests, too low for high-stakes decisions (aim ≥ 0.80)."))
    else:
        msgs.append(("error", f"Person reliability {r:.2f} — scores are not dependable. Usual causes: poor targeting, too few items, faulty items."))
    t = s["targeting"]
    if abs(t) <= 0.5:
        msgs.append(("success", f"Targeting {t:+.2f} logits — difficulty matches the cohort well."))
    elif t > 0:
        msgs.append(("warning" if t <= 1.5 else "error", f"Targeting {t:+.2f} logits — the paper is too EASY for these students. Add harder items."))
    else:
        msgs.append(("warning" if t >= -1.5 else "error", f"Targeting {t:+.2f} logits — the paper is too HARD for these students. Add easier items."))
    ext = s["n_extreme_persons"] / max(s["n_persons"], 1)
    if ext > 0.05:
        msgs.append(("warning", f"{s['n_extreme_persons']} students ({ext:.0%}) scored zero or full marks — "
                                "their ability cannot be measured precisely (floor/ceiling effect)."))
    if s["missing_pct"] > 2:
        msgs.append(("info", f"{s['missing_pct']:.1f}% of responses are blank — check the last questions for speededness."))
    return msgs


def wright_map(res: R.RaschResult, show_steps: bool) -> go.Figure:
    th = res.theta[~np.isnan(res.theta)]
    it = res.item_table.dropna(subset=["Measure"]).copy()
    pts = it[["Item", "Measure", "Verdict", "Facility", "Flags"]].copy()
    if show_steps:
        rows = []
        for j, name in enumerate(res.item_names):
            if j not in it.index:
                continue
            if res.max_score[j] > 1:
                for k, sv in enumerate(res.steps[j], 1):
                    rows.append({"Item": f"{name}.{k}", "Measure": sv, "Verdict": it.loc[j, "Verdict"],
                                 "Facility": it.loc[j, "Facility"], "Flags": f"Threshold {k-1}→{k}"})
            else:
                rows.append(it.loc[j, ["Item", "Measure", "Verdict", "Facility", "Flags"]].to_dict())
        pts = pd.DataFrame(rows)
    allv = np.concatenate([th, pts["Measure"].to_numpy()])
    lo, hi = float(np.floor(allv.min() - 0.5)), float(np.ceil(allv.max() + 0.5))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.45, 0.55], horizontal_spacing=0.02,
                        subplot_titles=("Students", "Questions" + (" (thresholds)" if show_steps else "")))
    fig.add_trace(go.Histogram(y=th, ybins=dict(start=lo, end=hi, size=0.25), marker_color=BLUE, name="Students",
                               hovertemplate="Ability %{y}<br>%{x} students<extra></extra>"), row=1, col=1)
    pts = pts.sort_values("Measure")
    pts["x"] = pts.groupby((pts["Measure"] / 0.25).round()).cumcount()
    for v, colr in VERDICT_COLORS.items():
        sub = pts[pts["Verdict"] == v]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["Measure"], mode="markers+text", text=sub["Item"], textposition="middle right",
                marker=dict(color=colr, size=9, symbol="diamond"), name=v,
                customdata=sub[["Facility", "Flags"]].fillna(""),
                hovertemplate="%{text}<br>Measure %{y:.2f}<br>Facility %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>"),
                row=1, col=2)
    fig.add_hline(y=float(np.mean(th)), line_dash="dash", line_color=BLUE, row=1, col=1,
                  annotation_text="Student mean", annotation_position="top left")
    fig.add_hline(y=float(it["Measure"].mean()), line_dash="dash", line_color=GRAY, row=1, col=2,
                  annotation_text="Question mean", annotation_position="top right")
    fig.update_xaxes(autorange="reversed", title_text="Number of students", row=1, col=1)
    fig.update_xaxes(visible=False, range=[-0.5, pts["x"].max() + 2.5], row=1, col=2)
    fig.update_yaxes(title_text="Logits (higher = more able / harder)", range=[lo, hi], row=1, col=1)
    fig.update_layout(height=max(500, int((hi - lo) * 70)), legend=dict(orientation="h", y=-0.08),
                      margin=dict(l=10, r=10, t=40, b=10), bargap=0.05)
    return fig


def targeting_gaps(res: R.RaschResult) -> list[str]:
    th = res.theta[~np.isnan(res.theta)]
    b = np.concatenate([np.atleast_1d(s) for s, d in zip(res.steps, res.delta) if not np.isnan(d)] + [res.delta[~np.isnan(res.delta)]])
    out = []
    edges = np.arange(np.floor(th.min()), np.ceil(th.max()) + 0.5, 0.5)
    for a, z in zip(edges[:-1], edges[1:]):
        n = ((th >= a) & (th < z)).sum()
        if n / len(th) >= 0.05 and ((b >= a - 0.25) & (b < z + 0.25)).sum() == 0:
            out.append(f"{a:+.1f} to {z:+.1f} logits: {n} students ({n/len(th):.0%}) but no questions or thresholds")
    return out


# ---------------------------------------------------------------------------
# Sidebar — data
# ---------------------------------------------------------------------------
st.sidebar.title("📏 Rasch Exam Analyser")
source = st.sidebar.radio("Data source", ["Practice dataset", "Upload my file"])
df_raw, fname = None, None
if source == "Practice dataset":
    files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    if files:
        fname = st.sidebar.selectbox("Choose a dataset", files)
        df_raw = read_file(fname, (DATA_DIR / fname).read_bytes())
    else:
        st.sidebar.warning("No practice datasets in /data. Run `python generate_datasets.py`.")
else:
    up = st.sidebar.file_uploader("CSV or Excel (one row per student, one column per question)", type=["csv", "xlsx", "xls"])
    if up is not None:
        fname = up.name
        df_raw = read_file(up.name, up.getvalue())

template = pd.DataFrame({"StudentID": ["KEY", "S001", "S002"], "Class": ["", "4A", "4B"], "Gender": ["", "F", "M"],
                         "Q1": ["B", "B", "C"], "Q2": ["D", "A", "D"], "Q3": ["4", "3", "1"], "Q4": ["2", "2", ""]})
st.sidebar.download_button("Download blank template (CSV)", template.to_csv(index=False), "rasch_template.csv", "text/csv")

if df_raw is None:
    st.title("Rasch Exam Analyser")
    st.markdown(
        "Upload an answer sheet or pick a practice dataset from the sidebar.\n\n"
        "**Layout:** one row per student, one column per question. The app works out the data type of every "
        "question and **chooses the Rasch model automatically**:\n\n"
        "| Your data | Model used |\n|---|---|\n"
        "| All right/wrong (MCQ letters or 0/1) | Dichotomous Rasch |\n"
        "| Different maximum marks (e.g. MCQ + structured questions out of 2–6) | Partial Credit Model |\n"
        "| All questions on the same scale (e.g. Likert 1–5 survey) | Rating Scale Model |\n\n"
        "- **MCQ letters**: add a row whose ID is `KEY` with the correct letters (or type them in the sidebar).\n"
        "- **Marks**: whole numbers (0, 1, 2, …). In the `KEY` row you may put each question's maximum mark.\n"
        "- Extra columns (Class, Gender, Stream…) are used for group comparisons and bias (DIF) checks.\n\n"
        "Remove student names before uploading — use index numbers instead."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — column setup & model
# ---------------------------------------------------------------------------
with st.sidebar.expander("Column setup", expanded=(source == "Upload my file")):
    cols = list(df_raw.columns)
    id_col = st.selectbox("Student ID column", cols, index=0)
    guess = [c for c in cols if c != id_col and R.column_type(df_raw[c]) is not None]
    item_cols = st.multiselect("Question columns", [c for c in cols if c != id_col], default=guess)
    group_cols = [c for c in cols if c not in item_cols and c != id_col]
    st.caption(f"Group columns: {', '.join(group_cols) if group_cols else 'none'}")

if len(item_cols) < 3:
    st.error("Select at least 3 question columns in **Column setup**.")
    st.stop()

is_key = df_raw[id_col].astype(str).str.strip().str.upper() == "KEY"
df = df_raw[~is_key].reset_index(drop=True)
key_row = {c: str(v) for c, v in df_raw[is_key].iloc[0][item_cols].items()} if is_key.any() else {}
types = {c: R.column_type(df[c]) for c in item_cols}
bad_cols = [c for c, t in types.items() if t is None]
if bad_cols:
    st.error(f"These columns are not letters or whole-number scores: {', '.join(bad_cols)}. Remove them from Question columns.")
    st.stop()

mcq_cols = [c for c in item_cols if types[c] == "mcq"]
num_cols = [c for c in item_cols if types[c] == "numeric"]
missing_key = [c for c in mcq_cols if str(key_row.get(c, "")).strip().upper() not in R.LETTERS]
if missing_key:
    txt = st.sidebar.text_input(f"Answer key for {len(missing_key)} MCQ question(s) ({missing_key[0]}…{missing_key[-1]}), e.g. ABDC…")
    letters = [p.strip().upper() for p in (txt.split(",") if "," in txt else list(txt.replace(" ", "")))]
    if len(letters) != len(missing_key) or not all(l in R.LETTERS for l in letters):
        st.info(f"{len(mcq_cols)} question(s) contain MCQ letters. Add a `KEY` row to your file or type the "
                f"{len(missing_key)} correct letters in the sidebar.")
        st.stop()
    key_row.update(dict(zip(missing_key, letters)))

items_df = df[item_cols]
with st.sidebar.expander("Options"):
    rev_cols = st.multiselect("Reverse-code questions (negatively worded survey items)", num_cols, default=[],
                              help="Flips the scale, e.g. 1↔5, 2↔4 on a 1–5 Likert item.") if num_cols else []
    try:
        prep_preview = R.prepare_data(reverse_code(items_df, rev_cols) if rev_cols else items_df, key_row)
    except ValueError as e:
        st.error(str(e)); st.stop()
    model_options = [f"Auto (recommended: {prep_preview.auto_model})"] + [m for m in prep_preview.allowed_models]
    model_choice = st.selectbox("Rasch model", model_options, index=0)

if rev_cols:
    items_df = reverse_code(items_df, rev_cols)
try:
    prep, res = run_analysis(items_df, tuple(df[id_col].tolist()), key_row, model_choice)
except ValueError as e:
    st.error(str(e)); st.stop()

explanation = R.explain_model_choice(prep)
meta = df[group_cols] if group_cols else None
s = res.summary
poly = bool((res.max_score > 1).any())
st.sidebar.success(f"Model: {res.model}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Rasch Exam Analyser")
st.caption(f"{fname} · {s['n_persons']} students · {s['n_items']} questions · {res.model} · "
           f"JML {'converged' if res.converged else 'did NOT converge'} in {res.iterations} iterations")
st.info("🔍 **Data type detected** — " + explanation + (f"\n\nReverse-coded: {', '.join(rev_cols)}" if rev_cols else ""))
for n in res.notes:
    st.caption("ℹ️ " + n)

tabs = st.tabs(["Overview", "Wright map", "Questions", "Question detail", "Students", "DIF (bias)", "Guide"])

# ---- Overview -------------------------------------------------------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Person reliability", fmt_num(s["person_reliability"]), help="Rasch equivalent of Cronbach's alpha. Aim ≥ 0.80.")
    c[1].metric("Person separation", fmt_num(s["person_separation"]), help="Distinct ability levels ≈ (4×separation+1)/3. Aim ≥ 2.")
    c[2].metric("Cronbach's alpha", fmt_num(s["alpha"]), help="Equals KR-20 for right/wrong data.")
    c[3].metric("Targeting", f"{s['targeting']:+.2f}", help="Student mean − question mean (logits). Near 0 is ideal.")
    c = st.columns(4)
    c[0].metric("Mean raw score", f"{s['raw_mean']:.1f} / {s['max_total']}")
    c[1].metric("Item reliability", fmt_num(s["item_reliability"]))
    c[2].metric("Zero / full-mark students", s["n_extreme_persons"])
    c[3].metric("Missing responses", f"{s['missing_pct']:.1f}%")

    st.subheader("What this means")
    for kind, m in interpret_summary(s):
        getattr(st, kind)(m)
    gaps = targeting_gaps(res)
    if gaps:
        st.warning("**Targeting gaps** — ability ranges with many students but nothing measuring them:\n\n- " + "\n- ".join(gaps))
    neg = res.item_table[res.item_table["Pt-Measure"] < 0]
    if res.model == R.MODEL_RSM and len(neg):
        st.warning(f"{', '.join(neg['Item'])} correlate negatively with the total. If these are negatively worded "
                   "statements, reverse-code them under **Options** in the sidebar and look again.")

    st.subheader("Question verdicts")
    vc = res.item_table["Verdict"].value_counts()
    c = st.columns(3)
    for col, v in zip(c, ["Keep", "Minor issue", "Review / revise"]):
        col.metric(v, int(vc.get(v, 0)))
    bad = res.item_table[res.item_table["Verdict"] == "Review / revise"][["Item", "Max", "Facility", "Measure", "Outfit MNSQ", "Pt-Measure", "Flags"]]
    if len(bad):
        st.dataframe(bad.round(2), hide_index=True, width="stretch")
    else:
        st.success("No questions need revision.")
    st.download_button("⬇️ Download full Excel report", excel_report(res, meta, explanation),
                       f"rasch_report_{Path(fname).stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---- Wright map -------------------------------------------------------------
with tabs[1]:
    show_steps = st.toggle("Show step thresholds of multi-mark questions", value=False, disabled=not poly)
    st.plotly_chart(wright_map(res, show_steps and poly), width="stretch")
    st.caption("Students (left) and questions (right) share one ruler. A student level with a right/wrong question "
               "has a 50% chance of getting it right. For multi-mark questions the point is the average of its "
               "thresholds — switch on thresholds to see where each extra mark becomes likely.")

# ---- Questions --------------------------------------------------------------
with tabs[2]:
    only_flag = st.toggle("Show only flagged questions", value=False)
    sort_by = st.selectbox("Sort by", ["Question order", "Measure (hardest first)", "Outfit MNSQ (worst first)", "Pt-Measure (lowest first)"])
    t = res.item_table.copy()
    t["_order"] = range(len(t))
    if only_flag:
        t = t[t["Verdict"] != "Keep"]
    t = {"Question order": t.sort_values("_order"),
         "Measure (hardest first)": t.sort_values("Measure", ascending=False),
         "Outfit MNSQ (worst first)": t.sort_values("Outfit MNSQ", ascending=False),
         "Pt-Measure (lowest first)": t.sort_values("Pt-Measure")}[sort_by]
    drop = ["_order", "Extreme"] + ([] if res.distractors else ["Key"]) + ([] if poly else ["Max", "Mean score", "Disordered"])
    show = t.drop(columns=drop)
    st.dataframe(show.style.map(lambda v: f"color: {VERDICT_COLORS.get(v, 'inherit')}; font-weight: 600", subset=["Verdict"])
                 .format(precision=2), hide_index=True, width="stretch", height=min(700, 38 + 35 * len(show)))
    st.caption("Facility = mean score ÷ maximum mark (= % correct for right/wrong questions). "
               "Infit/Outfit MNSQ 0.7–1.3 good · Pt-Measure should be positive (≥ 0.20 ideal) · "
               "ZSTD beyond ±2 = statistically significant misfit (very sensitive with large samples).")
    if poly:
        st.markdown("**Step thresholds** (Andrich) — should increase from one step to the next")
        st.dataframe(res.threshold_table.round(2), hide_index=True, width="stretch", height=300)

# ---- Question detail ----------------------------------------------------------
with tabs[3]:
    item = st.selectbox("Choose a question", res.item_names)
    j = res.item_names.index(item)
    row = res.item_table.iloc[j]
    mx = int(res.max_score[j])
    c = st.columns(5)
    c[0].metric("Facility" if mx > 1 else "% correct",
                f"{row['Mean score']:.2f} / {mx}" if mx > 1 else f"{100 * row['Facility']:.0f}%")
    c[1].metric("Difficulty", f"{row['Measure']:+.2f}")
    c[2].metric("Infit", fmt_num(row["Infit MNSQ"]))
    c[3].metric("Outfit", fmt_num(row["Outfit MNSQ"]))
    c[4].metric("Pt-Measure", fmt_num(row["Pt-Measure"]))
    if row["Flags"]:
        (st.error if row["Verdict"] == "Review / revise" else st.warning)(f"**{row['Verdict']}:** {row['Flags']}")
    else:
        st.success("This question behaves as the Rasch model expects.")

    lo_t, hi_t = float(np.nanmin(res.theta)) - 0.5, float(np.nanmax(res.theta)) + 0.5
    tt, E, P = R.expected_curve(res, j, lo=lo_t, hi=hi_t)
    left, right = st.columns(2)
    with left:
        st.markdown("**Expected vs observed score**")
        emp = R.empirical_icc(res, j)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt, y=E, mode="lines", name="Expected (Rasch)", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=emp["Ability"], y=emp["Observed"], mode="lines+markers", name="Observed",
                                 line=dict(color=ORANGE, dash="dash", width=2), marker=dict(size=8, symbol="square"),
                                 customdata=emp["N"], hovertemplate="Ability %{x:.2f}<br>Mean %{y:.2f}<br>n=%{customdata}<extra></extra>"))
        fig.update_layout(height=360, yaxis=dict(range=[0, mx], title="Mean score" if mx > 1 else "Proportion correct"),
                          xaxis_title="Student ability (logits)", legend=dict(orientation="h", y=-0.25),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Observed points should follow the blue curve. Flatter = weak discrimination (noise, guessing, "
                   "inconsistent marking). Steeper = over-predictable. Downward = faulty question or wrong key.")
    with right:
        if mx > 1:
            st.markdown("**Category probability curves**")
            fig = go.Figure()
            for k in range(mx + 1):
                fig.add_trace(go.Scatter(x=tt, y=P[:, k], mode="lines", name=f"Score {k}",
                                         line=dict(color=CAT_COLORS[k % len(CAT_COLORS)], width=2,
                                                   dash=["solid", "dash", "dot", "dashdot"][k % 4])))
            fig.update_layout(height=360, yaxis=dict(range=[0, 1], title="Probability"),
                              xaxis_title="Student ability (logits)", legend=dict(orientation="h", y=-0.25),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Each score should have its own peak (be the most likely score for some ability range). "
                       "A score whose curve never peaks means disordered thresholds — markers rarely award it, "
                       "or the marking scheme does not separate it from its neighbours.")
            st.dataframe(res.threshold_table[res.threshold_table["Item"] == item].round(2), hide_index=True, width="stretch")
        elif item in res.distractors:
            st.markdown("**Distractor analysis**")
            d = res.distractors[item]
            st.dataframe(d.style.format({"%": "{:.1f}", "Mean ability": "{:+.2f}", "Pt-biserial": "{:+.2f}"}),
                         hide_index=True, width="stretch")
            dd = d[d["Option"] != "(blank)"]
            fig = go.Figure(go.Bar(x=dd["Option"], y=dd["Mean ability"],
                                   marker_color=[BLUE if k == "✓" else GRAY for k in dd["Key"]],
                                   text=[f"{p:.0f}%" for p in dd["%"]], textposition="outside"))
            fig.update_layout(height=260, yaxis_title="Mean ability of choosers", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("The key (blue) should attract the most able students and have a positive pt-biserial. "
                       "Distractors should attract weaker students (negative pt-biserial). Options chosen by < 3% are not working.")
        else:
            st.info("Distractor analysis needs MCQ letter responses plus an answer key.")

# ---- Students -----------------------------------------------------------------
with tabs[4]:
    pt = res.person_table.copy()
    if meta is not None:
        pt = pd.concat([pt, meta.reset_index(drop=True)], axis=1)
    c = st.columns(3)
    c[0].metric("Misfitting students (outfit > 2)", int((pt["Outfit MNSQ"] > 2).sum()),
                help="Unexpected response patterns: careless errors, lucky guessing, copying, or skipping.")
    c[1].metric("Full marks", int((pt["Extreme"] & (pt["Raw score"] == pt["Max possible"])).sum()))
    c[2].metric("Zero score", int((pt["Extreme"] & (pt["Raw score"] == 0)).sum()))
    if group_cols:
        g = st.selectbox("Compare groups by", group_cols)
        comp = pt.groupby(g).agg(Students=("ID", "count"), Mean_measure=("Measure", "mean"),
                                 SD=("Measure", "std"), Mean_raw=("Raw score", "mean")).round(2).reset_index()
        comp = comp.sort_values("Mean_measure", ascending=False)
        fig = go.Figure(go.Bar(x=comp[g].astype(str), y=comp["Mean_measure"], marker_color=BLUE,
                               error_y=dict(type="data", array=comp["SD"] / np.sqrt(comp["Students"]))))
        fig.update_layout(height=300, yaxis_title="Mean ability (logits)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comp, hide_index=True, width="stretch")
    only_mis = st.toggle("Show only misfitting students", value=False)
    view = pt[pt["Outfit MNSQ"] > 2] if only_mis else pt
    st.dataframe(view.round(2).sort_values("Measure", ascending=False), hide_index=True, width="stretch", height=450)
    miss = pd.Series(100 * np.isnan(res.X).mean(0), index=res.item_names)
    if miss.max() > 3:
        st.markdown("**Blank responses by question** (rising towards the end suggests students ran out of time)")
        fig = go.Figure(go.Bar(x=miss.index, y=miss.values, marker_color=GRAY))
        fig.update_layout(height=250, yaxis_title="% blank", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

# ---- DIF ---------------------------------------------------------------------
with tabs[5]:
    if not group_cols:
        st.info("Add a group column (e.g. Gender, Class, Stream) to your file to check for question bias.")
    else:
        g = st.selectbox("Group variable", group_cols, key="dif_g")
        levels = sorted(v for v in df[g].astype(str).unique() if v.strip())
        if len(levels) < 2:
            st.warning("This column has fewer than two groups.")
        else:
            c = st.columns(2)
            focal = c[0].selectbox("Focal group", levels, index=0)
            reference = c[1].selectbox("Reference group", [v for v in levels if v != focal], index=0)
            dif = R.dif_analysis(res, df[g], focal, reference)
            n_f, n_r = (df[g].astype(str) == focal).sum(), (df[g].astype(str) == reference).sum()
            st.caption(f"{focal}: {n_f} students · {reference}: {n_r} students. DIF compares students of the SAME "
                       "ability — a real difference in group ability (impact) is not bias.")
            fig = go.Figure(go.Bar(
                x=dif["Item"], y=dif["Contrast"],
                marker_color=["#e34948" if str(v).startswith("C") else "#eda100" if str(v).startswith("B") else "#b4b2a9"
                              for v in dif["DIF class"]],
                hovertemplate="%{x}<br>Contrast %{y:.2f}<extra></extra>"))
            for yv in (0.43, -0.43, 0.64, -0.64):
                fig.add_hline(y=yv, line_dash="dot", line_color=GRAY)
            fig.update_layout(height=320, yaxis_title=f"Contrast (+ = harder for {focal})", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            flagged = dif[dif["DIF class"].str[:1].isin(["B", "C"])]
            if len(flagged):
                st.warning(f"{len(flagged)} question(s) show moderate (B) or large (C) DIF. Review their content, "
                           "context and wording before keeping, revising or dropping them.")
            else:
                st.success("No meaningful DIF between these groups.")
            st.dataframe(dif.round(2), hide_index=True, width="stretch")
            st.caption("Classes: A negligible (< 0.43 logits) · B moderate (0.43–0.64, |t| ≥ 2) · C large (≥ 0.64, |t| ≥ 2).")

# ---- Guide -------------------------------------------------------------------
with tabs[6]:
    st.markdown("""
### How the model is chosen
| Detected data | Model | Why |
|---|---|---|
| Every question right/wrong (MCQ letters scored with the key, or 0/1) | **Dichotomous Rasch** | One difficulty per question |
| Questions with different maximum marks (e.g. MCQ + structured questions) | **Partial Credit Model** (Masters, 1982) | Each question gets its own step thresholds |
| Every question on the same multi-category scale (e.g. Likert 1–5) | **Rating Scale Model** (Andrich, 1978) | All questions share one threshold structure |

You can override *Auto* under **Options** (e.g. use PCM for a structured paper where every question is out of 4 marks
but the marking schemes differ). Scores that are never awarded are collapsed automatically, and scales starting
at 1 are recoded to start at 0.

### Reading the results
| Statistic | Good | Concern |
|---|---|---|
| Person reliability | ≥ 0.80 | < 0.70 |
| Person separation | ≥ 2.0 | < 1.5 |
| Targeting (student mean − question mean) | −0.5 to +0.5 | beyond ±1 |
| Infit / Outfit MNSQ | 0.7 – 1.3 | > 1.3 noisy (underfit) · < 0.7 too predictable (overfit) |
| Pt-Measure correlation | ≥ 0.20 | < 0.15 or negative |
| Step thresholds (multi-mark) | increase step by step | disordered = a score is never the most likely |
| Distractor (MCQ) | chosen by ≥ 5%, by weaker students | < 3% (dead) or chosen by abler students than the key |
| DIF contrast | < 0.43 logits | ≥ 0.64 with \\|t\\| ≥ 2 |

### Typical causes of a flagged question
- **Negative pt-measure / possible wrong key** → check the marking scheme first (surveys: reverse-code negatively worded items).
- **Underfit** → ambiguous wording, two defensible answers, guessing, inconsistent marking, or a different skill.
- **Overfit** → may give away the answer or duplicate another question. Usually harmless.
- **Disordered thresholds** → a mark is rarely awarded or not clearly defined in the marking scheme; consider merging it with a neighbouring mark.
- **Non-functioning distractor** → rewrite it using a common student misconception.

### Method
Joint Maximum Likelihood estimation (as in Winsteps) with (L−1)/L bias correction; Newton–Raphson with full
threshold Hessian for the Partial Credit Model; extreme scores estimated with a 0.3 score-point adjustment;
infit/outfit with Wilson–Hilferty ZSTD. DIF: separate calibration per group with person measures and thresholds
anchored. Results may differ slightly from Winsteps, RUMM2030 or ConQuest.
""")
