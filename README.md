# 📏 Rasch Exam Analyser

A free, browser-based Rasch analysis tool for school examinations and surveys, built with Streamlit.
Upload an answer sheet and the app **detects the data type of every question and chooses the Rasch model automatically**:

| Detected data | Model used |
|---|---|
| All right/wrong — MCQ letters (scored with the key) or 0/1 | **Dichotomous Rasch** |
| Different maximum marks — e.g. MCQ + structured questions out of 2–6 | **Partial Credit Model** (Masters) |
| Same multi-category scale on every question — e.g. Likert 1–5 survey | **Rating Scale Model** (Andrich) |

The automatic choice can be overridden under **Options** in the sidebar. Unused score categories are collapsed
and scales starting at 1 are recoded to 0 automatically; negatively worded survey items can be reverse-coded.

You get:

- **Overview** — person reliability, separation, KR-20, targeting, plain-English interpretation
- **Wright map** — students and questions on one logit ruler, with targeting gaps highlighted
- **Item table** — difficulty, SE, infit/outfit MNSQ & ZSTD, point-measure correlation, automatic flags and a *Keep / Minor issue / Review* verdict
- **Question detail** — expected vs observed score curve; distractor analysis (MCQ); category probability curves and step thresholds (multi-mark)
- **Students** — ability measures, misfitting students, group comparisons (class, gender…), blank-response pattern
- **DIF (bias)** — differential item functioning between any two groups (ETS A/B/C classes)
- **Excel report** download

Seven practice datasets of 1,000 students each (five right/wrong, one mixed MCQ + structured paper, one Likert survey) are included in `data/` (see `data/PRACTICE_ANSWERS.md` *after* you try them).

---

## Project structure

```
rasch-app/
├── app.py                  # Streamlit interface
├── rasch_engine.py         # Rasch estimation & statistics (numpy/pandas only)
├── generate_datasets.py    # Re-creates the practice datasets
├── requirements.txt
├── .streamlit/config.toml  # Theme & upload size
└── data/
    ├── 01_science_good_paper.csv
    ├── 02_maths_too_easy.csv
    ├── 03_english_faulty_items.csv
    ├── 04_geography_dif.csv
    ├── 05_biology_scored_01.csv
    ├── 06_physics_structured_pcm.csv
    ├── 07_maths_attitude_survey_likert.csv
    └── PRACTICE_ANSWERS.md
```

## Data format

One row per student, one column per question. Any other columns (Class, Gender, Stream…) become group variables.

**MCQ letters** — first data row has ID `KEY` with the correct answers (or type the key in the sidebar):

| StudentID | Class | Gender | Q1 | Q2 | Q3 |
|---|---|---|---|---|---|
| KEY | | | B | D | A |
| S001 | 4A | F | B | A | A |
| S002 | 4B | M | C | D | |

**Already scored** — `1` correct, `0` wrong, blank = not answered.

**Marks / ratings** — whole numbers (0, 1, 2, 3 … or Likert 1–5). You can mix MCQ letter columns and mark columns
in one file; in the `KEY` row put the correct letter for MCQ questions and (optionally) the maximum mark for
structured questions:

| StudentID | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| KEY | B | D | 4 | 2 |
| S001 | B | A | 3 | 2 |
| S002 | C | D | 1 | |

CSV or Excel (.xlsx). ⚠️ Replace student names with index numbers before uploading.

---

## Run it on your computer

```bash
# 1. Install Python 3.10+ from python.org, then in a terminal:
cd rasch-app
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the app (opens in your browser at http://localhost:8501)
streamlit run app.py
```

## Deploy free on GitHub + Streamlit Community Cloud

### Step 1 — Put the code on GitHub
**Option A: website only (no Git needed)**
1. Sign in at <https://github.com> → **New repository** → name it e.g. `rasch-exam-analyser` → *Public* → **Create**.
2. Click **uploading an existing file**, drag in **all files and folders** from this project
   (including the `data` and `.streamlit` folders) → **Commit changes**.
   *Tip:* `.streamlit` is a hidden folder on Mac — press `Cmd + Shift + .` in Finder to see it.
   The app still works without it (only the theme and 50 MB upload limit are lost).

**Option B: command line**
```bash
cd rasch-app
git init
git add .
git commit -m "Rasch Exam Analyser"
git branch -M main
git remote add origin https://github.com/<your-username>/rasch-exam-analyser.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit
1. Go to <https://share.streamlit.io> and **sign in with GitHub**.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/rasch-exam-analyser` · Branch: `main` · Main file path: `app.py`.
4. (Optional) choose a custom URL, e.g. `smp-rasch.streamlit.app`.
5. Click **Deploy**. The first build takes 2–5 minutes.

Any later push to GitHub redeploys the app automatically.

### Privacy note
A public Streamlit app can be opened by anyone with the link. Uploaded files are processed in memory and
not saved, but for real student data:
- remove names / IC numbers before uploading, **or**
- make the GitHub repo private and, in Streamlit Cloud → app **Settings → Sharing**, restrict viewers to
  invited email addresses, **or**
- run the app locally (`streamlit run app.py`) so data never leaves your computer.

---

## Method

- Dichotomous Rasch, Partial Credit and Rating Scale models, estimated by **Joint Maximum Likelihood (JML)**
  as in Winsteps, with the (L−1)/L bias correction and Newton–Raphson using the full threshold Hessian;
  missing responses handled; zero/perfect scores estimated with a 0.3 score-point adjustment.
- Andrich step thresholds with a disordering check; category probability curves.
- Fit: infit & outfit mean-squares with Wilson–Hilferty ZSTD; point-measure correlations.
- Reliability: Rasch person/item reliability and separation; KR-20 for comparison.
- DIF: separate item calibration per group with person measures anchored; contrast t-test;
  ETS-style classes (A < 0.43, B 0.43–0.64, C ≥ 0.64 logits).

Results can differ slightly from Winsteps, RUMM2030 or ConQuest (different estimators and defaults).

Re-create the practice data any time with `python generate_datasets.py`.
