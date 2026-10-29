# Practice answers — what is hidden in each dataset

> Try analysing each dataset in the app **before** reading this page.
> Every dataset has 1,000 students. Files with MCQ letters have a `KEY` row as the first data row.

---

## 01_science_good_paper.csv — the benchmark (40 MCQ)
A well-built paper. Use it to learn what "healthy" output looks like.

- Person reliability ≈ 0.89, KR-20 ≈ 0.90 → dependable results.
- Targeting ≈ −0.3 logits → paper difficulty matches the cohort.
- All items fit (infit/outfit close to 1.0), all pt-measures positive.
- Every distractor is chosen by weaker students; no wrong keys.
- Several items are only 15–20 % correct — they are **hard but healthy** (good fit, good discrimination). Hard ≠ bad.

---

## 02_maths_too_easy.csv — poor targeting (40 MCQ)
- Students average about **+3 logits above** the items → paper far too easy.
- ~140 students score **full marks** (ceiling effect) → their ability can't be measured.
- Person reliability only ≈ 0.53 despite 40 items, because almost everyone scores 35–40.
- **Q1**: every student correct → gives zero information.
- Many items above 95 % correct with weak discrimination (Q2, Q13, Q16, Q24, Q27…).
- **Non-functioning distractors** on every third item (Q7, Q10, Q19, Q28, Q40…): one or two options chosen by < 3 %.
- Wright map: a big empty region at the top where most students sit.

**Fix:** add harder items (+1 to +4 logits), rewrite weak distractors, drop Q1.

---

## 03_english_faulty_items.csv — faulty items (40 MCQ, 2 % blanks)
| Item | Planted fault | What the app shows |
|---|---|---|
| **Q7** | Wrong key in marking scheme | Negative pt-measure, outfit ≈ 2.1, "Possible wrong key (B)" |
| **Q15** | Wrong key in marking scheme | Negative pt-measure, able students choose B |
| **Q12** | Two defensible answers | Distractor B has *positive* pt-biserial → "also correct?" |
| **Q23** | Trap item (able students over-think) | Negative pt-measure, able students pick D |
| **Q31** | Nonsense / pure guessing | Each option ≈ 25 %, pt-measure ≈ 0, underfit |
| Q4, Q19, Q27 | Dead distractor (option D) | "Non-functioning distractor D" |

Lesson: Q7, Q15 and Q23 look like "hard questions" from % correct alone. Rasch fit + distractor analysis shows they are **faulty**. Re-key Q7/Q15 and re-score before releasing results.

---

## 04_geography_dif.csv — bias (DIF) (30 MCQ, extra `Stream` column)
Run **DIF → Gender (F vs M)** and **DIF → Stream (Arts vs Science)**.

| Item | Planted DIF | Expected result |
|---|---|---|
| **Q5** | Harder for girls (+1.0 logit) | Class C (large) |
| **Q14** | Harder for boys (+0.9 logit) | Class C (large) |
| **Q22** | Harder for girls (+0.7 logit) | Class B/C |
| **Q9** | Harder for Arts stream (+1.1 logit) | Class C (large) |

Also: Science-stream students are genuinely **more able on average** (+0.5 logit). That is *impact*, not bias — DIF compares students **of the same ability**, so it is not flagged on every item.
Note: one or two extra items may show small "B" DIF by chance — always check the item content before acting.

---

## 05_biology_scored_01.csv — already scored 0/1 (35 items, ~3 % blanks)
No answer key needed (no distractor analysis possible).

- **Q10**: very weak discrimination → underfit (infit ≈ 1.27, outfit ≈ 1.45). Item measures something other than biology ability (ambiguous / poorly worded).
- **Q18**: over-discriminating → overfit (infit ≈ 0.67). Not harmful, but may duplicate another item or give away the answer (check Q17/Q19).
- **25 careless / random-guessing students** → person outfit > 2 in the Students tab (20 or so are flagged).
- **Speededness**: Q30–Q35 have more blanks, mostly from weaker students who ran out of time. Check the "Answered" column and missing % per item.
- Real class differences: 4A strongest → 4H weakest (compare class means in the Students tab).

---

## 06_physics_structured_pcm.csv — MCQ + structured questions → **Partial Credit Model**
Section A Q1–Q10 are MCQ letters; Section B Q11–Q18 are structured questions with maximum marks
2, 3, 4, 5, 3, 6, 4, 2 (given in the `KEY` row). Different maximum marks → the app picks **PCM** automatically.

| Item | Planted problem | What the app shows |
|---|---|---|
| **Q3** | Wrong key in marking scheme | Negative pt-measure, "Possible wrong key (C)" |
| **Q14** (out of 5) | Mark 2 is rarely awarded / poorly defined in the marking scheme | **Disordered thresholds**: step 2→3 is *lower* than step 1→2; the curve for score 2 never peaks |
| **Q16** (out of 6) | Inconsistent marking between markers | Underfit (infit/outfit ≈ 1.4), flat observed curve |
| **Q18** (out of 2) | Markers only gave 0 or 2, never 1 | Note: "score 1 never awarded — categories renumbered"; Q18 becomes right/wrong |

Tip: open **Question detail → Q14** and compare its category curves with a healthy item such as Q12.

---

## 07_maths_attitude_survey_likert.csv — Likert 1–5 survey → **Rating Scale Model**
15 attitude statements, all on the same 1–5 scale → the app picks **RSM** automatically and recodes 1–5 to 0–4.

| Item | Planted problem | What the app shows |
|---|---|---|
| **Q4, Q11** | Negatively worded statements NOT reverse-coded | Strong negative pt-measure (≈ −0.6), huge underfit, and every other item looks "overfit" |
| **Q8** | Vague statement, weak link to attitude | Hidden at first — appears as underfit only after Q4/Q11 are fixed |

Exercise: in the sidebar open **Options → Reverse-code questions** and select Q4 and Q11.
Person reliability rises from ≈ 0.80 to ≈ 0.92, the false "overfit" flags disappear, and Q8 is revealed
as the real weak item. Lesson: fix coding errors first, then judge item fit.
Also try **Options → Rasch model → Partial Credit Model** and compare: with the same scale for all items, RSM
and PCM give very similar results, and RSM is simpler.
