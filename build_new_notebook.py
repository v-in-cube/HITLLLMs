"""
Build four analysis notebooks from human_vs_llm_new_llms.ipynb.

Source notebook already uses the new LLM identifiers and new CSV.
The only code patch needed is adding llm_repeat=0 to the signature of
analyze_llm_internal_agreement (the body already uses the variable; the
caller run_complete_agreement_analysis already passes it).

Cell layout in human_vs_llm_new_llms.ipynb (0-indexed):
  0  imports
  1  LLM_IDENTIFIERS + category maps (flat)
  2  LLM_IDENTIFIERS + CategoryMappings / ConsensusAnalyzer / GroupConsensusAnalyzer / HumanLLMComparator
  3  analysis functions: generate_confusion_matrices, calculate_agreement_statistics,
     analyze_disagreements, compare_category_distributions,
     analyze_human_vs_llm_agreement, run_human_vs_llm_analysis
  4  markdown: primary analysis header
  5  per-repeat MCC code (imports + data load + ground truth + MCC computation + bootstrap CI)
  6  pairwise permutation test + bar chart
  7  inter-LLM kappa + confidence calibration
  8  markdown: ---
  9  run_human_vs_llm_analysis reactions call
  10 run_human_vs_llm_analysis routes call
  11 save_per_llm_outputs + run_human_vs_llm_analysis_per_model definitions
  12 run_human_vs_llm_analysis_per_model reactions call
  13 run_human_vs_llm_analysis_per_model routes call
  14 save_category_distribution_per_llm + run_category_distribution_analysis definitions
  15 run_category_distribution_analysis reactions call
  16 run_category_distribution_analysis routes call
  17 agreement functions (analyze_llm_internal_agreement, etc.)
  18 run_complete_agreement_analysis reactions call
  19 run_complete_agreement_analysis routes call
  20 empty
"""
import json, copy, uuid

SRC = "human_vs_llm_new_llms.ipynb"

NEW_ORDER_TUPLE = '("Claude Opus 4.8", "Gemini 3.1 Pro", "GPT-5.5", "Llama 3.1 70B")'


def patch_source(src):
    """Apply the one remaining code patch: add llm_repeat=0 to
    analyze_llm_internal_agreement's parameter list."""
    text = "".join(src) if isinstance(src, list) else src

    # The signature is missing llm_repeat=0; body + caller already have it.
    text = text.replace(
        "def analyze_llm_internal_agreement(feedback_df, hash_column='reaction_hash',\n"
        "                                 category_column='local_feedback',\n"
        "                                 expert_column='source_file',\n"
        "                                 text_column='local_feedback_text',\n"
        "                                 confidence_column='confidence',\n"
        "                                 min_runs=2,\n"
        "                                 llm_order=" + NEW_ORDER_TUPLE + "):",
        "def analyze_llm_internal_agreement(feedback_df, hash_column='reaction_hash',\n"
        "                                 category_column='local_feedback',\n"
        "                                 expert_column='source_file',\n"
        "                                 text_column='local_feedback_text',\n"
        "                                 confidence_column='confidence',\n"
        "                                 min_runs=2,\n"
        "                                 llm_order=" + NEW_ORDER_TUPLE + ",\n"
        "                                 llm_repeat=0):",
    )

    return text.splitlines(keepends=True) if isinstance(src, list) else text


# ── Notebook construction helpers ─────────────────────────────────────────────

def _new_id():
    return uuid.uuid4().hex[:8]

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _new_id(),
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }

def clean_cell(cell):
    """Return a copy with outputs cleared, execution_count reset, id refreshed."""
    c = copy.deepcopy(cell)
    c["id"] = _new_id()
    c["outputs"] = []
    if "execution_count" in c:
        c["execution_count"] = None
    return c

def save_nb(cells, path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Written {path}  ({len(cells)} cells)")


# ── Load and patch source notebook ────────────────────────────────────────────

with open(SRC) as f:
    nb_src = json.load(f)

# Build patched cells list: clear outputs, apply patch
patched = []
for cell in nb_src["cells"]:
    c = clean_cell(cell)
    c["source"] = patch_source(c["source"])
    patched.append(c)

RUN_SETUP = make_code_cell('%run nb1_setup.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# nb1_setup.ipynb  — cells 0-3: all definitions (no analysis calls)
# ─────────────────────────────────────────────────────────────────────────────

save_nb(patched[0:4], "nb1_setup.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# nb2_primary_analysis.ipynb  — 10 cells
# Per-repeat MCC, pairwise tests, inter-LLM kappa, confidence calibration
# ─────────────────────────────────────────────────────────────────────────────

MD_PRIMARY = """\
# Notebook 2: Primary Analysis

## Per-Repeat Human vs LLM Comparison

Each model was queried **4 independent times** (repeats 0-3). Within each repeat the model answered every route **4 times** (reruns); majority-vote consensus is the model's prediction - identical to human presence-vote majority.

This gives **one MCC per model per repeat** (4 independent observations). Comparison is symmetric:
- **Humans:** consensus from ~2-4 expert votes (fixed ground truth)
- **LLMs:** consensus from 4 reruns within one repeat

**Uncertainty quantification:** Mean MCC +/- SD, 95% bootstrap CI, exact pairwise permutation test (C(8,4)=70, min p=0.014)"""

MD_KAPPA = """\
## Inter-LLM Agreement: Pairwise Cohen's Kappa

Reviewer suggestion: a pairwise kappa table between LLM majority votes across 204 reactions. Uses repeat=0 for symmetric comparison."""

CODE_KAPPA = """\
# ── Inter-LLM pairwise Cohen's kappa ─────────────────────────────────────────
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

REPEAT_FOR_KAPPA = 0

# preds_per_repeat built in cell above: {llm: {repeat: (y_true, y_pred, hashes)}}
llm_consensus = {}
for llm in LLM_IDENTIFIERS:
    yt, yp, hsh = preds_per_repeat[llm][REPEAT_FOR_KAPPA]
    llm_consensus[llm] = dict(zip(hsh, yp))

kappa_matrix = pd.DataFrame(index=LLM_IDENTIFIERS, columns=LLM_IDENTIFIERS, dtype=float)
for llm in LLM_IDENTIFIERS:
    kappa_matrix.loc[llm, llm] = 1.0

for llm_a, llm_b in combinations(LLM_IDENTIFIERS, 2):
    common = sorted(set(llm_consensus[llm_a]) & set(llm_consensus[llm_b]))
    if len(common) < 2:
        kappa_matrix.loc[llm_a, llm_b] = float("nan")
        kappa_matrix.loc[llm_b, llm_a] = float("nan")
        continue
    ya = [llm_consensus[llm_a][h] for h in common]
    yb = [llm_consensus[llm_b][h] for h in common]
    k = (cohen_kappa_score(ya, yb)
         if len(set(ya)) > 1 and len(set(yb)) > 1 else float("nan"))
    kappa_matrix.loc[llm_a, llm_b] = round(k, 3)
    kappa_matrix.loc[llm_b, llm_a] = round(k, 3)

print("=== Inter-LLM Cohen's kappa (repeat=0 consensus, 204 reactions) ===")
print(kappa_matrix.to_string())
kappa_matrix.to_csv("inter_llm_kappa.csv")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(kappa_matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=-0.2, vmax=1.0, linewidths=0.5, ax=ax,
            xticklabels=LLM_IDENTIFIERS, yticklabels=LLM_IDENTIFIERS,
            annot_kws={"size": 12})
ax.set_title("Pairwise Cohen's kappa between LLM consensus predictions\\n"
             "(repeat 0, sentiment binary)", fontsize=12)
plt.xticks(rotation=20, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("inter_llm_kappa_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved inter_llm_kappa_heatmap.png and inter_llm_kappa.csv")
"""

MD_CALIBRATION = """\
## Confidence Calibration Analysis

Reviewer suggestion: do high-confidence LLM errors cluster where experts disagreed? Uses repeat=0."""

CODE_CALIBRATION = """\
# ── Confidence calibration analysis ──────────────────────────────────────────
REPEAT_FOR_CALIB = 0

def human_agreement_pct_for_rxn(rxn_hash, human_df, group_analyzer):
    grp = human_df[human_df["reaction_hash"] == rxn_hash]
    if grp.empty:
        return float("nan")
    res = group_analyzer.get_group_consensus(
        grp, "source_file", "local_feedback", "local_feedback_text"
    )
    return res["agreement_pct"]

h_agree = {h: human_agreement_pct_for_rxn(h, human_df, group_analyzer_rxn) for h in human_labels}

llm_r0 = llm_df[llm_df["repeat"] == REPEAT_FOR_CALIB]

calib_rows = []
for llm in LLM_IDENTIFIERS:
    sub = llm_r0[llm_r0["source_file"] == llm]
    for rxn_hash, grp in sub.groupby("reaction_hash"):
        if rxn_hash not in human_labels:
            continue
        res = group_analyzer_rxn.get_group_consensus(
            grp, "source_file", "local_feedback", "local_feedback_text"
        )
        sent = res["dominant_sentiment"]
        if sent not in ("Positive", "Negative"):
            continue
        pred    = 1 if sent == "Positive" else 0
        correct = int(pred == human_labels[rxn_hash])
        conf    = grp["confidence"].dropna().mean()
        calib_rows.append({
            "llm": llm,
            "reaction_hash": rxn_hash,
            "confidence": conf,
            "correct": correct,
            "human_agreement_pct": h_agree.get(rxn_hash, float("nan")),
        })

calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv("confidence_calibration.csv", index=False)

# ── Confidence distribution: correct vs error per model ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
for ax, llm in zip(axes.flat, LLM_IDENTIFIERS):
    sub     = calib_df[calib_df["llm"] == llm].dropna(subset=["confidence"])
    correct = sub[sub["correct"] == 1]["confidence"]
    errors  = sub[sub["correct"] == 0]["confidence"]
    ax.hist(correct, bins=20, alpha=0.6, color="steelblue",
            label=f"Correct (n={len(correct)})", density=True)
    ax.hist(errors,  bins=20, alpha=0.6, color="crimson",
            label=f"Error (n={len(errors)})",   density=True)
    ax.set_title(llm, fontsize=11, fontweight="bold")
    ax.set_xlabel("Confidence score (0-100)")
    ax.set_ylabel("Density")
    if len(correct):
        ax.axvline(correct.mean(), color="steelblue", ls="--", lw=1.5)
    if len(errors):
        ax.axvline(errors.mean(),  color="crimson",   ls="--", lw=1.5)
    ax.legend(fontsize=8)
plt.suptitle("Confidence distribution: correct vs error predictions (repeat 0)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confidence_calibration_histograms.png", dpi=200, bbox_inches="tight")
plt.show()

# ── Confidence vs human agreement (errors only) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, llm in zip(axes.flat, LLM_IDENTIFIERS):
    sub = calib_df[(calib_df["llm"] == llm) & (calib_df["correct"] == 0)].dropna(
        subset=["confidence", "human_agreement_pct"]
    )
    if sub.empty:
        ax.set_title(f"{llm} (no errors)")
        continue
    ax.scatter(sub["human_agreement_pct"], sub["confidence"],
               alpha=0.5, s=30, color="crimson")
    z  = np.polyfit(sub["human_agreement_pct"], sub["confidence"], 1)
    xs = np.linspace(sub["human_agreement_pct"].min(), sub["human_agreement_pct"].max(), 50)
    ax.plot(xs, np.poly1d(z)(xs), color="black", lw=1.5)
    corr = sub[["human_agreement_pct", "confidence"]].corr().iloc[0, 1]
    ax.set_title(f"{llm}  (r={corr:.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Human agreement % on reaction")
    ax.set_ylabel("LLM confidence score")
plt.suptitle("LLM errors: confidence vs human agreement\\n"
             "(do models stay confident where humans also disagree?)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("confidence_vs_human_agreement.png", dpi=200, bbox_inches="tight")
plt.show()

print("\\n=== Calibration summary per model ===")
for llm in LLM_IDENTIFIERS:
    sub = calib_df[calib_df["llm"] == llm].dropna(subset=["confidence"])
    c = sub[sub["correct"] == 1]["confidence"]
    e = sub[sub["correct"] == 0]["confidence"]
    if len(c) and len(e):
        print(f"  {llm}:  mean conf correct={c.mean():.1f}  "
              f"mean conf error={e.mean():.1f}  gap={c.mean() - e.mean():.1f}")
print("Saved confidence_calibration.csv, confidence_calibration_histograms.png, "
      "confidence_vs_human_agreement.png")
"""

save_nb(
    [
        RUN_SETUP,
        make_md_cell(MD_PRIMARY),
        patched[5],                      # cell 3: imports + data load + ground truth + MCC loop + bootstrap CI + summary table
        patched[6],                      # cell 4: pairwise permutation test + results
        patched[7],                      # cell 5: bar chart (mcc_uncertainty_plot.png)
        make_md_cell(MD_KAPPA),          # cell 6: markdown
        make_code_cell(CODE_KAPPA),      # cell 7: kappa heatmap
        make_md_cell(MD_CALIBRATION),    # cell 8: markdown
        make_code_cell(CODE_CALIBRATION),# cell 9: calibration analysis
    ],
    "nb2_primary_analysis.ipynb",
)


# ─────────────────────────────────────────────────────────────────────────────
# nb3_human_vs_llm_comparison.ipynb  — 5 cells
# Per-model confusion matrices and agreement statistics (repeat=0).
# Pooled-across-LLMs run_human_vs_llm_analysis calls removed — not relevant,
# since analysis is per individual LLM, not averaged consensus.
# ─────────────────────────────────────────────────────────────────────────────

MD_COMPARISON = """\
# Notebook 3: Human vs LLM Comparison

Per-model confusion matrices and agreement statistics.
LLM data filtered to **repeat=0** for symmetric comparison with human experts."""

save_nb(
    [
        RUN_SETUP,
        make_md_cell(MD_COMPARISON),
        patched[11],   # save_per_llm_outputs + run_human_vs_llm_analysis_per_model definitions
        patched[12],   # run_human_vs_llm_analysis_per_model reactions call
        patched[13],   # run_human_vs_llm_analysis_per_model routes call
    ],
    "nb3_human_vs_llm_comparison.ipynb",
)


# ─────────────────────────────────────────────────────────────────────────────
# nb4_distributions_consistency.ipynb  — 8 cells
# Category distributions (all repeats pooled) + internal consistency (repeat=0)
# ─────────────────────────────────────────────────────────────────────────────

MD_DIST = """\
# Notebook 4: Category Distributions & Internal Consistency

**Distributions** pool all 4 repeats (LLM-only analysis). **Consistency** uses repeat=0."""

nb4_cells = [
    RUN_SETUP,
    make_md_cell(MD_DIST),
    patched[14],   # save_category_distribution_per_llm + run_category_distribution_analysis defs
    patched[15],   # run_category_distribution_analysis reactions call
    patched[16],   # run_category_distribution_analysis routes call
    patched[17],   # agreement functions (analyze_llm_internal_agreement patched, etc.)
    patched[18],   # run_complete_agreement_analysis reactions call
]
# cell 19 = routes call (non-empty); cell 20 = empty — include 19 only
if len(patched) > 19 and "".join(patched[19]["source"]).strip():
    nb4_cells.append(patched[19])

save_nb(nb4_cells, "nb4_distributions_consistency.ipynb")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\nDone. Four notebooks built:")
print("  nb1_setup.ipynb                      - 4 cells  (definitions)")
print("  nb2_primary_analysis.ipynb           - 9 cells  (MCC, kappa, calibration)")
print("  nb3_human_vs_llm_comparison.ipynb    - 5 cells  (per-model confusion matrices)")
print("  nb4_distributions_consistency.ipynb  - 8 cells  (distributions, consistency)")
