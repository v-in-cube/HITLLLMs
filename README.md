## Description

This repository contains the code and data for the study ["Do humans and large language models agree on the quality of synthesis plans?"](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001730/v2).

> The original preprint analysis is preserved in the [`preprint`](https://github.com/v-in-cube/HITLLLMs/tree/preprint) branch.

---

## Data

Human expert and LLM feedback on 50 retrosynthetic routes (204 reactions) is combined in [`expert_feedback_combined_new_llms.csv`](expert_feedback_combined_new_llms.csv). LLMs (Claude Opus 4.8, Gemini 3.1 Pro, GPT-5.5, Llama 3.1 70B) were each queried in 4 independent repeats × 4 reruns per route. Raw aggregated LLM responses are in [`llms_querying/responses_llms/`](llms_querying/responses_llms/). The retrosynthetic routes are in [`llms_querying/master_paths.json`](llms_querying/master_paths.json).

---

## Analysis notebooks

Run `nb1_setup.ipynb` first — it defines all shared classes and functions used by the other notebooks.

### [`nb1_setup.ipynb`](nb1_setup.ipynb) — Definitions
Imports, category mappings, and consensus/voting logic (pessimistic majority vote with tie-breaking).

### [`nb2_primary_analysis.ipynb`](nb2_primary_analysis.ipynb) — Primary statistical analysis
Per-repeat MCC with bootstrap CI and exact pairwise permutation tests; inter-LLM Cohen's kappa averaged across repeats; confidence calibration analysis (error clustering vs reaction difficulty).

### [`nb3_human_vs_llm_comparison.ipynb`](nb3_human_vs_llm_comparison.ipynb) — Human vs LLM comparison
Averaged TP/FP/TN/FN confusion matrices (raw counts and row-normalised rates); full category-level confusion matrices.

### [`nb4_distributions_consistency.ipynb`](nb4_distributions_consistency.ipynb) — Distributions and internal consistency
Category distributions per LLM vs human baseline (all repeats pooled); internal consistency (agreement levels across 4 reruns, averaged across repeats ± SD) including human expert consistency as reference.

---

## LLM querying

Scripts for querying each model are in [`llms_querying/`](llms_querying/):

| Script | Model | API |
|---|---|---|
| `claude_opus48.py` | `us.anthropic.claude-opus-4-8` | Amazon Bedrock via AI Gateway |
| `gemini_35_pro.py` | `gemini-3.1-pro-preview` | Vertex AI Express via AI Gateway |
| `gpt55.py` | `gpt-5.5` | OpenAI via AI Gateway |
| `llama31_70b.py` | `us.meta.llama3-1-70b-instruct-v1:0` | Amazon Bedrock via AI Gateway |

All scripts support resumable execution. After all jobs finish, run [`llms_querying/postprocess.py`](llms_querying/postprocess.py) to rebuild the combined CSV. The prompt is in [`llms_querying/feasibility.py`](llms_querying/feasibility.py).

---

## Environment

```bash
conda env create -f environment.yml
conda activate stats_hitl_llms
```

Set `AI_GATEWAY_URL` and `AI_GATEWAY_KEY` in a `.env` file before running querying scripts.

---

## License
[MIT](https://choosealicense.com/licenses/mit/)
