"""
Run after all 4 repeats are complete for every model.
1. Regenerates responses_llms/llms_feedback_new_models.csv from all repeat files.
2. Rebuilds ../expert_feedback_combined_new_llms.csv (human rows + fresh LLM rows).
"""
import os
import sys
import json
import re
import pandas as pd

RESPONSES_DIR = "responses_llms"
COMBINED_CSV  = "../expert_feedback_combined_new_llms.csv"
LLM_CSV       = f"{RESPONSES_DIR}/llms_feedback_new_models.csv"

LLM_FILES = {
    "Claude Opus 4.8": "claude_opus48",
    "Gemini 3.1 Pro":  "gemini_35_pro",
    "GPT-5.5":         "gpt55",
    "Llama 3.1 70B":   "llama31_70b",
}
N_REPEATS = 4
N_ROUTES  = 50

# ── Verify all repeat files exist before doing anything ───────────────────────
missing = []
for label, short in LLM_FILES.items():
    for repeat in range(N_REPEATS):
        path = f"{RESPONSES_DIR}/{short}_repeat{repeat}_response_all.json"
        if not os.path.exists(path):
            missing.append(path)

if missing:
    print("ERROR: the following files are missing — jobs may not be finished yet:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)

print("All repeat files present. Starting postprocessing...")

# ── Step 1: regenerate LLM feedback CSV ───────────────────────────────────────
# Import parse/load logic from format_responses (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from format_responses import load_llm_data

all_reaction_rows, all_general_rows = [], []
for llm_label, model_short in LLM_FILES.items():
    print(f"Loading {llm_label}...")
    r_rows, g_rows = load_llm_data(llm_label, model_short)
    all_reaction_rows.extend(r_rows)
    all_general_rows.extend(g_rows)

df_reactions = pd.DataFrame(all_reaction_rows)
df_general   = pd.DataFrame(all_general_rows)
df_llms      = pd.concat([df_reactions, df_general], ignore_index=True)

df_llms.to_csv(LLM_CSV, index=False)
print(f"Saved {len(df_llms)} LLM rows to {LLM_CSV}")

# ── Step 2: rebuild combined CSV ──────────────────────────────────────────────
old_combined = pd.read_csv(COMBINED_CSV)

# Keep only human rows (drop any existing LLM rows)
llm_labels = set(LLM_FILES.keys())
human = old_combined[~old_combined["source_file"].isin(llm_labels)].copy()
print(f"Human rows retained: {len(human)}")

combined = pd.concat([human, df_llms], ignore_index=True)
combined.to_csv(COMBINED_CSV, index=False)
print(f"Saved {len(combined)} total rows to {COMBINED_CSV}")
print(f"  Human rows : {len(human)}")
print(f"  LLM rows   : {len(df_llms)}")
