import json
import re
import os
import pandas as pd

RESPONSES_DIR = "responses_llms"
RESPONSES_PER_ROUTE = 4
N_ROUTES = 50
N_REPEATS = 4


def _load_valid_hashes(master_paths_file="master_paths.json"):
    """
    Load all valid hashes from master_paths.json.
    Returns (all_hashes, root_hashes) where root_hashes are the top-level
    molecule hashes — one per route (type='mol', direct children of the list).
    """
    def collect_all(node, found):
        if isinstance(node, dict):
            if "hash" in node:
                found.add(node["hash"])
            for v in node.values():
                collect_all(v, found)
        elif isinstance(node, list):
            for item in node:
                collect_all(item, found)

    with open(master_paths_file) as f:
        routes = json.load(f)

    all_found = set()
    collect_all(routes, all_found)

    # Root hashes are the 'hash' of the top-level mol node in each route
    root_found = set()
    route_idx_to_root = {}
    for idx, route in enumerate(routes):
        if isinstance(route, dict) and route.get("type") == "mol" and "hash" in route:
            root_found.add(route["hash"])
            route_idx_to_root[idx] = route["hash"]

    return all_found, root_found, route_idx_to_root


VALID_HASHES, ROOT_HASHES, _ROUTE_IDX_TO_ROOT_HASH = _load_valid_hashes()

# ── Category normalisation ────────────────────────────────────────────────────

VALID_REACTION_CATEGORIES = {
    "Reaction feasible, all good",
    "Reaction feasible, unexpected disconnection",
    "Protecting group strategy is wrong / non-optimal",
    "Non-optimal reagent",
    "Unnecessary step",
    "Selectivity (regio-, stereo-, chemo-) issues",
    "Problems with reaction type and functional group compatibility",
    "Unlikely disconnection",
}

VALID_ROUTE_CATEGORIES = {
    "Route feasible as it is",
    "Route feasible with few modifications",
    "Route feasible with significant modifications",
    "Route unfeasible",
    "Route was not solved to building blocks",
}

# Explicit aliases: non-standard strings → canonical valid category
REACTION_ALIASES = {
    # shortened / variant names (slash vs space around slash)
    "Functional group compatibility problems":
        "Problems with reaction type and functional group compatibility",
    "Protecting group strategy is wrong/non-optimal":
        "Protecting group strategy is wrong / non-optimal",
    # hybrid "Reaction feasible, <issue>" phrases — map to the specific issue category
    # "few modifications" = minor issue → non-optimal reagent (most common minor issue)
    "Reaction feasible with few modifications":
        "Non-optimal reagent",
    # "significant modifications" = serious issue → selectivity/FG problems
    "Reaction feasible with significant modifications":
        "Selectivity (regio-, stereo-, chemo-) issues",
    # unexpected disconnection variant spellings
    "Reaction feasible with unexpected disconnection":
        "Reaction feasible, unexpected disconnection",
    # selectivity variants
    "Reaction feasible with selectivity issues":
        "Selectivity (regio-, stereo-, chemo-) issues",
    "Reaction feasible, selectivity issues":
        "Selectivity (regio-, stereo-, chemo-) issues",
    "Reaction feasible, with selectivity issues":
        "Selectivity (regio-, stereo-, chemo-) issues",
    # functional group compatibility variants
    "Reaction feasible, with functional group compatibility problems":
        "Problems with reaction type and functional group compatibility",
    # reagent variant
    "Reaction feasible, non-optimal reagent":
        "Non-optimal reagent",
    # protecting group variant
    "Reaction feasible, protecting group strategy is wrong":
        "Protecting group strategy is wrong / non-optimal",
    # typo: missing closing parenthesis
    "Selectivity (regio-, stereo-, chemo- issues":
        "Selectivity (regio-, stereo-, chemo-) issues",
}

ROUTE_ALIASES = {
    # The notebook uses "Route feasible" as a short form for "Route feasible as it is"
    "Route feasible": "Route feasible as it is",
}

# Separators used by LLMs to join multiple categories in one string.
# Only split on semicolons or " / " (slash with spaces) — NOT plain commas,
# because commas appear inside valid category names (e.g. "Reaction feasible, all good").
_MULTI_SEP = re.compile(r"\s*;\s*|\s+/\s+|\s+\+\s+")


def _split_and_normalise(raw_value, aliases, valid_set):
    """
    Split a potentially multi-category string, apply aliases, and return only
    the valid canonical categories as a list.
    """
    if not isinstance(raw_value, str):
        return []
    parts = _MULTI_SEP.split(raw_value.strip())
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mapped = aliases.get(part, part)
        if mapped in valid_set:
            result.append(mapped)
        else:
            lower = part.lower()
            match = next((v for v in valid_set if v.lower() == lower), None)
            if match:
                result.append(match)
    return result


# Hierarchy scores for pessimistic selection (lower = more pessimistic)
_REACTION_HIERARCHY = {
    "Reaction feasible, all good": 5,
    "Reaction feasible, unexpected disconnection": 5,
    "Non-optimal reagent": 4,
    "Unnecessary step": 4,
    "Protecting group strategy is wrong / non-optimal": 3,
    "Selectivity (regio-, stereo-, chemo-) issues": 2,
    "Problems with reaction type and functional group compatibility": 2,
    "Unlikely disconnection": 1,
}
_ROUTE_HIERARCHY = {
    "Route feasible as it is": 5,
    "Route feasible with few modifications": 4,
    "Route feasible with significant modifications": 2,
    "Route unfeasible": 1,
    "Route was not solved to building blocks": 1,
}


def _most_pessimistic(categories, hierarchy):
    """Return the single most pessimistic category from a list (lowest hierarchy score)."""
    if not categories:
        return None
    return min(categories, key=lambda c: hierarchy.get(c, -1))


LLM_FILES = {
    "Claude Opus 4.8": "claude_opus48",
    "Gemini 3.1 Pro":  "gemini_35_pro",
    "GPT-5.5":         "gpt55",
    "Llama 3.1 70B":   "llama31_70b",
}


def parse_response(raw):
    """Parse a single LLM response string into a dict. Returns None on failure."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()

    # Try direct parse first (already clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    text2 = re.sub(r"^```json\s*", "", text)
    text2 = re.sub(r"\s*```$", "", text2).strip()
    try:
        return json.loads(text2)
    except json.JSONDecodeError:
        pass

    # Extract the outermost {...} block (handles preamble/postamble text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # "Extra data" case: valid JSON object followed by trailing prose.
    # Walk backward from each } to find the first complete valid object.
    if start != -1:
        for end in range(len(text) - 1, start, -1):
            if text[end] == "}":
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

    return None


def load_llm_data(llm_label, model_short):
    reaction_rows = []
    general_rows = []
    parse_errors = 0
    multi_choice_count = 0

    for repeat in range(N_REPEATS):
        path = os.path.join(RESPONSES_DIR, f"{model_short}_repeat{repeat}_response_all.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping repeat {repeat}")
            continue

        with open(path, "r") as f:
            responses = json.load(f)

        print(f"  repeat {repeat}: {len(responses)} responses loaded from {os.path.basename(path)}")

        for route_idx in range(N_ROUTES):
            start = route_idx * RESPONSES_PER_ROUTE
            end = start + RESPONSES_PER_ROUTE
            route_responses = responses[start:end]

            for iteration, raw in enumerate(route_responses, start=1):
                parsed = parse_response(raw)
                if parsed is None:
                    print(f"    Parse error: {llm_label} repeat={repeat} route={route_idx} iter={iteration}")
                    parse_errors += 1
                    continue

                # Determine root molecule hash from general_feedback.
                # Only accept hashes that are in the validated ROOT_HASHES set;
                # hallucinated values (e.g. "root", "root_molecule_hash", SMILES)
                # are replaced with the correct root hash for this route_idx.
                root_hash = None
                gf = parsed.get("general_feedback", {})
                if isinstance(gf, dict) and gf:
                    candidate = next(iter(gf))
                    if candidate in ROOT_HASHES:
                        root_hash = candidate
                if root_hash is None:
                    # Fall back to the actual root hash for this route from master_paths
                    root_hash = _ROUTE_IDX_TO_ROOT_HASH.get(route_idx)

                # Reaction-level feedback
                # Skip hashes that are root molecule hashes (mis-placed route entry)
                # or hallucinated (not present in master_paths at all).
                for reaction_hash, fd in parsed.get("feedback", {}).items():
                    if reaction_hash in ROOT_HASHES:
                        continue
                    if reaction_hash not in VALID_HASHES:
                        continue
                    raw_feedback = fd.get("feedback", "Unknown")
                    feedback_text = fd.get("feedback_text", "")
                    confidence = fd.get("confidence", None)

                    # Collect all categories from this single rerun.
                    if isinstance(raw_feedback, list):
                        multi_choice_count += 1
                        raw_values = raw_feedback
                    else:
                        raw_values = [raw_feedback]

                    normalised = []
                    for rv in raw_values:
                        normalised.extend(_split_and_normalise(rv, REACTION_ALIASES, VALID_REACTION_CATEGORIES))

                    # Each rerun counts as ONE vote: if the model returned multiple
                    # categories, take only the most pessimistic to stay consistent
                    # with the pessimistic tie-breaking used at consensus time.
                    if len(normalised) > 1:
                        normalised = [_most_pessimistic(normalised, _REACTION_HIERARCHY)]
                    elif not normalised:
                        continue

                    for cat in normalised:
                        reaction_rows.append({
                            "source_file":         llm_label,
                            "repeat":              repeat,
                            "route_idx":           route_idx,
                            "root_molecule_hash":  root_hash,
                            "reaction_hash":       reaction_hash,
                            "iteration":           iteration,
                            "local_feedback":      cat,
                            "local_feedback_text": feedback_text,
                            "confidence":          confidence,
                            "feedback_type":       "reaction",
                        })

                # Route-level (general) feedback
                # Skip hallucinated root hashes (not in the valid ROOT_HASHES set)
                for hash_id, fd in gf.items():
                    if hash_id not in ROOT_HASHES:
                        continue
                    raw_feedback = fd.get("feedback", "Unknown")
                    feedback_text = fd.get("feedback_text", "")
                    confidence = fd.get("confidence", None)

                    if isinstance(raw_feedback, list):
                        multi_choice_count += 1
                        raw_values = raw_feedback
                    else:
                        raw_values = [raw_feedback]

                    normalised = []
                    for rv in raw_values:
                        normalised.extend(_split_and_normalise(rv, ROUTE_ALIASES, VALID_ROUTE_CATEGORIES))

                    # Same one-vote-per-rerun rule for route feedback
                    if len(normalised) > 1:
                        normalised = [_most_pessimistic(normalised, _ROUTE_HIERARCHY)]
                    elif not normalised:
                        continue

                    for cat in normalised:
                        general_rows.append({
                            "source_file":          llm_label,
                            "repeat":               repeat,
                            "route_idx":            route_idx,
                            "root_molecule_hash":   hash_id,
                            "iteration":            iteration,
                            "general_feedback":     cat,
                            "general_feedback_text": feedback_text,
                            "confidence":           confidence,
                            "feedback_type":        "general",
                        })

    print(f"  {llm_label}: {len(reaction_rows)} reaction rows, {len(general_rows)} general rows "
          f"({parse_errors} parse errors, {multi_choice_count} multi-choice responses)")
    return reaction_rows, general_rows


all_reaction_rows = []
all_general_rows = []

for llm_label, model_short in LLM_FILES.items():
    print(f"Loading {llm_label}...")
    r_rows, g_rows = load_llm_data(llm_label, model_short)
    all_reaction_rows.extend(r_rows)
    all_general_rows.extend(g_rows)

df_reactions = pd.DataFrame(all_reaction_rows)
df_general   = pd.DataFrame(all_general_rows)
df_all       = pd.concat([df_reactions, df_general], ignore_index=True)

out_path = "responses_llms/llms_feedback_new_models.csv"
df_all.to_csv(out_path, index=False)

print(f"\nSaved {len(df_all)} total rows to {out_path}")
print(f"  Reaction rows : {len(df_reactions)}")
print(f"  General rows  : {len(df_general)}")
print(f"  Columns       : {list(df_all.columns)}")
