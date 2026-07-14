import os
import json
import copy
import re
from datetime import date
from dotenv import dotenv_values
from rdkit.Chem import AllChem
from google.genai import Client
from google.genai.types import HttpOptions, Content, Part, GenerateContentConfig

from feasibility import prefix as prompt_prefix
from feasibility import suffix as prompt_suffix
from hash_utils import build_hash_map, compress_hashes, restore_hashes

gateway_url = os.getenv("AI_GATEWAY_URL")
api_key = os.getenv("AI_GATEWAY_KEY")
if not gateway_url or not api_key:
    env = dotenv_values("../.env")
    gateway_url = gateway_url or env.get("AI_GATEWAY_URL") or "https://ai-gateway.astrazeneca.net"
    api_key = api_key or env.get("AI_GATEWAY_KEY")
if not gateway_url or not api_key:
    raise ValueError("AI_GATEWAY_URL and AI_GATEWAY_KEY must be set")

ENDPOINT = f"{gateway_url}/vertex-ai-express"
MODEL_ID = "gemini-3.1-pro-preview"
TEMPERATURE = 1.0
N_ROUTES = 50
N_RERUNS = 4
N_REPEATS = 4
MODEL_SHORT = "gemini_35_pro"


def remove_image_paths(data):
    image_keys = [
        "image_paths", "image_path", "highlighted_image_paths", "highlighted_image_path",
        "merged_image_paths", "merged_image_path", "has_multiple_images", "is_chemical",
        "is_reaction", "scores", "library_occurence", "policy_probability",
        "policy_probability_rank", "policy_name", "template_code", "template",
        "hide", "created_at_iteration", "template_hash", "classification",
    ]
    if isinstance(data, dict):
        return {k: remove_image_paths(v) for k, v in data.items() if k not in image_keys}
    elif isinstance(data, list):
        return [remove_image_paths(item) for item in data]
    return data


def remove_molecule_hashes(data, is_root=True):
    if isinstance(data, dict):
        result = copy.deepcopy(data)
        if data.get("type") == "mol" and not is_root:
            result.pop("hash", None)
        if data.get("type") == "reaction" and not is_root:
            result.pop("smiles", None)
            if "metadata" in result and "mapped_reaction_smiles" in result["metadata"]:
                rxn = AllChem.ReactionFromSmarts(result["metadata"]["mapped_reaction_smiles"], useSmiles=True)
                AllChem.RemoveMappingNumbersFromReactions(rxn)
                result["smiles"] = AllChem.ReactionToSmiles(rxn)
                result.pop("metadata", None)
        if "children" in result:
            result["children"] = [remove_molecule_hashes(child, is_root=False) for child in result["children"]]
        return result
    elif isinstance(data, list):
        return [remove_molecule_hashes(item, is_root=True) for item in data]
    return data


with open("./master_paths.json", "r") as f:
    test_input = json.load(f)
test_input = remove_image_paths(test_input)
processed = remove_molecule_hashes(test_input, is_root=True)

client = Client(vertexai=True, api_key=api_key, http_options=HttpOptions(base_url=ENDPOINT))

system_prompt = prompt_prefix + " " + prompt_suffix

for repeat in range(N_REPEATS):
    # Skip repeats that already have a complete _all.json
    all_file = f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_all.json"
    if os.path.exists(all_file):
        print(f"Skipping repeat {repeat} (already complete)")
        continue

    # Reconstruct accumulated responses from any per-route files already saved
    repeat_responses_all = []
    for i in range(N_ROUTES):
        route_file = f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_{i}.json"
        if os.path.exists(route_file):
            with open(route_file, "r") as f:
                repeat_responses_all.extend(json.load(f))

    for i in range(N_ROUTES):
        route_file = f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_{i}.json"
        if os.path.exists(route_file):
            print(f"Skipping repeat {repeat} route {i} (already saved)")
            continue

        short_to_orig, orig_to_short = build_hash_map(processed[i])
        route_responses = []
        for j in range(N_RERUNS):
            route_text = compress_hashes(str(processed[i]), orig_to_short)
            response_obj = client.models.generate_content(
                model=MODEL_ID,
                contents=Content(role="user", parts=[Part(text=route_text)]),
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                ),
            )
            response = restore_hashes(response_obj.text, short_to_orig)
            repeat_responses_all.append(response)
            route_responses.append(response)
        with open(route_file, "w") as f:
            json.dump(route_responses, f)

    with open(all_file, "w") as f:
        json.dump(repeat_responses_all, f)

    cleaned = []
    for i in range(N_ROUTES):
        with open(f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_{i}.json", "r") as f:
            r = json.load(f)
        for string in r:
            clean_json = re.sub(r"^```json\n|\n```$", "", string, flags=re.MULTILINE)
            cleaned.append(clean_json)
    with open(all_file, "w") as f:
        json.dump(cleaned, f)

metadata = {
    "model_short_name": MODEL_SHORT,
    "model_id": MODEL_ID,
    "provider": "Google Vertex AI Express via AI Gateway",
    "gateway_endpoint": ENDPOINT,
    "temperature": TEMPERATURE,
    "n_routes": N_ROUTES,
    "n_reruns_per_route": N_RERUNS,
    "n_repeats": N_REPEATS,
    "total_responses": N_ROUTES * N_RERUNS * N_REPEATS,
    "run_date": date.today().isoformat(),
}
with open(f"responses_llms/{MODEL_SHORT}_run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done. Metadata saved to responses_llms/{MODEL_SHORT}_run_metadata.json")
