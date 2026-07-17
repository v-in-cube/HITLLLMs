import os
import json
import copy
import re
from datetime import date
import boto3
from botocore.config import Config
from dotenv import dotenv_values
from rdkit.Chem import AllChem
from feasibility import prefix as prompt_prefix
from feasibility import suffix as prompt_suffix
from hash_utils import build_hash_map, compress_hashes, restore_hashes

gateway_url = os.getenv("AI_GATEWAY_URL")
api_key = os.getenv("AI_GATEWAY_KEY")
if not gateway_url or not api_key:
    env = dotenv_values("../.env")
    gateway_url = gateway_url or env.get("AI_GATEWAY_URL")
    api_key = api_key or env.get("AI_GATEWAY_KEY")
if not gateway_url or not api_key:
    raise ValueError("AI_GATEWAY_URL and AI_GATEWAY_KEY must be set")

ENDPOINT = f"{gateway_url}/bedrock"
AWS_REGION_NAME = "us-east-1"
MODEL_ID = "us.anthropic.claude-opus-4-8"
TEMPERATURE = 1.0
N_ROUTES = 50
N_RERUNS = 4
N_REPEATS = 4
MODEL_SHORT = "claude_opus48"

MAX_TOKENS = 8192

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key


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


boto3_client = boto3.client(
    region_name=AWS_REGION_NAME,
    service_name="bedrock-runtime",
    endpoint_url=ENDPOINT,
    aws_access_key_id="",
    aws_secret_access_key="",
    config=Config(
        retries={"total_max_attempts": 4, "mode": "adaptive"},
        connect_timeout=60,
        read_timeout=1000,
    ),
)

system_prompt = prompt_prefix + " " + prompt_suffix

for repeat in range(N_REPEATS):
    all_file = f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_all.json"
    if os.path.exists(all_file):
        print(f"Skipping repeat {repeat} (already complete)")
        continue

    repeat_responses_all = []
    for i in range(N_ROUTES):
        route_file = f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_{i}.json"
        if os.path.exists(route_file):
            with open(route_file) as f:
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
            response_obj = boto3_client.converse(
                modelId=MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": route_text}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
            )
            response = restore_hashes(response_obj["output"]["message"]["content"][0]["text"], short_to_orig)
            repeat_responses_all.append(response)
            route_responses.append(response)
        with open(route_file, "w") as f:
            json.dump(route_responses, f)

    with open(all_file, "w") as f:
        json.dump(repeat_responses_all, f)

    cleaned = []
    for i in range(N_ROUTES):
        with open(f"responses_llms/{MODEL_SHORT}_repeat{repeat}_response_{i}.json") as f:
            r = json.load(f)
        for string in r:
            clean_json = re.sub(r"^```json\n|\n```$", "", string, flags=re.MULTILINE)
            cleaned.append(clean_json)
    with open(all_file, "w") as f:
        json.dump(cleaned, f)

metadata = {
    "model_short_name": MODEL_SHORT,
    "model_id": MODEL_ID,
    "provider": "Amazon Bedrock via AI Gateway",
    "gateway_endpoint": ENDPOINT,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "n_routes": N_ROUTES,
    "n_reruns_per_route": N_RERUNS,
    "n_repeats": N_REPEATS,
    "total_responses": N_ROUTES * N_RERUNS * N_REPEATS,
    "run_date": date.today().isoformat(),
}
with open(f"responses_llms/{MODEL_SHORT}_run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done. Metadata saved to responses_llms/{MODEL_SHORT}_run_metadata.json")
