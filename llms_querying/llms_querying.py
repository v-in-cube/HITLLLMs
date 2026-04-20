import dotenv
dotenv.load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
import json
import copy
from rdkit.Chem import AllChem
import re

from feasibility import prefix as prompt_prefix
from feasibility import suffix as prompt_suffix

def remove_image_paths(data):
    """
    Recursively remove all image-related keys from nested dictionaries and lists
    """
    image_keys = [
        "image_paths", 
        "image_path", 
        "highlighted_image_paths", 
        "highlighted_image_path",
        "merged_image_paths",
        "merged_image_path",
        "has_multiple_images",
        "is_chemical",
        "is_reaction",
        "scores", 
        "library_occurence",
        "policy_probability",
        "policy_probability_rank",
        "policy_name",
        "template_code",
        "template",
        "hide",
        "created_at_iteration",
        "template_hash",
        "classification",
    ]
    
    if isinstance(data, dict):
        # Create a new dict without the image keys
        return {
            k: remove_image_paths(v) 
            for k, v in data.items() 
            if k not in image_keys
        }
    elif isinstance(data, list):
        # Process each item in the list
        return [remove_image_paths(item) for item in data]
    else:
        # Return unchanged for non-dict and non-list items
        return data
 

def remove_molecule_hashes(data, is_root=True):
    """
    Recursively remove hashes from molecules except for the root molecule.
    Keeps reaction hashes untouched.
    
    Args:
        data: The JSON object (dict or list)
        is_root: Boolean indicating if this is the root molecule
    
    Returns:
        Modified data with molecule hashes removed (except root)
    """
    if isinstance(data, dict):
        # Make a copy to avoid modifying the original
        result = copy.deepcopy(data)
        
        # If this is a molecule and not the root, remove its hash
        if data.get("type") == "mol" and not is_root:
            result.pop("hash", None)
        if data.get("type") == "reaction" and not is_root:
            result.pop("smiles", None)
            if ('metadata' in result and 
                'mapped_reaction_smiles' in result['metadata']):
                
                
                rxn = AllChem.ReactionFromSmarts(result['metadata']['mapped_reaction_smiles'], useSmiles=True)
                AllChem.RemoveMappingNumbersFromReactions(rxn)
                result['smiles'] = AllChem.ReactionToSmiles(rxn)
                # Remove the entire metadata section
                result.pop('metadata', None)
        # Recursively process children
        if "children" in result:
            result["children"] = [
                remove_molecule_hashes(child, is_root=False) 
                for child in result["children"]
            ]
        
        return result
    
    elif isinstance(data, list):
        return [remove_molecule_hashes(item, is_root=True) for item in data]
    
    else:
        return data

with open('./master_paths.json', 'r') as f: #preprocessing our retrosynthetic data for LLM input so it is clean and neat. It is also dumped in file "clean_routes_for_querying.json"
    test_input = json.load(f)
test_input = remove_image_paths(test_input)
processed = remove_molecule_hashes(test_input, is_root=True)


def get_fresh_llm(llm_name):
    if llm_name=="gpt4.1":
        return ChatOpenAI(model="gpt-4.1")
    elif llm_name=="claude":
        return ChatAnthropic(model='claude-sonnet-4-5-20250929')
    elif llm_name=="gemini":
        return ChatVertexAI(model_name="gemini-2.5-pro")
    elif llm_name=="gpto3":
        return ChatOpenAI(model="o3")
    
for llm in ["gpt4.1", "claude", "gemini", "gpto3"]:
    llm_response_2 = []
    for i in range(50): #iterating over 50 retrosynthetic paths
        llm_response_j=[]
        for j in range(5): #5 re-runs of fresh instances of LLMs
            test_input = str(processed[i])
            system_prompt = prompt_prefix+" "+prompt_suffix
            prompt = [
                (
                    "system",
                    system_prompt
                ),
                    ("human",
                    test_input
                    )
            ]
            llm = get_fresh_llm()
            ai_msg = llm.invoke(prompt)
            response = ai_msg.content
            llm_response_2.append(response)
            llm_response_j.append(response)
        with open(f"{llm}_repeated_response_{i}.json", 'w') as f:
            json.dump(llm_response_j, f)    
    with open(f"{llm}_repeated_response_all.json", 'w') as f:
        json.dump(llm_response_2, f)


llm_response = []
for i in range(50):
   with open(f"{llm}_repeated_response_{i}.json", 'r') as f:
        r = json.load(f)
        for string in r:
         clean_json = re.sub(r'^```json\n|\n```$', '', string, flags=re.MULTILINE)
         llm_response.append(clean_json)
with open(f"{llm}_repeated_response_all.json", 'w') as f:
    json.dump(llm_response, f)