## Description
This repo contains supporting code (mostly statistical analysis) and raw feedback materials for the preprint ["Do humans and large language models agree on the quality of synthesis plans?"](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001730/v2). All of the analysis on comparing human experts' feedback and LLMs' feedback is in the [human_vs_llm.ipynb](https://github.com/v-in-cube/HITLLLMs/blob/main/human_vs_llm.ipynb) file. Also this repo contains raw human experts and LLMs reponses on the retrosynthetic trees of our interest.\
\
The plots for the paper can be re-generated from scratch with [human_vs_llm.ipynb](https://github.com/v-in-cube/HITLLLMs/blob/main/human_vs_llm.ipynb) file.\
\
In the [llms_querying](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/) folder there is [llms_querying.py](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/llms_querying.py) file from which we generated responses of LLMs and also this folder contains subfolder [responses_llms](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/responses_llms) which has raw json responses that were later parsed into a dataframe in the root of the repository [expert_feedback_combined_llms.csv](https://github.com/v-in-cube/HITLLLMs/blob/main/expert_feedback_combined_llms.csv). Also in the [llms_querying](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/) folder one could find [master_paths.json](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/master_paths.json) file which is the modified file with AIZynthFinder routes which were presented to chemist experts. File [feasibility.py](https://github.com/v-in-cube/HITLLLMs/blob/main/llms_querying/feasibility.py) in this folder contains prompt for LLMs.

## Environment
Necessary packages can be installed from the .yml file with 
```
conda env create -f environment.yml
conda activate stats_hitl_llms
```

For the access to the OpenAI and VertexAI services enter your credentials in the .env file.

## License
[MIT](https://choosealicense.com/licenses/mit/)
