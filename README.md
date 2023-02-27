# IKE
Source code for "Can We Edit Factual Knowledge by In-Context Learning?"
## Overview
## Requirements
```
jsonlines==3.1.0
nltk==3.6.7
numpy==1.22.3
openai==0.25.0
sentence_transformers==2.2.0
spacy==3.2.3
torch==1.11.0
tqdm==4.56.0
transformers==4.24.0

```
## How to run our experiments?
### Data Preparation
We conduct experiments on `CounterFact` dataset. You can download the dataset [here](https://rome.baulab.info/data/dsets/counterfact.json).
run code `clean_paraphrase.py` to remove unrelated prefixes in paraphrase prompts in `CounterFact` and keep all prompts in the same format.

### Demonstration Organization
We select first 2,000 records as test set. For each record, we use sentence transformers to retrieve 32 nearest neighbours from remaining records as in-context demonstrations. The indices of nearest neighbours are stored in `corpus_idx.txt`. You can also run code `encode_facts.py` and `semantic_search.py` to build `corpus_idx.txt`. 

### Main Experiments
IKE: `python icl.py`

PROMPT: `python prompt.py`

Other baselines: implemented by [rome](https://github.com/kmeng01/rome)

### Models for IKE

You can evaluate IKE based on different LLMs by specifying the model name of LLMs.

```
python icl.py --model_name [model name]
```
The model name can be `['gpt2-xl', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']`

### Contrastive Knowledge Assessment


