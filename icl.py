import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
# from transformers import GPT2Tokenizer, OPTForCausalLM
import json
import argparse
import random
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

device = 'cuda'
model_name = 'EleutherAI/gpt-j-6B'




with open('corpus_idx.txt', 'r') as fIn:
    lines = fIn.readlines()
    lines = [line[:-1] for line in lines]
    
    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]

def construct_icl_examples(idx, demos):
    order = [2, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    random.shuffle(order)
    icl_examples = []
    demo_ids = corpus_idx[idx]
    demo_ids = demo_ids[:len(order)]
    for demo_id, o in zip(demo_ids, order):
        line = demos[demo_id-2000]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']
        
        if o == 0:
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        elif o == 1:
            prompt = random.choice(line['paraphrase_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        elif o == 2:
            prompt = random.choice(line['neighborhood_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
    icl_examples.reverse()
    return icl_examples


def icl_lm_eval(model, tokenizer, icl_examples, targets, x):
    ppls = [] 
    for target in targets:
        tgt_len = len(tokenizer.encode(' ' + target))
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
    return ppls

def get_final_probs(yesno_ppls, icl_ppls, orig_ppls):
    yes_prob = 1 / yesno_ppls[0]
    no_prob = 1 / yesno_ppls[1]
    final_probs = [yes_prob / icl_ppls[0] + no_prob / orig_ppls[0], yes_prob / icl_ppls[1] + no_prob / orig_ppls[1]]
    return final_probs


if __name__ == '__main__':
    # random.seed(42)
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    model = GPTJForCausalLM.from_pretrained(model_name).to(device)
    # model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    # model = GPTNeoXForCausalLM.from_pretrained(model_name).half().to(device)
    # model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)
    # model = OPTForCausalLM.from_pretrained("facebook/opt-13b").to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-13b")

    lines = []

    with open('./counterfact.json', 'r') as f:
        lines = json.load(f)
    icl_examples = []
    demos = lines[2000:]
    lines = lines[:2000]
    calibrate_magnitude = .0
    success_cnt = 0
    para_success_cnt = 0
    magnitude = .0
    para_magnitude = .0
    orig_magnitude = .0
    total_cnt = 0
    para_total_cnt = 0
    orig_success_cnt = 0
    orig_total_cnt = 0

    # icl_cnt = 0
    example_idx = 0
    for i, line in enumerate(lines):

        if i % 10 == 0:
            print(i, success_cnt, total_cnt, magnitude / (total_cnt + 1e-12), para_success_cnt, para_magnitude / (para_total_cnt + 1e-12), orig_success_cnt ,orig_magnitude / (i + 1e-12))
        relation = line['requested_rewrite']['relation_id']
        prompt = line['requested_rewrite']['prompt']
        subject = line['requested_rewrite']['subject']
        prompt_calibrate = prompt.format('SUBJECT')
        prompt = prompt.format(subject)
        PROMPTS = [prompt, prompt_calibrate]

        target_true = line['requested_rewrite']['target_true']['str']
        target_new = line['requested_rewrite']['target_new']['str']
        
        PPLs = []
        targets = [target_new, target_true]
        icl_examples = construct_icl_examples(example_idx, demos)


        icl_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')

        example_idx += 1
       
        edit_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_new, target_true], f'New Fact: {prompt} {target_new}\nPrompt: {prompt}')

        edit_final_probs = [1 / edit_ppls[0], 1 / edit_ppls[1]]
        orig_total_cnt += 1
        if edit_final_probs[0] > edit_final_probs[1]:
            orig_success_cnt += 1
        orig_magnitude += edit_final_probs[0] - edit_final_probs[1]


        targets = [target_new, target_true]

        paraphrases = line['paraphrase_prompts']
        for paraphrase in paraphrases:
            paraphrase_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_new, target_true], f'New Fact: {prompt} {target_new}\nPrompt: {paraphrase}')
            paraphrase_final_probs = [1 / paraphrase_ppls[0], 1 / paraphrase_ppls[1]]
            
            if paraphrase_final_probs[0] > paraphrase_final_probs[1]:
                para_success_cnt += 1
            para_magnitude += paraphrase_final_probs[0] - paraphrase_final_probs[1]
            para_total_cnt += 1

        neighbors = line['neighborhood_prompts']
        for neighbor in neighbors:
            neighbor_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_true, target_new], f'New Fact: {prompt} {target_new}\nPrompt: {neighbor}')
            neighbor_final_probs = [1 / neighbor_ppls[0], 1 / neighbor_ppls[1]]
            
            if neighbor_final_probs[0] > neighbor_final_probs[1]:
                success_cnt += 1
            magnitude += neighbor_final_probs[0] - neighbor_final_probs[1]
            total_cnt += 1



    print(success_cnt/total_cnt, magnitude/total_cnt, para_success_cnt/para_total_cnt, para_magnitude/para_total_cnt, orig_success_cnt/orig_total_cnt, orig_magnitude/orig_total_cnt)
