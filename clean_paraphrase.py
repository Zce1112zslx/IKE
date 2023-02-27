import json
data_path = './counterfact.json'
with open(data_path, 'r') as f:
    lines = json.load(f)

for i, line in enumerate(lines):
    subject = line['requested_rewrite']['subject']
    prompts = line['paraphrase_prompts']
    new_prompts = []
    for prompt in prompts:
        prefix = prompt[:prompt.find(subject)]
        while '.' in prefix:
            prefix = prefix[prefix.find('.')+1:]
        while '\n' in prefix:
            prefix = prefix[prefix.find('\n')+1:]
        new_prompt = (prefix+prompt[prompt.find(subject):]).strip()
        if 'Category' in new_prompt:
            new_prompt = new_prompt[new_prompt.find(subject):]
        new_prompts.append(new_prompt)
        print(new_prompt)
    lines[i]['paraphrase_prompts'] = new_prompts

with open(data_path, 'w') as f:
    json.dump(lines, f, indent=2)