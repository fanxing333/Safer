from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

def load_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:3"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=256):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    # 选择数据集
    data_files = {
        "instruction_attack_scenarios.json": ['Reverse_Exposure', 'Goal_Hijacking', 'Prompt_Leaking', 'Unsafe_Instruction_Topic', 'Role_Play_Instruction', 'Inquiry_With_Unsafe_Opinion'],
        "typical_safety_scenarios.json": ['Unfairness_And_Discrimination', 'Crimes_And_Illegal_Activities', 'Insult', 'Mental_Health', 'Physical_Harm', 'Privacy_And_Property', 'Ethics_And_Morality']
    }
    traget_files = list(data_files.keys())[0]
    target_field = data_files[traget_files][0]

    dataset = load_dataset("thu-coai/Safety-Prompts", data_files=traget_files, field=target_field, split='train')
    print(dataset)

    # 选择模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = load_model(model_name=model_name)
    output_file_name = f"output/{model_name.split('/')[1]}_{traget_files}_{target_field}_infer.jsonl"

    with open(output_file_name, 'w', encoding='utf-8') as f:

        for item in tqdm(dataset, total=len(dataset)):
            prompt = item['prompt']
            response = generate(model, tokenizer, item['prompt'])
            
            data = {
                        'prompt': prompt,
                        'response': response
                    }
            
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


    