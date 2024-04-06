from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
model = AutoPeftModelForCausalLM.from_pretrained(
    "dpo/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token



all_responses = []
dataset = load_dataset("dinaaaaaa/lima_rand_sel_50_preference")
selected_instructions = []
for i in range(400, 500, 10):
    selected_instructions.append(dataset['train']['prompt'][i])
# Use the appropriate chat template for llama2
system_message = "You are a helpful assistant. Please briefly respond to the instruction."
llama2_prompt_template = lambda system_message, user_message: f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] "

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate 1 responses for each instruction-DPO fine tuned model
for text in tqdm(selected_instructions):
    inputs = tokenizer(llama2_prompt_template(system_message, text), return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device), 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=64, 
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        # temperature=1.5
        )
    
    responses = [tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in outputs]
    all_responses.append(responses)

# 5 responses for each instruction from the original model
import pandas as pd
df = pd.DataFrame(dataset['train'])
original_responses = [[] for i in range(10)]
start, end, step = 400, 500, 10
cur_chosen_rating = 5
while cur_chosen_rating > 1:
    selected_chosen = []
    for i in range(start, end + 1, step):
        slice_df = df.iloc[i:i+10]  
        match = slice_df[slice_df['chosen-rating'] == cur_chosen_rating].head(1) 
        if not match.empty:
            selected_chosen.append(match['chosen'].values[0])
    for i in range(10):
        original_responses[i].append(f"chosen-rating {cur_chosen_rating}: {selected_chosen[i]}")
    cur_chosen_rating -= 1
    selected_chosen = []
for i in range(start, end + 1, step):
    slice_df = df.iloc[i:i+10]  
    match = slice_df[slice_df['rejected-rating'] == 1].head(1) 
    if not match.empty:
        selected_chosen.append(match['rejected'].values[0])
for i in range(10):
    original_responses[i].append(f"chosen-rating 1: {selected_chosen[i]}")

# Display the instruction, original model completion, and DPO fine-tuned model completion as a pandas dataframe
data = {'instruction': selected_instructions}
for i in range(1, 6):
    data[f'original_model{i}'] = [resp[i-1] for resp in original_responses]
df = pd.DataFrame(data)
df['DPO_model'] = all_responses
# print(df)


# Print out the dataframe to stdou
from tabulate import tabulate
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, stralign="left"))

# Push the PEFT adapter to huggingface
model_name = "dinaaaaaa/llama2_7b_DPO_lima_rand_sel_50_preference"
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)
