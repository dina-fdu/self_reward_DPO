## Implementation of DPO and the paper: Self-Rewarding Language Model (Unofficial)


------------------------------
# Step 1:
Tasks:

Generate a preference dataset. Extract the Lima dataset’s instruction. Sample 50 instructions. Then, use meta/llama-2-chat-hf to generate 5 responses for each instruction, make sure to use the appropriate chat template for llama2. Then, use PairRM to create a preference dataset. Push this dataset to huggingface.

Install environment:
```
pip install -r requirements.txt
```
```
git clone https://github.com/yuchenlin/LLM-Blender.git
cd LLM-Blender
pip install -e .
```
Then, run ```step1_enerate_preference_dataset.ipynb```

# Step 2:
Tasks:

Use DPO to fine tune meta/llama-2-chat. Then, sample 10 instructions that were not seen in training and generate samples. Compare the completions from the original model (meta/llama-2-chat) and your DPO fine tuned model. Display the instruction, original model completion, and DPO fine-tuned model completion as a pandas dataframe. Then, print out the dataframe to stdout. Push the PEFT adapter to huggingface.

Use DPO to fine tune meta/llama-2-chat:

```bash DPO_train.sh```

Evaluate the model:

```python eval.py```

Display the dataframe:

run ```step2_dpo.ipynb```

# Step 3:
Tasks:

Iterative DPO has been an intriguing development and achieves strong empirical results. One example is discussed in the paper, “Self Rewarding Language Models”. It combines the idea of LLM-as-a-Judge with DPO trained in an iterative manner.

Self Rewarding can been seen as a replace of llm-blender, so modify the step1:

run ```Iterative_DPO.ipynb``` to generate new  preference dataset

then,
```bash DPO_train_bonus.sh```to generate the Iterative DPO model

