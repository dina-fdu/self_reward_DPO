accelerate launch DPO_train.py \
    --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
    --output_dir="dpo"\
    --max_step=400 \
    --max_length=2048 \
    --warmup_steps=10 \
    --beta=0.1





