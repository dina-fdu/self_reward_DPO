accelerate launch DPO_train.py \
    --model_name_or_path="dinaaaaaa/llama2_7b_DPO_lima_rand_sel_50_preference" \
    --output_dir="dpo"\
    --max_step=400 \
    --max_length=2048 \
    --warmup_steps=10 \
    --beta=0.1 \
    --dataset_name_or_path="dinaaaaaa/lima_rand_sel_50_preference_self_reward"

