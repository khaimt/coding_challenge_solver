# Leetcode challenges solver
This repo is about building LLMs to solve code challenges from [LeetCode](https://leetcode.com/)

## Training 
We use the packing implementation from: https://github.com/MeetKai/functionary/tree/main/functionary/train/packing to speed up training


```bash
python -m training.training.py \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --train_path data/algorithm_train.json \
    --validation_path data/algo_val.json \
    --model_type llama \
    --use_lora True \
    --qlora True \
    --bf16 True \
    --output_dir models/llm-algorithm \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 40 \
    --save_strategy "epoch" \
    --save_steps 80 \
    --save_total_limit 3 \
    --learning_rate 1.2e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --packing True

```