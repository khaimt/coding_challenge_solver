python -m training.training \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --train_path data/algorithm_train.json \
    --validation_path data/algo_val.json \
    --model_type llama \
    --use_lora True \
    --qlora True \
    --bf16 False \
    --output_dir models/llm-algorithm \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --eval_steps 40 \
    --save_strategy "epoch" \
    --save_steps 80 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1300 \
    --gradient_checkpointing True \
    --packing False