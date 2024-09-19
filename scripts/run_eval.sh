# source activate spin
# wandb login $WANDB_KEYS
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

nvidia-smi
which python3
echo "Job is starting on $(hostname)"

accelerate launch --config_file configs/deepspeed_zero3.yaml \
    --main_process_port 29599 \
    --num_processes 1 \
    spin/run_spin.py configs/config_7b.yaml \
    --model_name_or_path='HuggingFaceH4/zephyr-7b-beta'\
    --output_dir='outputs/zephyr-7b-spin-test/' \
    --per_device_train_batch_size=16

echo "----EVALUATE ARC DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks ai2_arc  \
#     --num_fewshot 25  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks ai2_arc  \
    --num_fewshot 25  \
    --batch_size 8

# echo "----Evaluate the zephyr 7b model with new spin----"
# lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks ai2_arc  \
#     --num_fewshot 25  \
#     --device cuda:0 \
#     --batch_size 8

echo "----EVALUATE TRUTHFULQA DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks truthfulqa  \
#     --num_fewshot 0  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks truthfulqa  \
    --num_fewshot 0  \
    --batch_size 8

# echo "----Evaluate the zephyr 7b model with new spin----"
# lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks truthfulqa  \
#     --num_fewshot 0  \
#     --device cuda:0 \
#     --batch_size 8

echo "----EVALUATE WINOGRANDE DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks winogrande  \
#     --num_fewshot 5  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks winogrande  \
    --num_fewshot 5  \
    --batch_size 8

# echo "----Evaluate the zephyr 7b model with new spin----"
# lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks winogrande  \
#     --num_fewshot 5  \
#     --device cuda:0 \
#     --batch_size 8


echo "----EVALUATE GSM8K DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks gsm8k  \
#     --num_fewshot 5  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks gsm8k  \
    --num_fewshot 5  \
    --batch_size 8

# echo "----Evaluate the zephyr 7b model with new spin----"
# accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks gsm8k  \
#     --num_fewshot 5  \
#     --device cuda:0 \
#     --batch_size 8

echo "----EVALUATE HELLASWAG DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks hellaswag  \
#     --num_fewshot 10  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks hellaswag  \
    --num_fewshot 10  \
    --batch_size 8

# echo "----Evaluate the zephyr 7b model with new spin----"
# lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks hellaswag  \
#     --num_fewshot 10  \
#     --device cuda:0 \
#     --batch_size 8

echo "----EVALUATE MMLU DATASET----"
# echo "----Evaluate the original zephyr 7b model----"
# lm_eval --model hf \
#     --model_args pretrained=alignment-handbook/zephyr-7b-sft-full \
#     --tasks mmlu  \
#     --num_fewshot 1  \
#     --device cuda:0 \
#     --batch_size 8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
    --tasks mmlu  \
    --num_fewshot 1  \
    --batch_size 2

# echo "----Evaluate the zephyr 7b model with new spin----"
# lm_eval --model hf \
#     --model_args pretrained=outputs/zephyr-7b-spin-new/ \
#     --tasks mmlu  \
#     --num_fewshot 1  \
#     --device cuda:0 \
#     --batch_size 8


# accelerate launch -m lm_eval --model hf \
#     --tasks lambada_openai,arc_easy \
#     --batch_size 16

echo "------------Evaluation Finished------------" 

exit