#!/bin/bash

# wandb login 
# export 'WANDB_ENTITY='
# export 'WANDB_PROJECT='
export 'WANDB_SILENT=true'
export 'WANDB_DISABLED=true'

stages=5
length=$((50000 / stages))

# rm -r generated/zephyr-iter1-sec1

echo "step 1: generate for data from 0 to ${length}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/generate_split_data.py \
    --model outputs/zephyr-7b-sec5-iter1\
    --batch_size 2 \
    --output_dir generated/zephyr-iter1-sec1\
    --begin_index 0 \
    --end_index ${length}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/generate_split_data.py \
    --model outputs/zephyr-7b-sec5-iter1\
    --batch_size 2 \
    --output_dir generated/zephyr-iter1-sec1 \
    --split 'test' 

export 'WANDB_NAME=spin_stagewise_sec1'
echo "step 2: spin with data from 0 to ${length}"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/deepspeed_zero3.yaml \
    --main_process_port 29515 \
    --num_processes 4 \
    src/run_IRFT.py configs/configs_stagewise_7b/config_sec1.yaml \
    --model_name_or_path=outputs/zephyr-7b-sec5-iter1 \
    --learning_rate=5.0e-7 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --use_peft=true \
    --lora_r=64 \
    --lora_alpha=16

for ((i=1; i<stages; i++))
do
    k=$((i+1))
    start_index=$((length * i))
    end_index=$((length * k))

    python src/merging_peft.py \
        --output_dir=outputs/zephyr-7b-sec${i}

    # rm -r generated/zephyr-iter1-sec${k}

    echo "step 1: generate for data from ${start_index} to ${end_index}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/generate_split_data.py \
        --model outputs/zephyr-7b-sec${i}\
        --batch_size 2 \
        --output_dir generated/zephyr-iter1-sec${k}\
        --begin_index ${start_index} \
        --end_index ${end_index}

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/generate_split_data.py \
        --model outputs/zephyr-7b-sec${i}\
        --batch_size 2 \
        --output_dir generated/zephyr-iter1-sec${k} \
        --split 'test'

    echo "step 2: spin with data from ${start_index} to ${end_index}"
    export 'WANDB_NAME=spin_stagewise_sec${k}'
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/deepspeed_zero3.yaml \
        --main_process_port 29515 \
        --num_processes 4 \
        src/run_IRFT.py configs/configs_stagewise_7b/config_sec${k}.yaml \
        --learning_rate=5.0e-7 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=1 \
        --use_peft=true \
        --lora_r=64 \
        --lora_alpha=16

    echo "step 2: spin with data from ${start_index} to ${end_index} finished!"
    # rm -r outputs/zephyr-7b-sec${i}
done

exit