#!/bin/bash

# wandb login 
# export 'WANDB_ENTITY='
# export 'WANDB_PROJECT='
export 'WANDB_SILENT=true'
export 'WANDB_DISABLED=true'

# CUDA_VISIBLE_DEVICES=5 python eval.py
set -x
set -e
epoch=5
# reward_model_name=EleutherAI/pythia-1.4b

# experiment setting
# demonstration_file=./data/Self_play_train.json
demonstration_file=UCLA-AGI/SPIN_iter0
checkpoint_path=IRL_SelfPlay
# ref_model_path=EleutherAI/pythia-1.4b
ref_model_path=alignment-handbook/zephyr-7b-sft-full
# ref_model_path=./checkpoint/policy_checkpoint/IRL_SelfPlay/step0/checkpoint_3000
max_step=2000

echo "step 1 : generate demonstration-agent pair in a json file"
WANDB_DISABLE_SERVICE=true accelerate launch --main_process_port 29082 \
    src/generate.py \
    --model ${ref_model_path} \
    --input_dir ${demonstration_file} \
    --batch_size 32 \
    --output_dir ./generated/temp_agent_demonstration.json

echo "step 2 : reward learning of data from step 1"
WANDB_DISABLE_SERVICE=true deepspeed --include localhost:0 --master_port 29052\
    src/reward_learning_AIHF.py \
    --reward_model_path ./outputs/reward_checkpoint/IRL_SelfPlay/step0/ \
    --output_dir ./outputs/reward_checkpoint/${checkpoint_path} \
    --demonstration_path ./generated/temp_agent_demonstration.json\
    --preference_weight 0 \
    --max_epoch 1 \
    --step 1

echo "step 3 : Using PPO to improve the policy"
WANDB_DISABLE_SERVICE=true accelerate launch --main_process_port 29053 \
    --config_file configs/accelerate/zero2-bf16.yaml \
    src/policy_training_selfplay.py \
    --reward_model_path ./outputs/reward_checkpoint/${checkpoint_path}/step1 \
    --ref_model_path ${ref_model_path} \
    --policy_model_path ${ref_model_path} \
    --max_step ${max_step} \
    --output_dir ./outputs/policy_checkpoint/${checkpoint_path} \
    --step 1

# echo "step 4 : iteratively do step 1,2,3"
# for ((i=1; i<epoch; i++))
# do
#     j=$((i-1))
#     k=$((i+1))
#     start_index=$((10000 * i))
#     end_index=$((10000 * k))
#     WANDB_DISABLE_SERVICE=true accelerate launch --main_process_port 29052 \
#         generate.py \
#         --model ./checkpoint/policy_checkpoint/${checkpoint_path}/step${j}/checkpoint_${max_step}  \
#         --input_dir ${demonstration_file} \
#         --start_index ${start_index} \
#         --end_index ${end_index} \
#         --batch_size 32 \
#         --output_dir ./data/temp_agent_demonstration.json

#     WANDB_DISABLE_SERVICE=true deepspeed --include localhost:0 --master_port 29051 reward_learning_AIHF.py \
#         --reward_model_path ./checkpoint/reward_checkpoint/${checkpoint_path}/step${j} \
#         --output_dir ./checkpoint/reward_checkpoint/${checkpoint_path} \
#         --preference_weight 0 \
#         --max_epoch 1 \
#         --step ${i}

#     WANDB_DISABLE_SERVICE=true accelerate launch --main_process_port 29053 \
#         --config_file configs/accelerate/zero2-bf16.yaml policy_training_selfplay.py \
#         --reward_model_path ./checkpoint/reward_checkpoint/${checkpoint_path}/step${i} \
#         --policy_model_path ./checkpoint/policy_checkpoint/${checkpoint_path}/step${j}/checkpoint_${max_step} \
#         --max_step ${max_step} \
#         --ref_model_path ${ref_model_path} \
#         --output_dir ./checkpoint/policy_checkpoint/${checkpoint_path} \
#         --step ${i}
# done

exit