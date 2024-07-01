# import torch
# from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer
# import datasets
# import json
# import jsonlines
# name="DPO"
# model_name='EleutherAI/pythia-1.4b'
# # model_name="./model_pythia/pythia14b_ultra_hh_sft"
# config = AutoConfig.from_pretrained(model_name)
# # model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8B", config = config, state_dict = state_dict['state'])
# model = GPTNeoXForCausalLM.from_pretrained(model_name)
# model = model.to("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.padding_side='left'
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# result=[]
# # eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
# eval_set = datasets.load_dataset('json', data_files="./data/ultra_hh.json", split='train')
# for example in eval_set:
#     temp_example={}
#     # print(example)
#     # tokens = tokenizer(temp_instruction, return_tensors="pt").to("cuda:0")
#     tokens = tokenizer(example["demon_prompt"], return_tensors="pt").to("cuda:0")
#     output_start_ix = len(example["demon_prompt"])
#     temp_example["demon_prompt"] = example["demon_prompt"]
#     temp_example["agent"] = model.generate(**tokens,max_new_tokens=200)
#     decoded_output = tokenizer.decode(temp_example["agent"][0],skip_special_tokens=False)
#     temp_example["agent"] = decoded_output[output_start_ix:]
#     result.append(temp_example)
#     if len(result)>=5:
#         break
# with jsonlines.open(f"data/ultrahh_output.json",'w') as writer:
#     writer.write_all(result)


import json
import math
import os
import sys

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

from dataclasses import dataclass, field
from typing import Optional
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer,HfArgumentParser
from tritonclient.utils import np_to_triton_dtype
import jsonlines
import os
os.environ["WANDB_MODE"] = "disabled"
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import random


@dataclass
class ScriptArguments:
	"""
	The name of the Casual LM model we wish to fine with PPO
	"""
	output_dir: str
	policy_model_name: str = field(default=None)
	ref_model_path: str = field(default=None)
	name : Optional[str] = field(default='rlhf', metadata={"help": "use 'wandb' to log with wandb"})
	step : Optional[int] = field(default = 0, metadata={"help": "the step number"})
	local_rank : Optional[int] = field(default = 0, metadata={"help": "the rank number"})
	max_steps : Optional[int] = field(default = 10000)
	demonstration_number: Optional[int] = field(default=500, metadata={"help": "the number of demonstration"})
	demonstration_path : Optional[str] = field(default=None)
	start_index : Optional[int] = field(default = 0)
	end_index : Optional[int] = field(default = -1)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


RANDOM_SEED = 0
MODEL_SIZE = "1B"
OUTPUT_DIR = "./output"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) 
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=20000,
        batch_size=4,
        checkpoint_interval=1000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
        seed=RANDOM_SEED,
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


config_name = MODEL_SIZE 
if config_name == "125M":
    default_config.train.batch_size = 96
    default_config.method.chunk_size = 16
    default_config.train.total_steps = 20000
    default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.num_rollouts = 128

elif config_name == "1B":
    default_config.train.batch_size = 10
    default_config.train.total_steps = 20000
    default_config.model.model_path = "EleutherAI/pythia-1b" # "Dahoas/pythia-1B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-1b" #"Dahoas/pythia-1B-static-sft"
    default_config.method.chunk_size = 4


# elif config_name == "6B":
#     default_config.train.batch_size = 5
#     default_config.train.total_steps = 20000
#     #default_config.model.model_path = "Dahoas/pythia-6B-static-sft" # "databricks/dolly-v2-7b" #  
#     default_config.model.model_path = "/home/zeyi/irl_sentiment_chenliang/RLHF-APA-main/checkpoints/sft_hh/best_checkpoint" 
#     default_config.tokenizer.tokenizer_path =  "EleutherAI/gpt-neox-20b" # "databricks/dolly-v2-7b" #
#     default_config.method.chunk_size = 1 

def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

if __name__ == "__main__":
    hparams = {} 
    output_dir = OUTPUT_DIR
    config = TRLConfig.update(default_config, hparams)
    config.model.peft_config = None
    config.train.checkpoint_dir = output_dir
    config.train.logging_dir = output_dir
    config.train.tracker = None
    config.ref_model_path = script_args.ref_model_path
    
    # dataset = load_dataset("json",data_files = "./data/ultra_hh.json",split='train')[-script_args.demonstration_number:]
    dataset = load_dataset("json",data_files = script_args.demonstration_path,split=f'train[{script_args.start_index}:{script_args.end_index}]')
    sample_size=min(len(dataset),script_args.demonstration_number)
    random_indices = random.sample(range(len(dataset)), sample_size)
    dataset = dataset.select(indices=random_indices)
    # dataset_size = len(dataset["train"])

    # 随机选择 1000 条数据的索引
    # random_indices = random.sample(range(dataset_size), script_args.demonstration_number)
    # random_data = [dataset[index] for index in random_indices]
    eval_prompts = dataset["prompts"]
    demonstration_response = dataset["chosen"]
    config.policy_model_path = script_args.policy_model_name #'/home/ubuntu/RLHF-APA/output_RLHF_125M_APA/best_checkpoint'
    samples,prompts,outputs=(trlx.train3(
        prompts=eval_prompts,
        eval_prompts=eval_prompts,
        reward_fn=None,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    ))
    #f = open('./IRL-HF/agent.json',"w")
    f=open(script_args.output_dir,'w')
    writer = jsonlines.Writer(f)
    for i in range(len(samples)):
        writer.write({"samples":samples[i],"prompts":eval_prompts[i],"agent":outputs[i], "demon":demonstration_response[i]})
    