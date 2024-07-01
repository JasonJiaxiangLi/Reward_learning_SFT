import json
import math
import os
import sys

import numpy as np
import torch
import tritonclient.grpc as client_util
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer,HfArgumentParser
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SPPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import random

LOSS = "square" # "square" or "log", square for APA and log for AWR
ADV_COEFF_SQ = 10
ADV_COEFF_LOG = 0.5

@dataclass
class ScriptArguments:
	"""
	The name of the Casual LM model we wish to fine with PPO
	"""
	output_dir: str
	reward_model_path: str
	ref_model_path: str
	policy_model_path: str = field(default=None)
	#data_path: str = field(default="/home/zeyi/irl_sentiment/data/preference_and_demon_toy.json")
	name : Optional[str] = field(default='AIHF', metadata={"help": "use 'wandb' to log with wandb"})
	step : Optional[int] = field(default=0, metadata={"help": "the step number"})
	max_step : Optional[int] = field(default=3000, metadata={"help" : "the max step number"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

RANDOM_SEED = random.randint(0,1000)
MODEL_SIZE = "1B"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) 
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1000,
        epochs=10000,
        total_steps=5000,
        batch_size=4,
        checkpoint_interval=1000,
        eval_interval=10020,
        pipeline="PromptPipeline",
        trainer="AccelerateSPPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
        seed=RANDOM_SEED,
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1e-6)),
    method=SPPOConfig(
        name="SPPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=100,
        cliprange_value=100,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        loss_str=LOSS,
        adv_coeff_sq=ADV_COEFF_SQ,
        adv_coeff_log=ADV_COEFF_LOG,        
        cliprange_reward=100,
        gen_kwargs=dict(
            max_new_tokens=128,
            do_sample=True,
        ),
    ),
)


config_name = MODEL_SIZE 
if config_name == "125M":
    default_config.train.batch_size = 4
    default_config.method.chunk_size = 4
    default_config.train.total_steps = 2000
    default_config.model.model_path =  "EleutherAI/pythia-160m-deduped" #"Dahoas/pythia-125M-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-160m-deduped"
    default_config.method.num_rollouts = 64
elif config_name == "1B":
    default_config.train.batch_size = 2
    default_config.train.total_steps = script_args.max_step
    default_config.model.model_path = "EleutherAI/pythia-1.4b" # "Dahoas/pythia-1B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-1.4b" #"Dahoas/pythia-1B-static-sft"
    default_config.method.chunk_size = 4
elif config_name == "6B":
    default_config.train.batch_size = 1
    default_config.train.total_steps = 1000
    #default_config.model.model_path = "Dahoas/pythia-6B-static-sft" # "databricks/dolly-v2-7b" #  
    default_config.model.model_path = "/home/zeyi/irl_sentiment_chenliang/RLHF-APA-main/checkpoints/sft_hh/best_checkpoint" 
    default_config.tokenizer.tokenizer_path =  "EleutherAI/gpt-neox-20b" # "databricks/dolly-v2-7b" #
    default_config.method.chunk_size = 1 
    OUTPUT_DIR = "output_2.8B"


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_reward_fn():  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    triton_host = os.environ.get("TRITON_HOST")

    if triton_host:
        triton_url, triton_model = triton_host.split("/")
        client = client_util.InferenceServerClient(url=triton_url, verbose=False)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, max_length=1024)

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

                result = client.infer(triton_model, [prepare_tensor("input_ids", input_ids)])
                rewards = result.as_numpy("rewards")
                out.extend(rewards)

            return out

    elif os.environ.get("RANK", "0") == "0":

        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path, eos_token_id):
                super().__init__()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                self.transformer = model.gpt_neox
                self.config = model.config
                self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
                self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
                self.eos_token_id = eos_token_id

            def forward(self, input_ids):
                states = self.transformer(input_ids)[0]
                rewards = self.v_head(states).squeeze(-1)
                ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
                returns = torch.gather(rewards, 1, ends).squeeze(-1)
                return returns

        reward_model = RewardModel("EleutherAI/pythia-1.4b", reward_tokenizer.eos_token_id)
        directory = script_args.reward_model_path
        checkpoint = None
        for fpath1 in os.listdir(directory):
            if fpath1.startswith("checkpoint"):
                checkpoint = os.path.join(directory, fpath1)
        for fpath in os.listdir(checkpoint):
            if fpath.endswith("model.bin"):
                checkpoint = os.path.join(checkpoint, fpath)
                break
        print(checkpoint)
        #checkpoint='/home/zeyi/irl_sentiment_chenliang/HH_eval/RLHF-APA/IRL-HF/reward_checkpoint/IRLHF_2.8B/step0/checkpoint-100/'
        #checkpoint='/home/ubuntu/RLHF-APA/reward_model/checkpoint-500'
        
        reward_model.load_state_dict(torch.load(checkpoint))
        reward_model.eval()
        reward_model.requires_grad_(False)
        device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(device)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(
                device
            )

            mbs = 6
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = reward_model(input_ids)
                out.extend(rewards)

            return out

    else:
        reward_fn = True

    return reward_fn


if __name__ == "__main__":
    hparams = {}
    output_dir = script_args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = output_dir + '/step'+ str(script_args.step)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    config = TRLConfig.update(default_config, hparams)
    config.train.checkpoint_dir = output_dir
    config.train.logging_dir = output_dir
    config.train.tracker = None
    config.ref_model_path = script_args.ref_model_path
    # config.train.epochs = 1
    config.policy_model_path = script_args.policy_model_path
    
    print('-----------------------')
    print(config.policy_model_path)
    print('-----------------------')

    # dataset = load_dataset("openbmb/UltraFeedback")
    # prompts = dataset["train"]["instruction"][20000:30000]
    # eval_prompts = dataset["train"]["instruction"][-20:]
    dataset= load_dataset("json", data_files = "./data/Self_play_train.json")
    eval_prompts = load_dataset("json", data_files = "./data/Self_play_train.json")
    # dataset = load_dataset("Dahoas/rm-static")
    prompts = dataset["train"]["prompts"]
    eval_prompts = dataset["train"]["prompts"][-30:]
    reward_fn = create_reward_fn()

    # dataset = load_dataset("Dahoas/rm-static")
    # prompts = dataset["train"]["prompt"][:3000]
    # eval_prompts = dataset["test"]["prompt"][:280]
    # reward_fn = create_reward_fn()
    print(len(prompts))
    print(len(eval_prompts))
    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )
