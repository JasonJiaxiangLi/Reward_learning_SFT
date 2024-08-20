import shutil
import argparse
import json
import math
import os
import random
from pathlib import Path
from torch.utils.data import Dataset,DataLoader,random_split
from datetime import datetime
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
) 
from dataclasses import dataclass, field
from datasets import load_dataset
from reward_model import GPTRewardModel
import numpy as np
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict as ddict
import jsonlines
import logging
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb

@dataclass
class ScriptArguments:
	"""
	The name of the Casual LM model we wish to fine with PPO
	"""
	output_dir: str
	reward_model_path: str = field(default=None)
	name : Optional[str] = field(default='rlhf', metadata={"help": "use 'wandb' to log with wandb"})
	step : Optional[int] = field(default=0, metadata={"help": "the step number"})
	local_rank : Optional[int] = field(default=0, metadata={"help": "the rank number"})
	max_steps : Optional[int] = field(default = 10000)
	max_epoch : Optional[int] = field(default = 1)
	preference_weight : Optional[float] =field(default=1.00)
	demonstration_weight : Optional[float] =field(default=1.00)
	demonstration_path : Optional[str] = field(default=None)
	


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def read_fn(path):
    data_l = []
    f0 = open(path)
    reader = jsonlines.Reader(f0)
    for line in reader:
        data_l.append(line)
    return data_l

def create_expert_dataset(path, split='train'):
    dataset = load_dataset("json", data_files = path, split=split)
    pairs = []
    # random_indices = random.sample(range(len(dataset)), sample_size)
    # dataset = dataset.select(indices=random_indices)
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompts"]
        chosen_summary = sample["demon"]
        rejected_summary = sample["agent"]

        if chosen_summary == rejected_summary:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


def create_comparison_dataset(path="Dahoas/static-hh", split="train",sample_size=2500):
    dataset = load_dataset("json", data_files = path, split=split)
    sample_size=min(len(dataset),sample_size)
    random_indices = random.sample(range(len(dataset)), sample_size)
    dataset = dataset.select(indices=random_indices)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompts"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, preference_pairs, demonstration_pairs, tokenizer, max_length, preference_weight=1, demonstration_weight=1):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.weight = []

        for pair in tqdm(demonstration_pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            if chosen != rejected:
                chosen_encodings_dict = tokenizer(
                    "<|startoftext|>" + chosen + "<|endoftext|>",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_encodings_dict = tokenizer(
                    "<|startoftext|>" + rejected + "<|endoftext|>",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                if not torch.all(chosen_encodings_dict["input_ids"] == rejected_encodings_dict["input_ids"]).item(): 
                    self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                    self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                    self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                    self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
                    self.weight.append(demonstration_weight)

        if preference_weight > 0:
            for pair in tqdm(preference_pairs):
                chosen, rejected = pair["chosen"], pair["rejected"]
                if chosen != rejected:
                    chosen_encodings_dict = tokenizer(
                        "<|startoftext|>" + chosen + "<|endoftext|>",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    rejected_encodings_dict = tokenizer(
                        "<|startoftext|>" + rejected + "<|endoftext|>",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    if not torch.all(chosen_encodings_dict["input_ids"] == rejected_encodings_dict["input_ids"]).item(): 
                        self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                        self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                        self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                        self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
                        self.weight.append(preference_weight)
        

    def __len__(self):
        return len(self.chosen_input_ids)
        # return min(len(self.chosen_input_ids),len(self.E_chosen_input_ids))

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
            self.weight[idx]
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        batch["weight"] = torch.tensor([f[4] for f in data])

        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc
    print(result["accuracy"])
    return result

if __name__=='__main__':
    
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = script_args.output_dir 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = output_dir + '/step' + str(script_args.step)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    max_steps = script_args.max_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=script_args.max_epoch,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=1,
        eval_steps=1000,
        # save_steps=50,
        warmup_steps=50,#100 to 0
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-6,
        deepspeed="./configs/ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel('EleutherAI/pythia-1.4b')
    directory = script_args.reward_model_path
    if directory != None:
        checkpoint = None
        for fpath1 in os.listdir(directory):
            if fpath1.startswith("checkpoint"):
                checkpoint = os.path.join(directory, fpath1)
        for fpath in os.listdir(checkpoint):
            if fpath.endswith("model.bin"):
                checkpoint = os.path.join(checkpoint, fpath)
                break
        print('--------------------')
        print(checkpoint)
        print('-------------------------------')

        model.load_state_dict(torch.load(checkpoint))
    
    for name, param in model.named_parameters():
        if 'transformer.layers' in name :
            if int(name.split('.')[2]) < 20:
                print(name,'has locked')
                param.requires_grad = False


    max_length = 256
    # upload preference pairs
    # if script_args.preference_weight == 0:
    #     preference_dataset = None
    # else:
    preference_dataset = create_comparison_dataset("./data/HH_preference.json", split="train",sample_size=10000)
    valid_preference_dataset = create_comparison_dataset("./data/HH_preference.json", split="train[-50:]")
    
    # upload expert pairs
    demonstration_dataset = create_expert_dataset(script_args.demonstration_path, split="train")
    train_dataset = PairwiseDataset(preference_dataset,demonstration_dataset, tokenizer, max_length=max_length, preference_weight=script_args.preference_weight, demonstration_weight=script_args.demonstration_weight)
    val_dataset = PairwiseDataset(valid_preference_dataset, valid_preference_dataset, tokenizer, max_length=max_length)

    # print(len(train_dataset))
    data_collator = DataCollatorReward()
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=train_dataset,
        data_collator=data_collator,
    ).train()

    # model.save_pretrained(output_dir+"/last")