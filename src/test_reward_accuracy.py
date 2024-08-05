# reference: https://github.com/uclaml/SPIN/blob/main/spin/generate.py
import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl.trainer.utils import disable_dropout_in_model, pad_to_length
from trl.models import PreTrainedModelWrapper
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import deepspeed
from copy import deepcopy
import torch
import torch.nn as nn
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

label_pad_token_id = -100
padding_value = 0

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    # parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default='HuggingFaceH4/ultrafeedback_binarized')
    parser.add_argument('--split', type=str, default='test_prefs')
    parser.add_argument('--begin_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=256)
    return parser.parse_args()

def prepare_prompts(all_prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    chosen_prompts = [all_prompts[i]["chosen"] for i in range(len(all_prompts))]
    rejected_prompts = [all_prompts[i]["rejected"] for i in range(len(all_prompts))]
    prompts_only = [all_prompts[i]["prompt"] for i in range(len(all_prompts))]
    
    def prepare_one_prompt(prompts):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]
        tokenizer.padding_side="left"     
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False, 
                    pad_to_multiple_of=8,
                    add_special_tokens=False).to("cuda") 
                )
        tokenizer.padding_side="right"
        return batches_tok
    
    return prepare_one_prompt(chosen_prompts), prepare_one_prompt(rejected_prompts), prepare_one_prompt(prompts_only)
    
def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the real and generated inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'åreal_input_ids' and 'generated_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    max_length = max(batch["real_input_ids"].shape[1], batch["generated_input_ids"].shape[1])

    for k in batch:
        if k.startswith("real") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("real", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("generated") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("generated", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(accelerator.device)

    # if self.is_encoder_decoder:
    #     concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
    #     concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

    return concatenated_batch

def _get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # if not self.is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

def concatenated_forward(
        model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the real and generated inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        len_real = batch["real_labels"].shape[0]

        model_kwargs = (
            # {
            #     "labels": concatenated_batch["concatenated_labels"],
            #     "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            # }
            # if self.is_encoder_decoder
            # else {}
            {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        real_logps = all_logps[:len_real]
        generated_logps = all_logps[len_real:]

        real_logits = all_logits[:len_real]
        generated_logits = all_logits[len_real:]

        return (real_logps, generated_logps, real_logits, generated_logits)
    
def _prepare_deepspeed(model: PreTrainedModelWrapper):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    print(f"deepspeed config_kwargs: {config_kwargs}")
    # exit()
    config_kwargs['train_micro_batch_size_per_gpu'] = 1
    config_kwargs['gradient_accumulation_steps'] = 1
    config_kwargs['train_batch_size'] = 2
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
            
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    
    return dataset.map(split_prompt_and_responses)

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    batch_size = args.batch_size
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    # model = _prepare_deepspeed(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    if args.input_dir == 'Anthropic/hh-rlhf':
        data = get_hh(split=args.split)
        data = data.shuffle(seed=42)
        prompts_all = [
            {
                "chosen": data['chosen'][idx],
                "rejected": data['rejected'][idx],
                "prompt": data['prompt'][idx]
            } 
            for idx in range(len(data['prompt']))
        ]
        # prompts_old = [data[idx]['chosen'][1]["content"] for idx in range(args.end_index)]
        # corrects_all = [data['chosen'][idx][1]["content"] for idx in range(args.end_index)]
        # rejected_all = [data['rejected'][idx][1]["content"] for idx in range(args.end_index)]

    else:
        data = load_dataset(args.input_dir, split=args.split)
        data = data.shuffle(seed=42)
        prompts_all = [
            {
                "chosen": data['chosen'][idx][1]["content"],
                "rejected": data['rejected'][idx][1]["content"],
                "prompt": "### Instruction: " + data['prompt'][idx] + "\n\n### Response: "
            } 
            for idx in range(len(data['prompt']))
        ]

    del data
    # if args.frac_len > 0:
    #     sub_len = args.frac_len 
    #     if sub_len*(data_frac+1) > len(data):
    #         data = data[sub_len*data_frac:]['real']
    #     else:
    #         data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    # else:
    #     data = data[:]['real']
    # print(data.info, data[0])
    # exit()
    
    # if args.end_index == None:
    #     args.end_index = len(data['prompt'])
    # data = data[args.begin_index: args.end_index]
    
    # print(f"len(data): {len(data)}\n")
    # print(f"data: {data}")
    # exit()

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        # print(f"prompts: {prompts}")
        # exit()
        chosen_prompts, rejected_prompts, prompts_only = prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)
        # print(f"prompts: {prompts}")
        # exit()
        # for prompts_tokenized in tqdm(prompt_batches):
        for i in tqdm(range(len(prompts))):
            # print(f"prompts_only[i]['input_ids']: {prompts_only[i]['input_ids'].shape}, chosen_prompts[i]['input_ids']: {chosen_prompts[i]['input_ids'].shape}")
            # exit()
            prompts_tokenized = { # [:, :args.max_tokens]
                'real_input_ids': torch.cat(
                    (prompts_only[i]['input_ids'].type(torch.int), chosen_prompts[i]['input_ids'].type(torch.int)), dim=1,
                ).to(accelerator.device), 
                'generated_input_ids': torch.cat(
                    (prompts_only[i]['input_ids'].type(torch.int), rejected_prompts[i]['input_ids'].type(torch.int)), dim=1,
                ).to(accelerator.device),
                'real_attention_mask': torch.cat(
                    (prompts_only[i]['attention_mask'], chosen_prompts[i]['attention_mask']), dim=1,
                ).to(accelerator.device),
                'generated_attention_mask': torch.cat(
                    (prompts_only[i]['attention_mask'], rejected_prompts[i]['attention_mask']), dim=1,
                ).to(accelerator.device),
                'real_labels': torch.cat(
                    (
                        label_pad_token_id * torch.ones_like(prompts_only[i]['input_ids'], dtype=torch.int),
                        chosen_prompts[i]['attention_mask'],
                    ),
                    dim=1,
                ).to(accelerator.device),
                'generated_labels': torch.cat(
                    (
                        label_pad_token_id * torch.ones_like(prompts_only[i]['input_ids'], dtype=torch.int),
                        rejected_prompts[i]['attention_mask'],
                    ),
                    dim=1,
                ).to(accelerator.device)
                }
            # set max_new_tokens smaller for faster inference
            # outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

            # real_logps, generated_logps, real_logits, generated_logits = concatenated_forward(model, prompts_tokenized)
            # real_logps, generated_logps, _, _ = concatenated_forward(model, prompts_tokenized)
            with torch.no_grad():
                try:
                    all_logits = model(
                        prompts_tokenized["real_input_ids"],
                        attention_mask=prompts_tokenized["real_attention_mask"],
                    ).logits.to(torch.float32)
                    
                    real_logps = _get_batch_logps(
                        all_logits,
                        prompts_tokenized["real_labels"],
                        average_log_prob=False,
                    )
                except:
                    print(f"A error occurs, prompts_tokenized['real_input_ids']: {prompts_tokenized['real_input_ids'].shape}, prompts_tokenized['real_attention_mask']: {prompts_tokenized['real_attention_mask'].shape}")
                    # real_logps = torch.Tensor(100.0)
                    
                try:
                    all_logits = model(
                        prompts_tokenized["generated_input_ids"],
                        attention_mask=prompts_tokenized["generated_attention_mask"],
                    ).logits.to(torch.float32)
                    
                    generated_logps = _get_batch_logps(
                        all_logits,
                        prompts_tokenized["generated_labels"],
                        average_log_prob=False,
                    )
                except:
                    print(f"A error occurs, prompts_tokenized['generated_input_ids']: {prompts_tokenized['generated_input_ids'].shape}, prompts_tokenized['generated_attention_mask']: {prompts_tokenized['generated_attention_mask'].shape}")
                    # generated_logps = torch.Tensor(100.0)

            # print(f"real_logps, generated_logps: {real_logps, generated_logps}")
            # print(f"len(prompts) {len(prompts)}")
            # exit()
            results.append((real_logps.item(), generated_logps.item()))
            # remove prompt from gen. tokens
            # outputs_tokenized=[ tok_out[len(tok_in):] 
            #     for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # # decode gen. tokens 
            # outputs=tokenizer.batch_decode(outputs_tokenized)
            # results.extend(outputs)
        results = np.array(results)
        results.tofile(model_path.split("/")[-1]+"-res.npy")
        
    # # collect results from all the GPUs and remove paddings
    # results_gathered=gather_object(results)
    # results = [r.replace("</s>","").lstrip() for r in results_gathered]

    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")

    #     # collecting data
    #     for idx in range(len(corrects_all)):
    #         d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
    #         if args.split == 'test':
    #             filename = f"{args.output_dir}/test.jsonl"
    #         elif args.split == 'train':
    #             filename = f"{args.output_dir}/train.jsonl"
    #         with open(filename, 'a') as f:
    #             json.dump(d, f)
    #             f.write('\n')


if __name__ == "__main__":
    main()