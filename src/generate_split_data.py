# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
from datasets import Dataset, load_dataset

# from modeling_mistral import (
#     MistralForCausalLM,
#     MistralConfig
# )
# AutoConfig.register("mistral", MistralConfig)
# AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--begin_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=50000)
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
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

def get_arc(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """
    The dataset is converted to a dictionary with the following structure:
    {
        "real": [{"role": "user", "content": List[str]}, 
                {"role": "assistant", "content": List[str]}], 
        "generated": [{"role": "user", "content": List[str]}, 
                    {"role": "assistant", "content": List[str]}]
    }

    Prompts should be structured as follows:
      
    """
    dataset = load_dataset('allenai/ai2_arc', "ARC-Easy", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = "Pick the most correct option to answer the following question.\n"+ sample["question"] + "\n" +\
                 "options: " + ", ".join([l + ": " + t for t, l in zip(sample["choices"]["text"], sample["choices"]["label"])]) +\
                 "\nAnswer: "
        d = {
            "real": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["answerKey"]}], 
            # "generated": [{"role": "user", "content": prompt}, 
            #             {"role": "assistant", "content": sample["option2"]}]
        }
        return d

    return dataset.map(split_prompt_and_responses)

def get_wino(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load the SPIN dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        "real": [{"role": "user", "content": List[str]}, 
                {"role": "assistant", "content": List[str]}], 
        "generated": [{"role": "user", "content": List[str]}, 
                    {"role": "assistant", "content": List[str]}]
    }

    Prompts should be structured as follows:
      
    """
    dataset = load_dataset('winogrande', "winogrande_debiased", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = sample["sentence"] + " In the previous sentence, does _ refer to " + sample["option1"] + " or " + sample["option2"] + "?\nAnswer: "
        if sample["answer"] == "1":
            d = {
            "real": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["option1"]}], 
            # "generated": [{"role": "user", "content": prompt}, 
            #             {"role": "assistant", "content": sample["option2"]}]
            }
        else:
            d = {
            "real": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["option2"]}], 
            # "generated": [{"role": "user", "content": prompt}, 
            #             {"role": "assistant", "content": sample["option1"]}]
            }
        return d

    return dataset.map(split_prompt_and_responses)

def extract_anthropic_hh_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant: "
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    # return prompt_and_response[len(prefix_term): search_term_idx]
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
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        print("Performing sanity check, i.e. take only the first 1000 rows")
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    print(f"preprocessing Anthropic HH dataset, length {len(dataset)}, example row: {dataset[0]}")

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_hh_prompt(sample["chosen"])
        return {
        "real": [{"role": "user", "content": prompt}, 
                {"role": "assistant", "content": sample["chosen"][len(prompt):]}], 
        "generated": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["rejected"][len(prompt):]}]
        }

    return dataset.map(split_prompt_and_responses)

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # debugging
    print(model, "\n", tokenizer)

    # load data
    if args.input_dir == 'winogrande':
        data = get_wino(args.split)
    elif args.input_dir == 'Anthropic/hh-rlhf':
        data = get_hh(args.split)
    elif args.input_dir == 'arc':
        data = get_arc(args.split)
    else:
        data = load_dataset(args.input_dir, split=args.split)

    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    else:
        data = data[:]['real']
        
    data = data[args.begin_index: args.end_index]
    print(f"load data with length {len(data)}, index from {args.begin_index} to {args.end_index}")
    print(f"example data: {data[0]}")

    if args.input_dir == 'UCLA-AGI/SPIN_iter0':
        # data = data[6300:] # the first 6000 is fine
        prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
        prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
        corrects_all = [data[idx][1]['content'] for idx in range(len(data))]
    elif args.input_dir == 'Anthropic/hh-rlhf':
        prefix_term = "\n\nHuman: "
        search_term = "\n\nAssistant: "
        prompts_all = [data[idx][0]['content'] for idx in range(len(data))]
        prompts_old = [prompt[len(prefix_term):-len(search_term)] for prompt in prompts_all]
        prompts_all = prompts_old
        corrects_all = [data[idx][1]['content'] for idx in range(len(data))]
    elif args.input_dir == 'winogrande' or args.input_dir == 'arc':
        prompts_all = [data[idx][0]['content'] for idx in range(len(data))]
        prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
        corrects_all = [data[idx][1]['content'] for idx in range(len(data))]
    else:
        raise ValueError(f"Need to config the schema for the dataset {args.input_dir}")

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)

        for idx, prompts_tokenized in enumerate(tqdm(prompt_batches)):
            # set max_new_tokens smaller for faster inference

            # outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
            # try:
            outputs_tokenized=model.generate(input_ids=prompts_tokenized['input_ids'], attention_mask=prompts_tokenized['attention_mask'], \
                                            max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            # except:
            #     print(f"batch {idx} contains unrecognizable tokens")
            #     # accelerator.wait_for_everyone()
            #     continue

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized)
            results.extend(outputs)
            # print(f"prompts_tokenized keys: {prompts_tokenized.keys()}")
            # print(f"prompts_tokenized['input_ids']: {prompts_tokenized['input_ids']}, shape: {prompts_tokenized['input_ids'].shape}")
            # print(f"prompts_tokenized['attention_mask']: {prompts_tokenized['attention_mask']}, shape: {prompts_tokenized['attention_mask'].shape}")
            # print(f"outputs_tokenized: {outputs_tokenized}, shape: {len(outputs_tokenized), outputs_tokenized[0].shape}")
            # quit()

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object(results)
    results = [r.replace("</s>","").lstrip() for r in results_gathered]
    print(f"len of results: {len(results)}")
    # print(f"results: {results}")
    
    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
            filename = f"{args.output_dir}/{args.split}.jsonl"
            with open(filename, 'a') as f:
                json.dump(d, f)
                f.write('\n')


if __name__ == "__main__":
    main()