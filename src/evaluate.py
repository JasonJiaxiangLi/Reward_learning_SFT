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

from generate import get_hh
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--trun_len', type=int, default=0)
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
        prompt = sample["sentence"] + " In the previous sentence, does _ refer to " + sample["option1"] + " or " + sample["option2"] + "?"
        if sample["answer"] == "1":
            d = {
            "real": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["option1"]}], 
            "generated": [{"role": "user", "content": prompt}, 
                        {"role": "assistant", "content": sample["option2"]}]
            }
        else:
            d = {
            "real": [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": sample["option2"]}], 
            "generated": [{"role": "user", "content": prompt}, 
                        {"role": "assistant", "content": sample["option1"]}]
            }
        return d

    return dataset.map(split_prompt_and_responses)

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

    # what we don't want
    exclude_id = ["NYSEDREGENTS_2008_4_6", "WASL_2003_5_8", "TIMSS_2003_4_pg35", "NYSEDREGENTS_2013_4_26", "WASL_2005_5_10",\
                  "TIMSS_1995_8_L3", "TIMSS_1995_8_P4", "NYSEDREGENTS_2010_4_23", "NYSEDREGENTS_2010_4_15", "NYSEDREGENTS_2010_4_19",\
                  "TIMSS_2003_8_pg19"]

    # create new dataset exluding those idx
    dataset = dataset.select(
        (
            i for i in range(len(dataset)) 
            if dataset[i]["id"] not in exclude_id
        )
    )

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = "Pick the most correct option to answer the following question.\n"+ sample["question"] + "\n" +\
                 "options: " + ", ".join([l + ": " + t for t, l in zip(sample["choices"]["text"], sample["choices"]["label"])]) +\
                 "\nAnswer: "
        d = {
            "real": [{"role": "user", "content": prompt}, 
                     {"role": "assistant", "content": sample["answerKey"]}], 
        }
        i = 0
        for l in sample["choices"]["label"]:
            if l != sample["answerKey"]:
                d[str(i)] = [{"role": "user", "content": prompt}, 
                             {"role": "assistant", "content": l}]
                i += 1
        # if len(sample["choices"]["label"]) != 4:
        #     print(sample)
        return d

    return dataset.map(split_prompt_and_responses)

def evaluate_prompts(prompts, model, tokenizer, args):
    # sync GPUs and start the timer
    accelerator.wait_for_everyone() 
    print("evaluating for prompts")
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts) as prompts:
        results = []
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)

        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference

            # outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
            # outputs_tokenized=model.generate(input_ids=prompts_tokenized['input_ids'], attention_mask=prompts_tokenized['attention_mask'])

            all_logits = model(
                prompts_tokenized["input_ids"],
                attention_mask=prompts_tokenized["attention_mask"]
            ).logits.to(torch.float32)

            # input_ids = prompts_tokenized["input_ids"]
            # print(f"input_ids dim: {input_ids.shape}, logis dim: {all_logits.shape}")
            # exit()

            probs = torch.log_softmax(all_logits, dim=-1).detach()
            probs = probs[:, :-1, :]
            input_ids = prompts_tokenized["input_ids"]
            input_ids = input_ids[:, 1:]
            gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
            
            # results.extend(gen_probs[:, -1])
            results.extend(gen_probs.sum(1).tolist())
            # print(f"results: {results}")
            # exit()
        
        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in tokenizer.all_special_ids:
                    text_sequence.append((tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        print(f"batch: {batch}")
        # print(f"gen_probs: {gen_probs}")

    # collect results from all the GPUs and remove paddings
    results = gather_object(results)

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")
    return results

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    batch_size = args.batch_size

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # # debugging
    # print(model, "\n", tokenizer)

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
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    elif args.trun_len > 0:
        data = data[:args.trun_len]
    else:
        data = data[:]
    print(f"load data with length {len(data['real'])}")

    if args.input_dir == 'UCLA-AGI/SPIN_iter0':
        prompts_all = ["### Instruction: " + data['real'][idx][0]['content'] + "\n\n### Response: " for idx in range(len(data['real']))]
        # prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
        corrects_all = [prompts_all[idx] + data['real'][idx][1]['content'] for idx in range(len(data['real']))]
        incorrects_all = [prompts_all[idx] + data['generated'][idx][1]['content'] for idx in range(len(data['real']))]

        correct_results = evaluate_prompts(corrects_all, model, tokenizer, args)
        incorrect_results = evaluate_prompts(incorrects_all, model, tokenizer, args)

        # print(f"correct_results: {correct_results}")
        # print(f"incorrect_results: {incorrect_results}")
        # exit()

        print(f"Accuracy among {len(correct_results)} samples: ")
        print(sum([w >= l for w, l in zip(correct_results, incorrect_results)]) / len(correct_results))
    elif args.input_dir == 'Anthropic/hh-rlhf':
        prefix_term = "\n\nHuman: "
        search_term = "\n\nAssistant: "
        prompts_all = [data[idx]['real'][0]['content'] for idx in range(len(data))]
        # prompts_old = [prompt[len(prefix_term):-len(search_term)] for prompt in prompts_all]
        corrects_all = [prompts_all[idx] + " " + data[idx]['real'][1]['content'] for idx in range(len(data))]
        incorrects_all = [prompts_all[idx] + " " + data[idx]['generated'][1]['content'] for idx in range(len(data))]
    elif args.input_dir == 'winogrande':
        corrects_all = [data['real'][idx][0]['content'] + " " + data['real'][idx][1]['content'] for idx in range(len(data['real']))]
        incorrects_all = [data['generated'][idx][0]['content'] + " " + data['generated'][idx][1]['content'] for idx in range(len(data['real']))]
        
        correct_results = evaluate_prompts(corrects_all, model, tokenizer, args)
        incorrect_results = evaluate_prompts(incorrects_all, model, tokenizer, args)

        # print(f"correct_results: {correct_results}")
        # print(f"incorrect_results: {incorrect_results}")
        # exit()

        print(f"Accuracy among {len(correct_results)} samples: ")
        print(sum([w >= l for w, l in zip(correct_results, incorrect_results)]) / len(correct_results))
    elif args.input_dir == 'arc':
        corrects_all = [data['real'][idx][0]['content'] + " " + data['real'][idx][1]['content'] for idx in range(len(data['real']))]
        incorrects_all = {}
        for i in range(3):
            incorrects_all[i] = [data[str(i)][idx][0]['content'] + " " + data[str(i)][idx][1]['content'] for idx in range(len(data['real']))]
        
        correct_results = evaluate_prompts(corrects_all, model, tokenizer, args)
        incorrect_results = {}
        for i in range(3):
            incorrect_results[i] = evaluate_prompts(incorrects_all[i], model, tokenizer, args)
        
        cnt = 0
        for idx in range(len(data['real'])):
            flag = 1
            for i in range(3):
                if correct_results[idx] < incorrect_results[i][idx]:
                    flag = 0
            cnt += flag
        
        print(f"Accuracy among {len(correct_results)} samples: ")
        print(cnt / len(correct_results))
    else:
        raise ValueError(f"Need to config the schema for the dataset {args.input_dir}")
    
    # print(f"corrects_all: {corrects_all}")
    # exit()

    
    
if __name__ == "__main__":
    main()