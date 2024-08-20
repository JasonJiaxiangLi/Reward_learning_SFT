# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from typing import Dict, Optional

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import jsonlines
import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alignment-handbook/zephyr-7b-sft-full')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    parser.add_argument('--max_prompt_length', type=int, default=128)
    parser.add_argument('--max_continuation_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default="UCLA-AGI/SPIN_iter0")
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, demonstration, batch_size=4, max_prompt_length = 128):
    """Prepare prompts for tokenization."""
    batches_prompt=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_demonstration=[demonstration[i:i+ batch_size] for i in range(0, len(demonstration), batch_size)]
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches_prompt:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                max_length=max_prompt_length,
                return_tensors="pt", 
                padding='longest', 
                truncation=True).to("cuda") 
            )
    tokenizer.padding_side="right"
    # print(len(batches_prompt)==len(batches_demonstration))#true
    return batches_prompt, batches_tok, batches_demonstration


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
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    if args.input_dir == "Anthropic/hh-rlhf":
        data = get_hh(split=args.split)
    elif args.input_dir == "UCLA-AGI/SPIN_iter0":
        data = load_dataset(args.input_dir, split=args.split)
        data = data[:]['real']
        data = data[args.begin_index: args.end_index]
    else:
        data = load_dataset("json", data_files=args.input_dir, split=f'train[{args.start_index}:{args.end_index}]') 
    # load_dataset("json", data_files=args.input_dir, split="train[:280]")
    data = data.shuffle(seed=42)
    print('the data has been uploaded')
    if args.input_dir == "UCLA-AGI/SPIN_iter0":
        prompts_all = [["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: ", data[idx][1]['content']] for idx in range(len(data))]
    else:
        prompts_all=[list(pair) for pair in zip(data['prompts'], data['chosen'])]
    # print(type(prompts_all))
    # quit()
    # prompts_all=zip(data["prompts"],data["demon"])
    # prompts_old = prompts_all

    accelerator.wait_for_everyone()    
    start=time.time()
    # directory = os.path.dirname(output_dir)
    # os.makedirs(directory, exist_ok=True)
    f=open(output_dir,'w')
    writer = jsonlines.Writer(f)
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as all_data:
        # prompts,demon= zip(*zipped)
        # print(len(all_data[0]))
        # print(all_data[0])
        prompts=[i[0] for i in all_data]
        demon=[i[1] for i in all_data]
        # print(len(prompts),len(demon))
        results = {'prompt': [], 'continuation': [], "demon":[]}
        original_prompt_batches, tokenized_prompt_batches, original_demon_batches = prepare_prompts(prompts, tokenizer, demon, batch_size=args.batch_size, max_prompt_length=args.max_prompt_length)
        for original_prompts, prompts_tokenized,original_demon in tqdm(zip(original_prompt_batches, tokenized_prompt_batches, original_demon_batches), total=len(tokenized_prompt_batches)):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=args.max_continuation_length, pad_token_id=tokenizer.eos_token_id)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized)
            results['prompt'].extend(original_prompts)
            results['continuation'].extend(outputs)
            results['demon'].extend(original_demon)
        # print(len(results['continuation']))
        # print(len(results['prompt']))
        # print(len(results['demon']))
            
    # collect results from all the GPUs and remove paddings
    result_to_save = {'prompt': [], 'continuation': [], "demon":[]}
    results_gathered=gather_object([results])
    for r in results_gathered:
        for i in range(len(r['continuation'])):
            result_to_save['continuation'].append(r['continuation'][i].replace("</s>","").lstrip())
            result_to_save['prompt'].append(r['prompt'][i])
            result_to_save['demon'].append(r['demon'][i])
    

    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")
        for the_prompt, the_continuation, the_demonstration in zip(result_to_save['prompt'], result_to_save['continuation'],result_to_save['demon']):
            for stop in ["Human:", "human:", "Assistant:", "assistant:"]:
                stop_ix = the_continuation.find(stop)
                if stop_ix >= 0:
                    the_continuation = the_continuation[:stop_ix].rstrip()
            writer.write({"prompts": the_prompt, "agent": the_continuation, "demon":the_demonstration})


if __name__ == "__main__":
    main()