from transformers import AutoModelForCausalLM
from safetensors.torch import load_file
from peft import AutoPeftModelForCausalLM
import os
import glob
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()

def merging(output_dir=None):
    """
    base_model: 'alignment-handbook/zephyr-7b-sft-full'
    peft_safetensors: 'outputs/zephyr-7b-sec1/adapter_model.safetensors'
    output_dir: 'outputs/zephyr-7b-sec1'
    """
    if output_dir==None:
        raise ValueError("Must provide a output dir")
    print(f"Merging the model by hand... output in {output_dir}")
        
    peft_model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
    merged_model = peft_model.merge_and_unload()
        
    # tensors = load_file(peft_safetensors)
    # for name, param in tensors.items():
    #     print(f"saved tensor name: {name}, Shape: {param.shape}")
        
    # model=AutoModelForCausalLM.from_pretrained(base_model)
    # for name, param in merged_model.state_dict().items():
    #     print(f"model tensor name: {name}, Shape: {param.shape}")
    
    # saving this merged model
    # glob.glob()
    files = ["/".join([output_dir, 'adapter_model.safetensors']), "/".join([output_dir, 'adapter_config.json'])]
    for f in files:
        os.remove(f)
    merged_model.save_pretrained(output_dir)
    print(f"Successfully saved to {output_dir}!")
    
if __name__ == "__main__":
    args = parse_arguments()
    merging(args.output_dir)