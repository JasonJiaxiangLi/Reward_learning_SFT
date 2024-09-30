# Reward-learning SFT

This is a repository containing the implementation for [Getting More Juice Out of the SFT Data: Reward Learning from Human Demonstration Improves SFT for LLM Alignment](https://arxiv.org/pdf/2405.17888), which has been accepted to NeurIPS 2024. The code is partially built upon [SPIN](https://github.com/uclaml/SPIN).

## Setup
The following steps provide the necessary setup to run our codes.
1. Create a Python virtual environment with Conda:
```
conda create -n myenv python=3.10
conda activate myenv
```
2. Install the following Python dependencies to run the codes.
```
python -m pip install .
python -m pip install flash-attn --no-build-isolation
```
3. Login to your huggingface account for downloading models
```
huggingface-cli login --token "${your_access_token}"
```

## Run Reward Learning SFT

```
bash run_RFT.sh
```

## Run Implicit Reward Learning SFT
```
bash run_IRFT.sh
```
