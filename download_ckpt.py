# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
print(checkpoint_path)
# model.load_state_dict(torch.load(checkpoint_path), strict=False)