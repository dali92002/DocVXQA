from transformers import AutoProcessor, Pix2StructForConditionalGeneration, Pix2StructProcessor
import torch
from torch import nn

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_pix2struct_model(task=None, ckpts=None):
    if task == "capt":
        processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-base")
    elif task == "vqa":
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
    if ckpts is not None:
        model.load_state_dict(torch.load(ckpts))
    return model, processor

def get_pix2struct_backbone(task=None):
    if task == "capt":
        processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-base")
    elif task == "vqa":
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
    return model, processor

