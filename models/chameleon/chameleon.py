from typing import List, Dict

from PIL import Image
import torch
from einops import repeat

from open_flamingo.eval.eval_model import BaseEvalModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import AutoProcessor, AutoModelForImageTextToText


class EvalModel(BaseEvalModel):
    """Chameleon model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        assert (
            "model_id" in model_args
        ), "model_args must contain a 'model_id' key with the model's ID"

        self.device = (
            model_args["device"]
            if ("device" in model_args and model_args["device"] >= 0)
            else "cpu"
        )
        # self.device = "cuda:0"

        self.processor = AutoProcessor.from_pretrained(model_args["model_id"], torch_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_args["model_id"], torch_dtype=torch.bfloat16)
        
        self.model.to(self.device)
        self.model.eval()
    
    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:

        # with self.maybe_autocast():
        model_inputs = self.processor(text=batch_text, images=batch_images, return_tensors="pt", padding="longest").to(self.device, torch.bfloat16)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                min_new_tokens=min_generation_length,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
            # When the model generates a response, it appends the generated tokens to this input sequence.
            outputs = outputs[:, input_len:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def __call__(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """
        model_inputs = self.processor(text=batch_text, images=batch_images, return_tensors="pt", padding="longest").to(self.device, torch.bfloat16)
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        return outputs

    def get_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def get_hateful_memes_prompt(self, text, label=None) -> str:
        return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"


if __name__ == "__main__":
    import requests

    # Example usage:
    model_id = "facebook/chameleon-7b"
    device = "cuda:0"

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)

    samples = {
        "image_raw": [[image], [image], [image], [image]],
        "text_input_raw": ["What is this?", "Is there a car?", "What color is the car?", "What is the brand of the car?"],
    }

    with torch.inference_mode():
        model = EvalModel({"model_id": model_id})
        samples["text_input_raw"] = [model.get_vqa_prompt(question=p) for p in samples["text_input_raw"]]
        print(samples["text_input_raw"])
        output = model.get_outputs(batch_text=samples["text_input_raw"], batch_images=samples["image_raw"], min_generation_length=0, max_generation_length=10, num_beams=3, length_penalty=0.0)
        print(output)