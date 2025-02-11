from typing import List, Dict

from PIL import Image
import torch
from einops import repeat

from open_flamingo.eval.eval_model import BaseEvalModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype

from transformers import Idefics2Processor, Idefics2ForConditionalGeneration, BitsAndBytesConfig


# HuggingFaceM4/idefics2-8b
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
        print(f"Using device: {self.device}")
        # self.device = "cuda:0"

        self.processor = Idefics2Processor.from_pretrained(model_args["model_id"], torch_dtype=torch.bfloat16)
        self.processor.tokenizer.padding_side = "left"
        
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            model_args["model_id"], 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        
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
        model_inputs = self.processor(
            text=batch_text, 
            images=batch_images, 
            return_tensors="pt", 
            padding="longest",
            return_for_text_completion=True,
        ).to(self.device, torch.bfloat16)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = unwrap_model(self.model).generate(
                **model_inputs,
                min_new_tokens=min_generation_length,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
            # When the model generates a response, it appends the generated tokens to this input sequence.
            outputs = outputs[:, input_len:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)

        print(output_text)
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

    def get_vqa_prompt(self, question, answer=None, if_apply_chat_template=True):
        if if_apply_chat_template:  # similarity search
            messages = [
                {   
                    "role": "user",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                }
            ]
        else:  # icl
            messages = []

        if answer is None:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True).strip()
        else:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False).strip()
        # add_generation_prompt: https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts
        if if_apply_chat_template:  # similarity search
            return text
        return messages  # icl
    
    def get_vqa_prompt_icl(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt).strip()


if __name__ == "__main__":
    import requests

    # Example usage:
    model_id = "HuggingFaceM4/idefics2-8b"
    device = 0

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)

    samples = {
        "image_raw": [[image], [image], [image], [image]],
        "text_input_raw": ["What is this?", "Is there a car?", "What color is the car?", "What is the brand of the car?"],
    }

    with torch.inference_mode():
        model = EvalModel({"model_id": model_id, "device": device})
        # print the gpu memory usage
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        
        context_text = [
            {   
                "role": "user",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }
        ]
        batch_text = []
        for i in range(len(samples["text_input_raw"])):
            b = model.get_vqa_prompt_icl(
                    context_text + model.get_vqa_prompt(question=samples["text_input_raw"][i], if_apply_chat_template=False),
                    add_generation_prompt=True,
                )
            batch_text.append(b)
            a = model.get_vqa_prompt(question=samples["text_input_raw"][i], if_apply_chat_template=True)
            assert a == b

        print(batch_text)
        output = model.get_outputs(batch_text=batch_text, batch_images=samples["image_raw"], min_generation_length=0, max_generation_length=5, num_beams=3, length_penalty=0.0)
        print(output)