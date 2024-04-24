import os
import fire
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(
    model_path,  # path to the fine-tuned adapter
    output_path="merged-model",  # path to save the output
    base_model_id="defog/sqlcoder-7b-2",  # huggingface model id
    num_gpus=1,  # number of GPUs to use
    seed=42,  # seed value for reproducibility
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
    torch.manual_seed(seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, add_bos_token=True, trust_remote_code=True
    )

    ft_model = PeftModel.from_pretrained(base_model, model_path)

    ft_model = ft_model.merge_and_unload()
    ft_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)
