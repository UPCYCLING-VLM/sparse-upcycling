import torch
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, set_seed
from configuration_qwen2_vl_moe import Qwen2VLConfig
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def main():
    # Load and configure Qwen2-VL with MoE and LoRA settings
    model_config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model_config.update({
        "pretraining_tp": 1,
        "moe_dtype": "bfloat16",
        "lora_r": 4,
        "lora_alpha": 4,
        "adapter_dim": 512,
        "topk": 2,
        "moe_scaling": 0.25,
        "num_experts": 4,
        "output_router_logits": False,
    })

    # Load model with quantization
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        config=model_config,
        cache_dir=".",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        output_loading_info=False,
    )

    model = model.eval().cuda()

    # Load processor (tokenizer + vision preprocessor)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Sample input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    trimmed_ids = [
        output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("### Output ###")
    print(output_text[0])

if __name__ == "__main__":
    main()
