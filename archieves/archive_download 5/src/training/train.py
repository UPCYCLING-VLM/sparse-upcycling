import os
import torch
import math
from os.path import exists, join, isdir

from peft import LoraConfig, get_peft_model
import ast
from peft.tuners.lora import LoraLayer

from transformers import AutoProcessor, BitsAndBytesConfig, HfArgumentParser, Qwen2_5_VLForConditionalGeneration #Qwen2VLForConditionalGeneration,
from training.configuration_qwen2_vl_moe import Qwen2VLConfig, Qwen2VLVisionConfig
from training.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from torch import nn

from training.trainer import QwenTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
import transformers
from training.transformers_utils import (
    get_keys_to_not_convert,
    # _load_pretrained_model,
)
transformers.integrations.get_keys_to_not_convert = get_keys_to_not_convert


local_rank = None

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})

        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")
        torch.save(moe_state, moe_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(device=device)#(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)


    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            #device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
    else:
        model_config = Qwen2VLConfig.from_pretrained(model_args.model_id)
        model_config.update({
        "pretraining_tp": 1,
        "moe_dtype": "bfloat16",
        "adapter_dim": 512,
        "topk": 2,
        "moe_scaling": 0.25,
        "num_experts": 4,
        "output_router_logits": False,
    })
        vision_config = Qwen2VLVisionConfig.from_pretrained(model_args.model_id)
        vision_config.moe_dtype = "bfloat16"
        vision_config.adapter_dim = 512
        vision_config.topk = 2
        vision_config.moe_scaling = 0.25
        vision_config.num_experts = 4

        model_config.vision_config = vision_config
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            config = model_config,
            # torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", #**bnb_model_from_pretrained_args)
            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_cpu_offload=True  # <- Offload quantized weights to CPU
        ))
    # **bnb_model_from_pretrained_args)

    model.config.use_cache = False
    # model_to_configure = model
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        # model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        # training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    lora_modules = []
    lora_modules += [
        "embed_tokens",
        "lm_head",
    ]
    lora_modules += [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_modules,
            # target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            task_type="CAUSAL_LM",
            bias='none'
        )
        # if training_args.bits == 16:
        #     if training_args.bf16:
        #         model.to(torch.bfloat16)
        #     if training_args.fp16:
        #         model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                            # The default setting is padding_side="left"
                                            # When training using the right-side padding is more efficient.
                                              padding_side="right")

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        if "adapter" in name:
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            else:
                module = module.to(torch.float32)

    for n, p in model.named_parameters():
        if "adapter_up" in n:
            nn.init.zeros_(p)
        if "adapter_down" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "router" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = True

    model.config.use_cache = False
    model.print_trainable_parameters()
    print("Expert IDs in model:")
    for n, p in model.named_parameters():
        if "expert_" in n:
            print(n, id(p))

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    rank0_print("data module prepared.\n")
    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )
    rank0_print("Trainer Intialized... \n")
    trainer.add_callback(SavePeftModelCallback)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()


    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()