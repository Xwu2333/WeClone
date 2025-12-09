import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from typing import Dict, List, cast
import json
import torch
import unsloth
from unsloth import FastLanguageModel, FastModel, is_bf16_supported
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer, apply_chat_template
from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.utils.config import load_config
from weclone.utils.config_models import WCMakeDatasetConfig, WCTrainSftConfig, WCModelConfig, WCLoraConfig
from weclone.utils.log import logger


def main():
    # os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_CUDA_ARCH_LIST"] ="12.0"
    # os.environ["TORCH_COMPILE_DISABLE"] = "1"
    # os.environ["TRITON_CACHE_DIR"] = "C:\\tmp\\triton"
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\tmp\\torchinductor"

    train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type="train_sft"))
    dataset_config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))

    # Load model and LoRA configurations

    model_config: WCModelConfig = cast(WCModelConfig, load_config(arg_type="model_args"))
    lora_config: WCLoraConfig = cast(WCLoraConfig, load_config(arg_type="lora_args"))

    # device = get_current_device()
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Please note you are using CPU for training, non-Mac devices may encounter issues")

    dataset_info_path = os.path.join(train_config.dataset_dir, "dataset_info.json")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
        data_path = os.path.join(
            train_config.dataset_dir, dataset_info.get(train_config.dataset, {}).get("file_name")
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file '{data_path}' does not exist, please check if make-dataset was executed"
            )

    if not dataset_config.clean_dataset.enable_clean:
        logger.info("Data cleaning is not enabled, will use the original dataset.")
    else:
        cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
        train_config.dataset = cleaner.clean()

    # Load training dataset
    logger.info(f"Loading dataset from: {data_path}")
    train_dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Dataset loaded with {len(train_dataset)} samples")

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"Fine-tuning configuration:\n{formatted_config}")

    #run_exp(train_config.model_dump(mode="json"))
    
    # Load model and tokenizer using Unsloth
    model_args_dict = model_config.model_dump()
    # model_args_dict["dtype"] = torch.bfloat16 if train_config.fp16 and is_bf16_supported() else (torch.float16 if train_config.fp16 else None)
    model, tokenizer = FastLanguageModel.from_pretrained(**model_args_dict)
    # model, tokenizer = FastModel.from_pretrained(**model_args_dict)
    logger.info(f"Model loaded with dtype: {model.dtype}")


    trainable_params = 0
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        total_params += parameter.numel()

    logger.info(f"Total parameters in model: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    # def apply_chat_template(example):
    #     messages = example["messages"]  # Get messages from this example
    #     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #     return {"text": text}  # Return new example with text field

    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})  # Apply to each example

    logger.info(train_dataset[0])
    logger.info("Applied chat template to dataset")

    # Build SFTConfig (maps to HF TrainingArguments)
    sft_config = SFTConfig(**train_config.sft_trainer_args.model_dump())

    # Optional LoRA configuration
    if train_config.finetuning_type.value == "lora":
        # Process target_modules from string to list for FastLanguageModel.get_peft_model()
        lora_config_dict = lora_config.model_dump()
        if isinstance(lora_config_dict.get("target_modules"), str):
            lora_config_dict["target_modules"] = [m.strip() for m in lora_config_dict["target_modules"].split(",") if m.strip()]

        model = FastLanguageModel.get_peft_model(model, **lora_config_dict)
        logger.info("Applied LoRA configuration using Unsloth")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    # Train and save
    if train_config.do_train:
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        output_dir = train_config.output_path
        # trainer.save_model(os.path.join(output_dir, "saved_model_from_trainer"))
        model.save_pretrained(os.path.join(output_dir, "Adapter"))
        logger.info(f"Adapter saved to {os.path.join(output_dir, 'Adapter')}")
        model.save_pretrained_merged(os.path.join(output_dir), tokenizer, save_method = "merged_16bit")
        # model.save_pretrained_gguf(os.path.join(output_dir, "sft_model"), tokenizer, quantization_method = "f16")
        logger.info(f"{os.getcwd()}")
        # tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
