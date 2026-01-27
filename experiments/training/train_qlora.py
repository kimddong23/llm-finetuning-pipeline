#!/usr/bin/env python3
"""
QLoRA 파인튜닝 스크립트

Mac M3 Pro 및 CUDA GPU 모두 지원하는 크로스 플랫폼 학습 스크립트
"""
import argparse
import logging
import os
import platform
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device():
    """최적 디바이스 자동 감지

    Returns:
        (device, use_4bit) 튜플
    """
    if torch.cuda.is_available():
        device = "cuda"
        use_4bit = True
        logger.info(f"디바이스: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        use_4bit = False  # MPS는 4-bit 미지원
        logger.info(f"디바이스: MPS (Apple Silicon)")
    else:
        device = "cpu"
        use_4bit = False
        logger.info("디바이스: CPU")

    return device, use_4bit


def load_model_and_tokenizer(model_name: str, device: str, use_4bit: bool):
    """모델과 토크나이저 로드

    Args:
        model_name: 베이스 모델 이름
        device: 디바이스
        use_4bit: 4-bit 양자화 사용 여부

    Returns:
        (model, tokenizer) 튜플
    """
    logger.info(f"모델 로드 중: {model_name}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    if use_4bit:
        # CUDA: 4-bit 양자화
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # MPS/CPU: FP16 또는 FP32
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if device != "cpu":
            model = model.to(device)

        model = prepare_model_for_kbit_training(model)

    logger.info(f"모델 로드 완료")
    return model, tokenizer


def setup_lora(model, lora_rank: int = 16, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """LoRA 설정 적용

    Args:
        model: 베이스 모델
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        LoRA가 적용된 모델
    """
    logger.info(f"LoRA 설정 (rank={lora_rank}, alpha={lora_alpha})")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(train_file: str, eval_file: str, tokenizer, max_length: int = 2048):
    """데이터셋 준비

    Args:
        train_file: 학습 데이터 파일
        eval_file: 평가 데이터 파일
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이

    Returns:
        (train_dataset, eval_dataset) 튜플
    """
    logger.info("데이터셋 로드 중...")

    # JSONL 파일 로드
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "eval": eval_file}
    )

    logger.info(f"Train: {len(dataset['train'])} samples")
    logger.info(f"Eval: {len(dataset['eval'])} samples")

    def preprocess_function(examples):
        """Chat 템플릿 적용 및 토크나이징"""
        texts = []
        for messages in examples["messages"]:
            # Chat 템플릿 적용
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Fallback: 간단한 포맷
                user_msg = messages[0]["content"]
                assistant_msg = messages[1]["content"]
                text = f"### 질문:\n{user_msg}\n\n### 답변:\n{assistant_msg}"

            texts.append(text)

        # 토크나이징
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # labels 설정 (input_ids와 동일)
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    logger.info("데이터 전처리 중...")
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    eval_dataset = dataset["eval"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["eval"].column_names
    )

    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="QLoRA 파인튜닝")
    parser.add_argument(
        "--model-name",
        type=str,
        default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        help="베이스 모델 이름"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/train.jsonl",
        help="학습 데이터 파일"
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/processed/eval.jsonl",
        help="평가 데이터 파일"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/models/qlora-exaone-2.4b",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="배치 크기"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="학습률"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="에폭 수"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="저장 간격"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="로깅 간격"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Weights & Biases 사용"
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스 감지
    device, use_4bit = get_device()

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device,
        use_4bit
    )

    # LoRA 적용
    model = setup_lora(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    # 데이터셋 준비
    train_dataset, eval_dataset = prepare_dataset(
        args.train_file,
        args.eval_file,
        tokenizer,
        args.max_length
    )

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=device == "cuda",  # CUDA만 FP16
        report_to="wandb" if args.wandb else "none",
        run_name=f"qlora-{args.model_name.split('/')[-1]}",
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 학습 시작
    logger.info("=" * 60)
    logger.info("학습 시작")
    logger.info("=" * 60)

    trainer.train()

    # 모델 저장
    logger.info(f"모델 저장 중: {output_dir}")
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")

    logger.info("학습 완료!")


if __name__ == "__main__":
    main()
