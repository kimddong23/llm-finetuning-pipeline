#!/usr/bin/env python3
"""
학습 데이터 준비

정제된 데이터를 Chat 포맷으로 변환하고 train/eval로 분할합니다.
"""
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_chat_format(snippet: Dict) -> Dict:
    """Alpaca 포맷을 Chat 포맷으로 변환

    Args:
        snippet: Alpaca 포맷 데이터
            - instruction: 지시사항
            - input: 추가 입력 (optional)
            - output: 출력

    Returns:
        Chat 포맷 데이터
    """
    # 사용자 메시지 구성
    user_message = snippet["instruction"]
    if snippet.get("input") and snippet["input"].strip():
        user_message += f"\n\n{snippet['input']}"

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": snippet["output"]}
        ]
    }


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """데이터셋을 train/eval로 분할

    Args:
        data: 전체 데이터
        train_ratio: 학습 데이터 비율
        seed: 랜덤 시드

    Returns:
        (train_data, eval_data) 튜플
    """
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    return train_data, eval_data


def save_jsonl(data: List[Dict], output_path: str):
    """JSONL 파일로 저장

    Args:
        data: 저장할 데이터
        output_path: 출력 파일 경로
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"저장 완료: {output_path} ({len(data)}개)")


def main():
    parser = argparse.ArgumentParser(description="학습 데이터 준비")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 JSONL 파일 (정제된 데이터)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="학습 데이터 비율 (0.0 ~ 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="최대 샘플 수 (테스트용)"
    )

    args = parser.parse_args()

    # 입력 파일 읽기
    logger.info(f"입력 파일 읽는 중: {args.input}")
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                continue

    logger.info(f"총 {len(data)}개 샘플 로드")

    # 최대 샘플 수 제한 (테스트용)
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"최대 샘플 수 제한: {len(data)}개")

    # Chat 포맷으로 변환
    logger.info("Chat 포맷으로 변환 중...")
    chat_data = []
    for item in tqdm(data, desc="포맷 변환"):
        try:
            chat_item = convert_to_chat_format(item)
            chat_data.append(chat_item)
        except Exception as e:
            logger.debug(f"변환 실패: {e}")
            continue

    logger.info(f"변환 완료: {len(chat_data)}개")

    # Train/Eval 분할
    logger.info("Train/Eval 분할 중...")
    train_data, eval_data = split_dataset(
        chat_data,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    logger.info(f"Train: {len(train_data)}개")
    logger.info(f"Eval: {len(eval_data)}개")

    # 저장
    output_dir = Path(args.output_dir)
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(eval_data, output_dir / "eval.jsonl")

    # 통계 출력
    logger.info("\n=== 데이터셋 통계 ===")
    logger.info(f"전체 샘플: {len(chat_data)}")
    logger.info(f"학습 샘플: {len(train_data)} ({len(train_data)/len(chat_data)*100:.1f}%)")
    logger.info(f"평가 샘플: {len(eval_data)} ({len(eval_data)/len(chat_data)*100:.1f}%)")

    # 샘플 출력
    logger.info("\n=== 학습 데이터 샘플 ===")
    sample = train_data[0]
    logger.info(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
