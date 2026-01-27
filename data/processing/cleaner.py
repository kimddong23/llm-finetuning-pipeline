#!/usr/bin/env python3
"""
데이터 정제 파이프라인

수집된 코드 스니펫을 정제하고 품질을 개선합니다.
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeCleaner:
    """코드 데이터 정제기"""

    def __init__(self):
        self.stats = {
            "total": 0,
            "removed_short": 0,
            "removed_long": 0,
            "removed_low_quality": 0,
            "cleaned": 0
        }

    def is_valid_code(self, code: str) -> bool:
        """유효한 코드인지 확인

        Args:
            code: 검사할 코드

        Returns:
            유효성 여부
        """
        # 너무 짧은 코드 제외
        if len(code) < 50:
            self.stats["removed_short"] += 1
            return False

        # 너무 긴 코드 제외 (2000자)
        if len(code) > 2000:
            self.stats["removed_long"] += 1
            return False

        # 기본 Python 문법 확인
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError:
            self.stats["removed_low_quality"] += 1
            return False

        # 코드 품질 휴리스틱
        # 1. 최소한의 구조 필요
        has_function = 'def ' in code or 'class ' in code
        has_logic = any(keyword in code for keyword in ['if ', 'for ', 'while ', 'return '])

        if not (has_function or has_logic):
            self.stats["removed_low_quality"] += 1
            return False

        return True

    def clean_instruction(self, instruction: str) -> str:
        """instruction 텍스트 정제

        Args:
            instruction: 원본 instruction

        Returns:
            정제된 instruction
        """
        # 여러 공백을 하나로
        instruction = re.sub(r'\s+', ' ', instruction)

        # 앞뒤 공백 제거
        instruction = instruction.strip()

        # 너무 짧으면 기본 instruction으로 대체
        if len(instruction) < 10:
            instruction = "다음 코드를 작성해주세요"

        return instruction

    def clean_input(self, input_text: str) -> str:
        """input 텍스트 정제

        Args:
            input_text: 원본 input

        Returns:
            정제된 input
        """
        # 여러 공백을 하나로
        input_text = re.sub(r'\s+', ' ', input_text)

        # 앞뒤 공백 제거
        input_text = input_text.strip()

        return input_text

    def clean_code(self, code: str) -> str:
        """코드 정제

        Args:
            code: 원본 코드

        Returns:
            정제된 코드
        """
        # 연속된 빈 줄을 최대 2개로 제한
        code = re.sub(r'\n{3,}', '\n\n', code)

        # 줄 끝의 공백 제거
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        code = '\n'.join(lines)

        # 앞뒤 공백 제거
        code = code.strip()

        return code

    def clean_snippet(self, snippet: Dict) -> Dict:
        """단일 스니펫 정제

        Args:
            snippet: 원본 스니펫

        Returns:
            정제된 스니펫
        """
        self.stats["total"] += 1

        # 코드 유효성 확인
        if not self.is_valid_code(snippet["output"]):
            return None

        # 각 필드 정제
        cleaned = {
            "instruction": self.clean_instruction(snippet.get("instruction", "")),
            "input": self.clean_input(snippet.get("input", "")),
            "output": self.clean_code(snippet["output"])
        }

        self.stats["cleaned"] += 1
        return cleaned

    def clean_dataset(self, input_path: str, output_path: str):
        """데이터셋 전체 정제

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로
        """
        logger.info(f"입력 파일: {input_path}")

        # 입력 파일 읽기
        snippets = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    snippet = json.loads(line)
                    snippets.append(snippet)
                except json.JSONDecodeError:
                    continue

        logger.info(f"총 {len(snippets)}개 스니펫 로드")

        # 정제
        cleaned_snippets = []
        for snippet in tqdm(snippets, desc="데이터 정제"):
            cleaned = self.clean_snippet(snippet)
            if cleaned:
                cleaned_snippets.append(cleaned)

        # 출력 파일 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for snippet in cleaned_snippets:
                f.write(json.dumps(snippet, ensure_ascii=False) + "\n")

        logger.info(f"출력 파일: {output_path}")
        logger.info(f"정제 완료: {len(cleaned_snippets)}개 스니펫")

        # 통계 출력
        logger.info("\n=== 정제 통계 ===")
        logger.info(f"전체: {self.stats['total']}")
        logger.info(f"정제 완료: {self.stats['cleaned']}")
        logger.info(f"제거 (너무 짧음): {self.stats['removed_short']}")
        logger.info(f"제거 (너무 긺): {self.stats['removed_long']}")
        logger.info(f"제거 (품질 낮음): {self.stats['removed_low_quality']}")
        logger.info(f"유지율: {self.stats['cleaned'] / self.stats['total'] * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="코드 데이터 정제")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 JSONL 파일"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 JSONL 파일"
    )

    args = parser.parse_args()

    cleaner = CodeCleaner()
    cleaner.clean_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
