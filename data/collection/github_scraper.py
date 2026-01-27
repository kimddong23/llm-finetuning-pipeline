#!/usr/bin/env python3
"""
GitHub Korean Code Scraper

한국어 주석이 포함된 Python 코드를 GitHub에서 수집합니다.
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubKoreanCodeScraper:
    """GitHub에서 한국어 주석이 있는 Python 코드를 수집하는 스크래퍼"""

    def __init__(self, token: Optional[str] = None, min_stars: int = 10):
        """
        Args:
            token: GitHub Personal Access Token (선택)
            min_stars: 최소 star 수
        """
        self.token = token
        self.min_stars = min_stars
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

        # API rate limit 체크
        self.check_rate_limit()

    def check_rate_limit(self):
        """GitHub API rate limit 확인"""
        response = requests.get(
            f"{self.base_url}/rate_limit",
            headers=self.headers
        )
        data = response.json()
        remaining = data["resources"]["core"]["remaining"]
        limit = data["resources"]["core"]["limit"]

        logger.info(f"GitHub API Rate Limit: {remaining}/{limit}")

        if remaining < 100:
            logger.warning("Rate limit이 부족합니다. Token을 사용하거나 잠시 기다려주세요.")

    def search_korean_repos(self, language: str = "python", max_repos: int = 100) -> List[Dict]:
        """한국어 README나 description이 있는 리포지토리 검색

        Args:
            language: 프로그래밍 언어
            max_repos: 최대 리포지토리 수

        Returns:
            리포지토리 정보 리스트
        """
        repos = []
        page = 1
        per_page = 30

        # 검색 쿼리: 한국어 관련 키워드
        queries = [
            f"language:{language} stars:>{self.min_stars} (한국어 OR 한글 OR korean)",
            f"language:{language} stars:>{self.min_stars} location:korea",
            f"language:{language} stars:>{self.min_stars} repo:korean",
        ]

        for query in queries:
            logger.info(f"검색 쿼리: {query}")

            while len(repos) < max_repos:
                try:
                    response = requests.get(
                        f"{self.base_url}/search/repositories",
                        headers=self.headers,
                        params={
                            "q": query,
                            "sort": "stars",
                            "order": "desc",
                            "per_page": per_page,
                            "page": page
                        }
                    )

                    if response.status_code == 403:
                        logger.warning("Rate limit 초과. 1시간 대기 필요.")
                        break

                    response.raise_for_status()
                    data = response.json()

                    if not data["items"]:
                        break

                    for repo in data["items"]:
                        if len(repos) >= max_repos:
                            break

                        # 중복 제거
                        if not any(r["full_name"] == repo["full_name"] for r in repos):
                            repos.append({
                                "full_name": repo["full_name"],
                                "description": repo.get("description", ""),
                                "stars": repo["stargazers_count"],
                                "language": repo["language"],
                                "url": repo["html_url"]
                            })

                    page += 1
                    time.sleep(2)  # API rate limit 고려

                except requests.exceptions.RequestException as e:
                    logger.error(f"리포지토리 검색 실패: {e}")
                    break

        logger.info(f"총 {len(repos)}개 리포지토리 발견")
        return repos[:max_repos]

    def has_korean(self, text: str) -> bool:
        """텍스트에 한국어가 포함되어 있는지 확인

        Args:
            text: 확인할 텍스트

        Returns:
            한국어 포함 여부
        """
        korean_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]')
        return bool(korean_pattern.search(text))

    def extract_code_snippets(self, content: str) -> List[Dict[str, str]]:
        """코드에서 한국어 주석이 있는 함수/클래스 추출

        Args:
            content: 파일 내용

        Returns:
            코드 스니펫 리스트 (설명, 코드 쌍)
        """
        snippets = []

        # 함수와 클래스 정의 찾기
        # 멀티라인 주석 (""")
        docstring_pattern = re.compile(
            r'(def |class )([^\n:]+):\n\s*"""([^"]+)"""',
            re.MULTILINE
        )

        for match in docstring_pattern.finditer(content):
            keyword, signature, docstring = match.groups()

            # 한국어 포함 여부 확인
            if self.has_korean(docstring):
                # 함수/클래스 전체 추출 (간단하게 다음 def/class까지)
                start = match.start()

                # 다음 정의 찾기
                next_def = content.find('\ndef ', start + 1)
                next_class = content.find('\nclass ', start + 1)

                end = len(content)
                if next_def != -1:
                    end = min(end, next_def)
                if next_class != -1:
                    end = min(end, next_class)

                code = content[start:end].strip()

                # 너무 긴 코드는 제외 (2000자 이상)
                if len(code) > 2000:
                    continue

                snippets.append({
                    "instruction": f"{keyword.strip()} {signature.strip()}에 대한 설명",
                    "input": docstring.strip(),
                    "output": code
                })

        # 인라인 주석 (#)이 있는 코드 블록
        inline_comment_pattern = re.compile(
            r'((?:^|\n)(?:[ \t]*#[^\n]*\n)+)((?:[ \t]+[^\n]+\n)+)',
            re.MULTILINE
        )

        for match in inline_comment_pattern.finditer(content):
            comments, code = match.groups()

            # 한국어 포함 여부 확인
            if self.has_korean(comments):
                comments_clean = '\n'.join([
                    line.strip().lstrip('#').strip()
                    for line in comments.split('\n')
                    if line.strip().startswith('#')
                ])

                code_clean = code.strip()

                # 너무 짧거나 긴 코드는 제외
                if len(code_clean) < 20 or len(code_clean) > 1000:
                    continue

                snippets.append({
                    "instruction": "다음 설명에 맞는 코드를 작성해주세요",
                    "input": comments_clean,
                    "output": code_clean
                })

        return snippets

    def scrape_repo_files(self, repo_full_name: str) -> List[Dict]:
        """리포지토리의 Python 파일들을 스크래핑

        Args:
            repo_full_name: 리포지토리 full name (owner/repo)

        Returns:
            코드 스니펫 리스트
        """
        all_snippets = []

        try:
            # 리포지토리의 Python 파일 검색
            response = requests.get(
                f"{self.base_url}/search/code",
                headers=self.headers,
                params={
                    "q": f"repo:{repo_full_name} language:python extension:py",
                    "per_page": 30
                }
            )

            if response.status_code == 403:
                logger.warning(f"Rate limit 초과: {repo_full_name}")
                return all_snippets

            response.raise_for_status()
            data = response.json()

            for item in data.get("items", [])[:10]:  # 최대 10개 파일
                try:
                    # 파일 내용 가져오기
                    file_response = requests.get(item["url"], headers=self.headers)
                    file_response.raise_for_status()
                    file_data = file_response.json()

                    # Base64 디코딩
                    import base64
                    content = base64.b64decode(file_data["content"]).decode("utf-8")

                    # 코드 스니펫 추출
                    snippets = self.extract_code_snippets(content)
                    all_snippets.extend(snippets)

                    time.sleep(1)  # Rate limit 고려

                except Exception as e:
                    logger.debug(f"파일 처리 실패 ({item['path']}): {e}")
                    continue

        except Exception as e:
            logger.error(f"리포지토리 스크래핑 실패 ({repo_full_name}): {e}")

        return all_snippets

    def scrape(self, max_repos: int = 50, max_snippets: int = 1000) -> List[Dict]:
        """한국어 코드 스니펫 수집

        Args:
            max_repos: 최대 리포지토리 수
            max_snippets: 최대 스니펫 수

        Returns:
            수집된 코드 스니펫 리스트
        """
        logger.info("한국어 리포지토리 검색 중...")
        repos = self.search_korean_repos(max_repos=max_repos)

        all_snippets = []

        logger.info(f"{len(repos)}개 리포지토리에서 코드 수집 중...")
        for repo in tqdm(repos, desc="리포지토리 처리"):
            if len(all_snippets) >= max_snippets:
                break

            snippets = self.scrape_repo_files(repo["full_name"])
            all_snippets.extend(snippets)

            logger.info(f"{repo['full_name']}: {len(snippets)}개 스니펫 수집")

        logger.info(f"총 {len(all_snippets)}개 코드 스니펫 수집 완료")
        return all_snippets[:max_snippets]


def main():
    parser = argparse.ArgumentParser(description="GitHub에서 한국어 코드 수집")
    parser.add_argument(
        "--token",
        type=str,
        help="GitHub Personal Access Token (rate limit 증가)"
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=10,
        help="최소 star 수"
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=50,
        help="최대 리포지토리 수"
    )
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=1000,
        help="최대 스니펫 수"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/github_korean.jsonl",
        help="출력 파일 경로"
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 스크래퍼 실행
    scraper = GitHubKoreanCodeScraper(
        token=args.token,
        min_stars=args.min_stars
    )

    snippets = scraper.scrape(
        max_repos=args.max_repos,
        max_snippets=args.max_snippets
    )

    # 결과 저장 (JSONL 형식)
    with open(output_path, "w", encoding="utf-8") as f:
        for snippet in snippets:
            f.write(json.dumps(snippet, ensure_ascii=False) + "\n")

    logger.info(f"결과 저장: {output_path}")
    logger.info(f"총 {len(snippets)}개 스니펫 저장")


if __name__ == "__main__":
    main()
