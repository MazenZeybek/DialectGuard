
from pathlib import Path
from urllib.parse import urlparse
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


INPUT = Path("data/qs_AK.csv")
OUTPUT = Path("results/retrieved_evidence.csv")
TEXT_DIR = Path("data/evidence_text")

HEADERS = {
    "User-Agent": "DialectGuard-Research/0.1 (academic evidence retrieval)"
}


def extract_html_text(content):
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup([
        "script", "style", "nav", "footer",
        "header", "noscript"
    ]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)


def retrieve(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "pdf" in content_type or urlparse(
        response.url
    ).path.lower().endswith(".pdf"):
        return {
            "status": "pdf_requires_extraction",
            "text": "",
            "http_status": response.status_code,
            "final_url": response.url,
        }

    text = extract_html_text(response.content)

    return {
        "status": "retrieved" if text else "empty_text",
        "text": text,
        "http_status": response.status_code,
        "final_url": response.url,
    }


def main():
    df = pd.read_csv(INPUT)

    checked = df[
        (df["verification_status"] == "source_checked")
        & df["source_url"].notna()
    ]

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in checked.iterrows():
        question_id = row["question_id"]
        url = row["source_url"]

        print(f"\nRetrieving {question_id}: {url}")

        try:
            result = retrieve(url)

            if result["text"]:
                text_path = TEXT_DIR / f"{question_id}.txt"
                text_path.write_text(
                    result["text"],
                    encoding="utf-8",
                )
            else:
                text_path = None

            record = {
                "question_id": question_id,
                "source_url": url,
                "final_url": result["final_url"],
                "retrieval_status": result["status"],
                "http_status": result["http_status"],
                "text_path": str(text_path) if text_path else "",
                "text_length": len(result["text"]),
                "error": "",
            }

        except requests.RequestException as exc:
            record = {
                "question_id": question_id,
                "source_url": url,
                "final_url": "",
                "retrieval_status": "retrieval_failed",
                "http_status": "",
                "text_path": "",
                "text_length": 0,
                "error": str(exc),
            }

        records.append(record)
        print("Status:", record["retrieval_status"])
        print("Text length:", record["text_length"])

        # Avoid rapid repeated requests.
        time.sleep(1)

    pd.DataFrame(records).to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEvidence retrieval completed!")
    print("Saved to:", OUTPUT)

    print("\nRetrieval status counts:")
    print(
        pd.DataFrame(records)["retrieval_status"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
