
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

INPUT = Path("data/week2_development_questions.csv")
REVIEWED = Path("results/week2_wikipedia_evidence_reviewed.csv")
OUTPUT = Path("results/week2_wikipedia_evidence_collected.csv")

API = "https://ar.wikipedia.org/w/api.php"

COLUMNS = [
    "question_id",
    "source_url",
    "evidence_passage",
    "retrieval_status",
    "verification_status",
    "source_title",
    "source_page_id",
    "collection_method",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    questions = pd.read_csv(INPUT)
    questions = questions[
        (questions["split"] == "dev")
        & (questions["variety"] == "msa")
    ].head(args.limit)

    if OUTPUT.exists():
        records = pd.read_csv(OUTPUT).to_dict("records")
    elif REVIEWED.exists():
        records = pd.read_csv(REVIEWED).to_dict("records")
    else:
        records = []

    completed = {str(r["question_id"]) for r in records}

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "DialectGuardAcademicResearch/0.2 "
            "(https://github.com/MazenZeybek/DialectGuard)"
        )
    })

    print("Questions selected:", len(questions))
    print("Previously saved passages:", len(records))

    for _, row in questions.iterrows():
        qid = str(row["base_question_id"])

        if qid in completed:
            print("Already collected:", qid)
            continue

        print("Searching:", qid, flush=True)

        try:
            response = session.get(
                API,
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": str(row["question"]),
                    "gsrlimit": 1,
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "exintro": 1,
                    "inprop": "url",
                    "format": "json",
                    "maxlag": 5,
                },
                timeout=30,
            )

            if response.status_code == 429:
                delay = response.headers.get(
                    "Retry-After", "60"
                )
                print(
                    f"Rate limited. Retry-After: {delay}. "
                    "Stopping collection safely."
                )
                break

            response.raise_for_status()
            result = response.json()

            if "error" in result:
                print("API error:", result["error"])
                break

            pages = result.get(
                "query", {}
            ).get("pages", {})

            candidates = [
                p for p in pages.values()
                if "extract" in p
            ]

            if not candidates:
                print("No useful article found")
                continue

            article = candidates[0]
            passage = " ".join(
                str(article["extract"]).split()
            )

            if len(passage) < 80:
                print("Article too short")
                continue

            records.append({
                "question_id": qid,
                "source_url": article.get("fullurl", ""),
                "evidence_passage": passage[:1800],
                "retrieval_status": "wikipedia_api_retrieved",
                "verification_status": "unverified",
                "source_title": article.get("title", ""),
                "source_page_id": article.get("pageid", ""),
                "collection_method": "wikipedia_search_api",
            })

            completed.add(qid)
            print("Added:", article.get("title", ""))

            OUTPUT.parent.mkdir(
                parents=True, exist_ok=True
            )
            pd.DataFrame(
                records, columns=COLUMNS
            ).to_csv(
                OUTPUT,
                index=False,
                encoding="utf-8-sig",
            )

            time.sleep(5)

        except requests.RequestException as exc:
            print("Network/API error:", exc)
            print("Stopping without discarding saved data.")
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=COLUMNS).to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    print("\nCollection complete or safely stopped.")
    print("Passages saved:", len(records))
    print("Output:", OUTPUT)
    print("New passages still require relevance review.")


if __name__ == "__main__":
    main()
