
from pathlib import Path
import pandas as pd

SOURCE_FILE = Path("results/evidence_passages.csv")
OUTPUT_FILE = Path("results/evidence_passages_expanded.csv")

# Development evidence candidates based on external sources.
# These are paraphrases, not verbatim quotations.
# They remain pending until manually reviewed.

NEW_EVIDENCE = [
    {
        "question_id": "DG-P003",
        "source_url": "https://www.britannica.com/place/Ottoman-Empire",
        "evidence_passage": (
            "The Ottoman Empire was founded by Osman I, "
            "also known as Osman Gazi, around 1300. "
            "The Ottoman dynasty was named after him."
        ),
    },
    {
        "question_id": "DG-P007",
        "source_url": (
            "https://inside.fifa.com/tournaments/mens/worldcup/"
            "2018russia/news/from-russia-2018-to-qatar-2022-2986026"
        ),
        "evidence_passage": (
            "Russia hosted the 2018 FIFA World Cup. "
            "After the tournament, Russia handed over "
            "the hosting role to Qatar for the 2022 World Cup."
        ),
    },
    {
        "question_id": "DG-P011",
        "source_url": (
            "https://www.fm.gov.om/en/about-oman/state/geography/"
        ),
        "evidence_passage": (
            "The Sultanate of Oman is located in the "
            "southeastern corner of the Arabian Peninsula."
        ),
    },
]


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    original = pd.read_csv(SOURCE_FILE)

    required_columns = [
        "question_id",
        "source_url",
        "evidence_passage",
        "retrieval_status",
        "verification_status",
    ]

    missing = set(required_columns) - set(original.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Preserve original evidence and metadata.
    expanded = original.copy()

    existing_ids = set(
        expanded["question_id"].astype(str)
    )

    additions = []

    for item in NEW_EVIDENCE:
        if item["question_id"] in existing_ids:
            print(
                "Skipping existing:",
                item["question_id"],
            )
            continue

        additions.append({
            "question_id": item["question_id"],
            "source_url": item["source_url"],
            "evidence_passage": item["evidence_passage"],
            "retrieval_status": "passage_found",
            "verification_status": (
                "pending_answer_verification"
            ),
        })

    if additions:
        expanded = pd.concat(
            [expanded, pd.DataFrame(additions)],
            ignore_index=True,
        )

    expanded = expanded.sort_values(
        "question_id"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expanded.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Evidence Expansion ===")
    print("Original passages:", len(original))
    print("New candidates:", len(additions))
    print("Total passages:", len(expanded))
    print("Saved to:", OUTPUT_FILE)

    print("\nVerification statuses:")
    print(
        expanded["verification_status"]
        .value_counts()
        .to_string()
    )

    print("\nEvidence IDs:")
    print(
        expanded["question_id"]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
