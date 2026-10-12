
from pathlib import Path
import pandas as pd

FILE = Path("results/evidence_passages_expanded.csv")

NEW_EVIDENCE = [
    {
        "question_id": "DG-P008",
        "source_url": (
            "https://en.wikisource.org/wiki/"
            "1911_Encyclop%C3%A6dia_Britannica/Saladin"
        ),
        "evidence_passage": (
            "Saladin died in Damascus on 4 March 1193 "
            "after a short illness."
        ),
    },
    {
        "question_id": "DG-P015",
        "source_url": (
            "https://www.nobelprize.org/prizes/"
            "literature/1988/mahfouz/facts/"
        ),
        "evidence_passage": (
            "Egyptian novelist and Nobel laureate "
            "Naguib Mahfouz died on 30 August 2006 "
            "in Cairo, Egypt."
        ),
    },
]

def main():
    if not FILE.exists():
        raise FileNotFoundError(FILE)

    df = pd.read_csv(FILE)

    existing_ids = set(df["question_id"].astype(str))
    additions = []

    for item in NEW_EVIDENCE:
        if item["question_id"] in existing_ids:
            print("Already exists:", item["question_id"])
            continue

        additions.append({
            **item,
            "retrieval_status": "passage_found",
            "verification_status": (
                "pending_answer_verification"
            ),
        })

    if additions:
        df = pd.concat(
            [df, pd.DataFrame(additions)],
            ignore_index=True,
        )

    df = df.sort_values(
        "question_id"
    ).reset_index(drop=True)

    df.to_csv(
        FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== Evidence Batch 2 ===")
    print("Added:", len(additions))
    print("Total passages:", len(df))
    print("Saved:", FILE)

    print("\nVerification statuses:")
    print(df["verification_status"].value_counts())

if __name__ == "__main__":
    main()
