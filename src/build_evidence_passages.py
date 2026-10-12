
from pathlib import Path
import pandas as pd

SOURCE_FILE = Path("results/retrieved_evidence.csv")
OUTPUT_FILE = Path("results/evidence_passages.csv")

SEARCH_TERMS = {
    "DG-P001": "ولد جمال خاشقجي",
    "DG-P002": "capital of the Kingdom of Morocco",
    "DG-P005": "July 28, 1914",
    "DG-P006": "1 September 1939",
    "DG-P010": "The Nobel Prize in Physics 1921 was awarded",
}


def main():
    sources = pd.read_csv(SOURCE_FILE)
    records = []

    for _, row in sources.iterrows():
        qid = row["question_id"]

        if qid not in SEARCH_TERMS:
            continue

        if pd.isna(row["text_path"]):
            continue

        path = Path(row["text_path"])

        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        term = SEARCH_TERMS[qid]

        position = text.casefold().find(term.casefold())

        if position < 0:
            print(f"{qid}: evidence phrase not found")
            continue

        start = max(0, position - 100)
        end = min(len(text), position + 350)

        passage = text[start:end]

        records.append({
            "question_id": qid,
            "source_url": row["source_url"],
            "evidence_passage": passage,
            "retrieval_status": "passage_found",
            "verification_status": "pending_answer_verification",
        })

        print(f"{qid}: evidence passage saved")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTotal passages:", len(records))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
