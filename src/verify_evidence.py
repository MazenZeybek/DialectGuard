
from pathlib import Path
import pandas as pd

FILE = Path("results/evidence_passages_expanded.csv")

VERIFIED_IDS = {
    "DG-P003",
    "DG-P007",
    "DG-P008",
    "DG-P011",
    "DG-P015",
}

def main():
    if not FILE.exists():
        raise FileNotFoundError(FILE)

    df = pd.read_csv(FILE)

    required_columns = {
        "question_id",
        "source_url",
        "evidence_passage",
        "retrieval_status",
        "verification_status",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    existing_ids = set(df["question_id"].astype(str))
    unknown_ids = VERIFIED_IDS - existing_ids

    if unknown_ids:
        raise ValueError(
            f"Unknown question IDs: {sorted(unknown_ids)}"
        )

    mask = df["question_id"].isin(VERIFIED_IDS)

    df.loc[mask, "verification_status"] = "manually_verified"

    df.to_csv(
        FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=== Evidence Verification ===")
    print("Total passages:", len(df))
    print("Verified IDs:", sorted(VERIFIED_IDS))

    print("\nVerification statuses:")
    print(df["verification_status"].value_counts())

    print("\nSaved:", FILE)


if __name__ == "__main__":
    main()
