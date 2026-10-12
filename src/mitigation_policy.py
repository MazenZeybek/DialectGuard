
from pathlib import Path

import pandas as pd


INPUT = Path("results/evidence_verification_results.csv")
OUTPUT = Path("results/mitigation_decisions.csv")


def choose_action(row):
    """
    Choose a mitigation action based on available signals.

    These are policy decisions, not confirmed factual judgments.
    """

    evidence = row["evidence_status"]
    risk = row["risk_level"]

    answer = (
        str(row["model_answer"]).strip()
        if pd.notna(row["model_answer"])
        else ""
    )

    question = (
        str(row["question"]).strip()
        if pd.notna(row["question"])
        else ""
    )

    non_answer = bool(row["possible_non_answer"])
    disagreement = bool(row["text_disagreement"])

    # Detect responses that repeat the instruction.
    if answer.startswith("أجب عن السؤال التالي"):
        non_answer = True

    # Detect question repetition inside an instruction.
    if question and question in answer and "أجب عن" in answer:
        non_answer = True

    # Detect potentially truncated responses.
    # This is a heuristic, not proof of truncation.
    if len(answer) > 80 and answer.endswith(
        ("التي ت", "الذي ت")
    ):
        non_answer = True

    # 1. Retry possible non-answers.
    if evidence == "no_answer" or non_answer:
        return "regenerate"

    # 2. Verify answers that do not match a checked reference.
    if evidence == "reference_nonmatch_needs_review":
        return "verify_with_evidence"

    # 3. Accept answers matching an accepted reference.
    if evidence == "reference_supported_match":
        return "accept"

    # 4. Handle responses without checked references.
    if risk == "high":
        return "human_review"

    if disagreement:
        return "verify_with_evidence"

    return "accept"


def choose_reason(row):
    action = row["mitigation_action"]

    if action == "regenerate":
        return "Possible non-answer or truncated response"

    if row["evidence_status"] == "reference_nonmatch_needs_review":
        return "Answer does not match source-checked reference"

    if row["evidence_status"] == "reference_supported_match":
        return "Answer matches an accepted reference expression"

    if action == "human_review":
        return "High heuristic risk without verified reference"

    if action == "verify_with_evidence":
        return "Cross-dialect text disagreement requires verification"

    return "No risk signal requiring intervention under current policy"


def main():
    df = pd.read_csv(INPUT)

    required = [
        "question_id",
        "variety",
        "model",
        "question",
        "model_answer",
        "evidence_status",
        "risk_level",
        "possible_non_answer",
        "text_disagreement",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Assign mitigation actions.
    df["mitigation_action"] = df.apply(
        choose_action,
        axis=1
    )

    # Explain each decision.
    df["mitigation_reason"] = df.apply(
        choose_reason,
        axis=1
    )

    # Create output directory if necessary.
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save results.
    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Mitigation policy completed!")
    print("Responses processed:", len(df))

    print("\nMitigation actions:")
    print(
        df["mitigation_action"]
        .value_counts()
        .to_string()
    )

    print("\nActions by evidence status:")
    print(
        pd.crosstab(
            df["evidence_status"],
            df["mitigation_action"]
        ).to_string()
    )

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
