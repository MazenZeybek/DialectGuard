
import os
import pandas as pd

INPUT_PATH = "results/all_baseline_evaluation_draft.csv"
OUTPUT_PATH = "results/failure_review.csv"

MODEL_NAMES = {
    "mlx-community/Qwen3-4B-4bit": "Qwen3",
    "mlx-community/gemma-3-4b-it-4bit": "Gemma3",
    "ibm-research/granite-4.0-h-3b-ar": "Granite",
}


def main():
    df = pd.read_csv(INPUT_PATH)

    # Review only questions with source-checked references.
    review = df[
        df["verification_status"] == "source_checked"
    ].copy()

    review["model"] = review["model_id"].map(MODEL_NAMES)

    # Human labels must remain independent of automatic matching.
    review["human_label"] = ""
    review["human_notes"] = ""
    review["reviewer"] = ""

    columns = [
        "question_id",
        "variety",
        "question",
        "model",
        "model_answer",
        "gold_answer_draft",
        "accepted_answers_draft",
        "source_url",
        "evaluation_label",
        "human_label",
        "human_notes",
        "reviewer",
    ]

    review = review[columns].sort_values(
        ["question_id", "model", "variety"]
    )

    os.makedirs("results", exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Rename or back it up before regenerating "
            "to avoid overwriting human annotations."
        )

    review.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Failure review file created!")
    print("Responses to review:", len(review))
    print("Unique questions:", review["question_id"].nunique())
    print("Models:", review["model"].nunique())
    print("Human labels filled:", review["human_label"].notna().sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
