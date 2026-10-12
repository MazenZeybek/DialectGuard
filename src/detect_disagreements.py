
import os
import re
import pandas as pd

INPUT_PATH = "results/question_review.csv"
OUTPUT_PATH = "results/dialect_disagreements.csv"

MODELS = ["Qwen3", "Gemma3", "Granite"]
DIALECTS = ["msa", "jordanian", "egyptian", "saudi"]


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def main():
    df = pd.read_csv(INPUT_PATH)
    results = []

    for _, row in df.iterrows():
        for model in MODELS:
            answers = [
                normalize(row[f"{model}_{dialect}"])
                for dialect in DIALECTS
            ]

            unique_answers = len(set(answers))

            results.append({
                "question_id": row["question_id"],
                "model": model,
                "verification_status": row["verification_status"],
                "unique_answer_count": unique_answers,
                "text_disagreement": unique_answers > 1,
                "msa_answer": row[f"{model}_msa"],
                "jordanian_answer": row[f"{model}_jordanian"],
                "egyptian_answer": row[f"{model}_egyptian"],
                "saudi_answer": row[f"{model}_saudi"],
            })

    result_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Disagreement analysis completed!")
    print("Total question-model combinations:", len(result_df))

    print("\nText disagreements by model:")
    print(
        pd.crosstab(
            result_df["model"],
            result_df["text_disagreement"]
        ).to_string()
    )

    print("\nSaved to:", OUTPUT_PATH)
    print("Note: Different wording does not necessarily mean different facts.")


if __name__ == "__main__":
    main()
