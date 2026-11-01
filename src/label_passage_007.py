
import pandas as pd

PATH = "results/failure_review.csv"

df = pd.read_csv(PATH, encoding="utf-8-sig")

for col in ["human_label", "human_notes", "reviewer"]:
    df[col] = df[col].astype("string")

# Format:
# question_id: model: [MSA, Jordanian, Egyptian, Saudi]
C = "correct"
F = "factual_error"
N = "non_answer"

annotations = {
    "DG-P007": {
        "Gemma3": [C, C, C, C],
        "Granite": [C, N, C, N],
        "Qwen3": [F, F, F, F],
    },
}

dialects = ["msa", "jordanian", "egyptian", "saudi"]

# Validate everything before modifying the dataset.
updates = []

for question_id, models in annotations.items():
    for model, labels in models.items():
        for dialect, label in zip(dialects, labels):
            mask = (
                (df["question_id"] == question_id)
                & (df["model"] == model)
                & (df["variety"] == dialect)
            )

            if mask.sum() != 1:
                raise ValueError(
                    f"Invalid row: {question_id}/{model}/{dialect}"
                )

            if df.loc[mask, "human_label"].notna().any():
                raise ValueError(
                    f"Already labeled: {question_id}/{model}/{dialect}"
                )

            updates.append((mask, question_id, model, dialect, label))

for mask, question_id, model, dialect, label in updates:
    note = "Preliminary assistant review; requires independent human confirmation."

    if question_id == "DG-P007" and model == "Granite":
        if dialect == "jordanian":
            note += " Repeated instruction instead of answering."
        elif dialect == "saudi":
            note += " No host country given; possible output truncation."

    df.loc[mask, "human_label"] = label
    df.loc[mask, "human_notes"] = note
    df.loc[mask, "reviewer"] = "assistant_preliminary"

df.to_csv(PATH, index=False, encoding="utf-8-sig")

print("New annotations:", len(updates))
print("Total annotated:", df["human_label"].notna().sum())
print(df["human_label"].value_counts().to_string())
