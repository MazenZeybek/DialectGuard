
import pandas as pd

FILE_PATH = "results/failure_review.csv"

df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
for column in ["human_label", "human_notes", "reviewer"]:
    df[column] = df[column].astype("string")

QUESTION_ID = "DG-P002"

# Labels based on our preliminary review.
labels = {
    "Qwen3": {
        "msa": "factual_error",
        "jordanian": "factual_error",
        "egyptian": "factual_error",
        "saudi": "correct",
    },
    "Gemma3": {
        "msa": "factual_error",
        "jordanian": "factual_error",
        "egyptian": "factual_error",
        "saudi": "factual_error",
    },
    "Granite": {
        "msa": "correct",
        "jordanian": "correct",
        "egyptian": "correct",
        "saudi": "correct",
    },
}

# Prevent overwriting existing annotations.
mask = df["question_id"] == QUESTION_ID

if mask.sum() != 12:
    raise ValueError(
        f"Expected 12 responses, found {mask.sum()}"
    )

if df.loc[mask, "human_label"].notna().any():
    raise ValueError(
        "Some responses already have labels. "
        "Stopping to prevent overwriting."
    )

for model, dialect_labels in labels.items():
    for dialect, label in dialect_labels.items():
        row_mask = (
            (df["question_id"] == QUESTION_ID)
            & (df["model"] == model)
            & (df["variety"] == dialect)
        )

        if row_mask.sum() != 1:
            raise ValueError(
                f"Expected one row for {model}/{dialect}"
            )

        df.loc[row_mask, "human_label"] = label
        df.loc[row_mask, "human_notes"] = (
            "Preliminary assistant review; "
            "requires independent human confirmation."
        )
        df.loc[row_mask, "reviewer"] = "assistant_preliminary"

df.to_csv(
    FILE_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Initial labels added successfully!")
print("Question:", QUESTION_ID)
print("New labels:", mask.sum())
print("Total labeled responses:", df["human_label"].notna().sum())
print("\nLabel distribution:")
print(df.loc[mask, "human_label"].value_counts().to_string())
