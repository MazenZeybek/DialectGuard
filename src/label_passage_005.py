
import pandas as pd

FILE_PATH = "results/failure_review.csv"
QUESTION_ID = "DG-P005"

df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")

# Make sure annotation columns accept text.
for column in ["human_label", "human_notes", "reviewer"]:
    df[column] = df[column].astype("string")

# Preliminary labels based on our review.
labels = {
    "Gemma3": {
        "msa": "correct",
        "jordanian": "factual_error",
        "egyptian": "correct",
        "saudi": "correct",
    },
    "Granite": {
        "msa": "correct",
        "jordanian": "correct",
        "egyptian": "incomplete",
        "saudi": "correct",
    },
    "Qwen3": {
        "msa": "correct",
        "jordanian": "correct",
        "egyptian": "correct",
        "saudi": "correct",
    },
}

mask = df["question_id"] == QUESTION_ID

if mask.sum() != 12:
    raise ValueError(
        f"Expected 12 responses, found {mask.sum()}"
    )

# Prevent overwriting existing annotations.
if df.loc[mask, "human_label"].notna().any():
    raise ValueError(
        "DG-P005 already has annotations. "
        "Stopping to prevent overwriting."
    )

# Verify every model/dialect combination exists.
for model, dialect_labels in labels.items():
    for dialect in dialect_labels:
        row_mask = (
            (df["question_id"] == QUESTION_ID)
            & (df["model"] == model)
            & (df["variety"] == dialect)
        )

        if row_mask.sum() != 1:
            raise ValueError(
                f"Expected exactly one row for {model}/{dialect}"
            )

# Add labels and notes.
for model, dialect_labels in labels.items():
    for dialect, label in dialect_labels.items():
        row_mask = (
            (df["question_id"] == QUESTION_ID)
            & (df["model"] == model)
            & (df["variety"] == dialect)
        )

        note = (
            "Preliminary assistant review; "
            "requires independent human confirmation."
        )

        if model == "Gemma3" and dialect == "jordanian":
            note += (
                " Answer gives 28 June 1914 instead "
                "of 28 July 1914."
            )

        if model == "Granite" and dialect == "egyptian":
            note += (
                " Answer gives only the year 1914, "
                "not the requested full date."
            )

        df.loc[row_mask, "human_label"] = label
        df.loc[row_mask, "human_notes"] = note
        df.loc[row_mask, "reviewer"] = "assistant_preliminary"

df.to_csv(
    FILE_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("DG-P005 annotations saved!")
print("New labels:", mask.sum())
print("Total labeled:", df["human_label"].notna().sum())

print("\nDG-P005 label distribution:")
print(df.loc[mask, "human_label"].value_counts().to_string())

print("\nAll labeled responses:")
print(df["human_label"].value_counts().to_string())
