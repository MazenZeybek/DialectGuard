
import pandas as pd

FILE_PATH = "results/failure_review.csv"
QUESTION_ID = "DG-P006"

df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")

# Ensure annotation columns accept text.
for column in ["human_label", "human_notes", "reviewer"]:
    df[column] = df[column].astype("string")

# Preliminary labels from our response review.
labels = {
    "Gemma3": {
        "msa": "correct",
        "jordanian": "correct",
        "egyptian": "correct",
        "saudi": "correct",
    },
    "Granite": {
        "msa": "factual_error",
        "jordanian": "correct",
        "egyptian": "correct",
        "saudi": "correct",
    },
    "Qwen3": {
        "msa": "factual_error",
        "jordanian": "factual_error",
        "egyptian": "factual_error",
        "saudi": "factual_error",
    },
}

mask = df["question_id"] == QUESTION_ID

if mask.sum() != 12:
    raise ValueError(
        f"Expected 12 responses, found {mask.sum()}"
    )

# Protect previously recorded annotations.
if df.loc[mask, "human_label"].notna().any():
    raise ValueError(
        "DG-P006 already has annotations. "
        "Stopping to prevent overwriting."
    )

# Validate every model/dialect combination before saving.
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

# Add preliminary labels and notes.
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

        if model == "Granite" and dialect == "msa":
            note += (
                " Answer gives 3 September 1939, "
                "when Britain and France declared war. "
                "Conventional European WWII start date "
                "is 1 September 1939. "
                "Review alternative interpretation."
            )

        if model == "Qwen3":
            note += (
                " Incorrect start date compared with "
                "the reference answer: 1 September 1939."
            )

        df.loc[row_mask, "human_label"] = label
        df.loc[row_mask, "human_notes"] = note
        df.loc[row_mask, "reviewer"] = "assistant_preliminary"

df.to_csv(
    FILE_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("DG-P006 annotations saved!")
print("New labels:", mask.sum())
print("Total labeled:", df["human_label"].notna().sum())

print("\nDG-P006 label distribution:")
print(df.loc[mask, "human_label"].value_counts().to_string())

print("\nAll labeled responses:")
print(df["human_label"].value_counts().to_string())
