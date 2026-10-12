
import os
import pandas as pd

RESULTS_PATH = "results/all_baseline_evaluation_draft.csv"
OUTPUT_PATH = "results/question_review.csv"

df = pd.read_csv(RESULTS_PATH)

# Give each model a shorter name.
model_names = {
    "mlx-community/Qwen3-4B-4bit": "Qwen3",
    "mlx-community/gemma-3-4b-it-4bit": "Gemma3",
    "ibm-research/granite-4.0-h-3b-ar": "Granite",
}

df["model"] = df["model_id"].map(model_names)

# One column per model and Arabic variety.
review = df.pivot(
    index="question_id",
    columns=["model", "variety"],
    values="model_answer"
)

review.columns = [
    f"{model}_{variety}"
    for model, variety in review.columns
]

review = review.reset_index()

# Include the reference answer and verification status.
reference = df[
    [
        "question_id",
        "gold_answer_draft",
        "verification_status",
        "review_notes",
    ]
].drop_duplicates(subset=["question_id"])

review = reference.merge(
    review,
    on="question_id",
    how="left",
    validate="one_to_one"
)

# Prioritize questions requiring verification.
priority = {
    "needs_context": 0,
    "needs_disambiguation": 1,
    "unverified_draft": 2,
    "source_checked": 3,
}

review["priority"] = review["verification_status"].map(priority)

review = review.sort_values(
    ["priority", "question_id"]
).drop(columns="priority")

os.makedirs("results", exist_ok=True)

review.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Review table created successfully!")
print("Total questions:", len(review))
print("Total columns:", len(review.columns))
print("Saved to:", OUTPUT_PATH)
