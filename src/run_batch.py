import os
import pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


DATASET_PATH = "data/new qs.csv"
MODEL_ID = "mlx-community/Qwen3-4B-4bit"
OUTPUT_PATH = "results/qwen3_full_results.csv"
TEST_LIMIT = 384

df = pd.read_csv(DATASET_PATH)

print(f"Dataset contains {len(df)} rows.")
print(f"Testing the first {TEST_LIMIT} questions.")

print("\nLoading Qwen3...")
model, tokenizer = load(MODEL_ID)

# Load previously saved results, if any
os.makedirs("results", exist_ok=True)

if os.path.exists(OUTPUT_PATH):
    results = pd.read_csv(OUTPUT_PATH).to_dict("records")
    print(f"Resuming: {len(results)} answers already saved.")
else:
    results = []

# Identify completed question/dialect combinations
completed = {
    (str(r["question_id"]), str(r["variety"]))
    for r in results
}

for index, row in df.head(TEST_LIMIT).iterrows():
    key = (str(row["question_id"]), str(row["variety"]))

    if key in completed:
        print(f"Skipping completed: {key}")
        continue
    question = row["question"]

    messages = [
        {
            "role": "user",
            "content": (
                "أجب عن السؤال التالي بإجابة قصيرة ومباشرة باللغة العربية: "
                + question
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print(f"\n[{len(results) + 1}/{TEST_LIMIT}] {row['question_id']} ({row['variety']})")

    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=80,
        sampler=make_sampler(temp=0),
    )

    results.append({
        "question_id": row["question_id"],
        "variety": row["variety"],
        "question": question,
        "model_id": MODEL_ID,
        "model_answer": answer,
    })

    print(f"Answer: {answer}")

    # Save progress after every question
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

print(f"\nSaved {len(results)} answers to {OUTPUT_PATH}")