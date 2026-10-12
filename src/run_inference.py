from mlx_lm.sample_utils import make_sampler
import pandas as pd
from mlx_lm import load, generate

DATASET_PATH = "data/new qs.csv"
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

# Read the dataset
df = pd.read_csv(DATASET_PATH)

print("DialectGuard — Qwen3 Inference Test")
print(f"Total questions: {len(df)}")

# Select the first question
row = df.iloc[0]
question = row["question"]

print(f"\nQuestion ID: {row['question_id']}")
print(f"Dialect: {row['variety']}")
print(f"Question: {question}")

# Load Qwen3
print("\nLoading Qwen3-4B...")
model, tokenizer = load(MODEL_ID)

# Format the prompt using Qwen's chat template
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

# Generate an answer
print("\nGenerating answer...")
answer = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=80,
    sampler=make_sampler(temp=0),
)

print(f"\nModel answer: {answer}")