
from pathlib import Path
import pandas as pd

from mlx_lm import load, generate


INPUT = Path("results/mitigation_decisions.csv")
OUTPUT = Path("results/regeneration_results.csv")

MODEL_PATHS = {
    "Qwen3": "mlx-community/Qwen3-4B-4bit",
    "Gemma3": "mlx-community/gemma-3-4b-it-4bit",
    "Granite": "ibm-research/granite-4.0-h-3b-ar",
}


def build_prompt(question):
    return (
        "أجب عن السؤال التالي باللغة العربية.\n"
        "أعطِ إجابة قصيرة ومباشرة ودقيقة.\n"
        "لا تكرر السؤال ولا تشرح التعليمات.\n"
        "إذا لم تكن متأكدًا من الإجابة، قل: لا أعرف.\n\n"
        f"السؤال: {question}\n"
        "الإجابة:"
    )


def regenerate(model, tokenizer, question):
    prompt = build_prompt(question)

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=160,
        verbose=False,
    ).strip()


def main():
    df = pd.read_csv(INPUT)

    pending = df[
        df["mitigation_action"] == "regenerate"
    ].copy()

    key = ["question_id", "variety", "model"]

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        completed = set(
            tuple(row)
            for row in existing[key].itertuples(
                index=False, name=None
            )
        )
    else:
        existing = pd.DataFrame()
        completed = set()

    pending = pending[
        ~pending[key].apply(tuple, axis=1).isin(completed)
    ]

    print("Remaining responses:", len(pending))

    if pending.empty:
        print("All regeneration cases completed.")
        return

    for model_name, group in pending.groupby("model"):
        print(f"\nLoading {model_name}...")
        model, tokenizer = load(MODEL_PATHS[model_name])

        for _, row in group.iterrows():
            print(
                f"\nRegenerating {row['question_id']} "
                f"({row['variety']})"
            )

            new_answer = regenerate(
                model, tokenizer, row["question"]
            )

            result = pd.DataFrame([{
                "question_id": row["question_id"],
                "variety": row["variety"],
                "model": model_name,
                "question": row["question"],
                "original_answer": row["model_answer"],
                "regenerated_answer": new_answer,
                "original_action": "regenerate",
            }])

            # Save each completed response immediately.
            write_header = not OUTPUT.exists()

            result.to_csv(
                OUTPUT,
                mode="a",
                header=write_header,
                index=False,
                encoding="utf-8-sig" if write_header else "utf-8",
            )

            print("Original:", row["model_answer"])
            print("Regenerated:", new_answer)

    print("\nRegeneration batch completed!")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
