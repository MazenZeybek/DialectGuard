
from pathlib import Path
import pandas as pd

from mlx_lm import load, generate


INPUT = Path("data/new qs.csv")
OUTPUT = Path("results/normalization_results.csv")

MODEL_PATH = "ibm-research/granite-4.0-h-3b-ar"

# Start with a small independent test.
MAX_QUESTIONS = 5



def build_prompt(question):
    return (
        "مهمتك إعادة صياغة السؤال باللهجة العربية إلى العربية الفصحى الحديثة فقط.\n"
        "هذه مهمة تحويل لغوي وليست مهمة إجابة عن السؤال.\n\n"
        "قواعد إلزامية:\n"
        "1. حافظ على المعنى الأصلي للسؤال بدقة.\n"
        "2. لا تضف أي معلومات أو أسئلة جديدة.\n"
        "3. لا تغيّر نوع المعلومة المطلوبة (مدينة، تاريخ، شخص، عدد).\n"
        "4. إذا كان السؤال يسأل عن مدينة، فلا تضف سؤالاً عن الزمن.\n"
        "5. أخرج سؤالاً واحداً فقط بالفصحى دون شرح.\n\n"
        "أمثلة:\n"
        "اللهجة: بأي مدينة انولد جمال خاشقجي؟\n"
        "الفصحى: في أي مدينة وُلد جمال خاشقجي؟\n\n"
        "اللهجة: شو عاصمة المغرب؟\n"
        "الفصحى: ما عاصمة المغرب؟\n\n"
        f"اللهجة: {question}\n"
        "الفصحى:"
    )



def normalize(model, tokenizer, question):
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
        max_tokens=100,
        verbose=False,
    ).strip()


def main():
    df = pd.read_csv(INPUT)

    msa = (
        df[df["variety"] == "msa"]
        [["question_id", "question"]]
        .rename(columns={"question": "reference_msa"})
    )

    dialects = df[df["variety"] != "msa"].copy()

    # Select five question IDs (15 dialectal prompts).
    selected_ids = (
        dialects["question_id"]
        .drop_duplicates()
        .head(MAX_QUESTIONS)
    )

    dialects = dialects[
        dialects["question_id"].isin(selected_ids)
    ].copy()

    dialects = dialects.merge(
        msa,
        on="question_id",
        how="left",
        validate="many_to_one",
    )

    print("Questions selected:", len(selected_ids))
    print("Dialectal prompts:", len(dialects))

    print("\nLoading Granite...")
    model, tokenizer = load(MODEL_PATH)

    results = []

    for _, row in dialects.iterrows():
        question = row["question"]

        print(
            f"\n{row['question_id']} "
            f"({row['variety']})"
        )

        normalized = normalize(
            model,
            tokenizer,
            question,
        )

        results.append({
            "question_id": row["question_id"],
            "variety": row["variety"],
            "original_question": question,
            "normalized_msa": normalized,
            "reference_msa": row["reference_msa"],
            "normalization_model": MODEL_PATH,
        })

        print("Original:", question)
        print("Normalized:", normalized)
        print("Reference:", row["reference_msa"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nNormalization test completed!")
    print("Results:", len(results))
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
