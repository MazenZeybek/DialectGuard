from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3-4B-4bit"

question = "في أي مدينة وُلد جمال خاشقجي؟"

print("Loading Qwen3...")
model, tokenizer = load(MODEL_ID)

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
    enable_thinking=True,
)

print("\nQuestion:", question)
print("Thinking mode: ENABLED")
print("\nGenerating answer...")

answer = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    sampler=make_sampler(temp=0),
)

print("\nModel output:")
print(answer)