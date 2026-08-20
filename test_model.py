from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading FLAN-T5 model...")

model_name = "google/flan-t5-small"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = """
You are a helpful HR assistant.

Company policy:
Employees are entitled to 14 days of annual leave per year.

Employee question:
How many annual leave days do I have?

Give a short and clear answer.
"""

# Convert text into tokens
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Generate AI response
outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

# Convert generated tokens back to text
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nAI Response:")
print(answer)