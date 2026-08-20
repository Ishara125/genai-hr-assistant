import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


print("Loading HR dataset...")

# Load dataset
df = pd.read_csv("data/hr_faq.csv")


# Prepare HR question search
vectorizer = TfidfVectorizer(stop_words="english")

question_vectors = vectorizer.fit_transform(
    df["question"]
)


print("Loading Generative AI model...")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Generative AI model loaded.")


# Ask employee question
user_question = input(
    "\nEnter your HR question: "
)


# Convert question to vector
user_vector = vectorizer.transform(
    [user_question]
)


# Find similarities
similarities = cosine_similarity(
    user_vector,
    question_vectors
)


# Get best match
best_index = similarities.argmax()

similarity_score = similarities[0][best_index]

matched_question = df.iloc[best_index]["question"]

matched_policy = df.iloc[best_index]["answer"]


print("\nMatched FAQ:")
print(matched_question)

print("\nSimilarity Score:")
print(round(similarity_score, 3))


# Check whether match is relevant enough
if similarity_score < 0.20:

    print("\nAI Answer:")
    print(
        "I could not find a relevant HR policy. "
        "Please contact the HR department."
    )

else:

    # Create prompt for FLAN-T5
    prompt = f"""
You are a helpful HR assistant.

Answer the employee question using only the company policy below.

Company Policy:
{matched_policy}

Employee Question:
{user_question}

Give a short, clear and professional answer.
"""

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=60
    )

    # Decode answer
    ai_answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nAI Answer:")
    print(ai_answer)

    print("\nMatched HR Policy:")
    print(matched_policy)