import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load HR FAQ dataset
df = pd.read_csv("data/hr_faq.csv")

print("HR dataset loaded successfully.")
print(f"Total HR questions: {len(df)}")


# Convert HR questions into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")

question_vectors = vectorizer.fit_transform(
    df["question"]
)


# Ask employee question
user_question = input("\nEnter your HR question: ")


# Convert employee question into a vector
user_vector = vectorizer.transform(
    [user_question]
)


# Compare employee question with all HR questions
similarities = cosine_similarity(
    user_vector,
    question_vectors
)


# Find the best matching question
best_index = similarities.argmax()

similarity_score = similarities[0][best_index]


# Get matched question and answer
matched_question = df.iloc[best_index]["question"]
matched_answer = df.iloc[best_index]["answer"]


print("\n-----------------------------")
print("Best Matching HR Question:")
print(matched_question)

print("\nHR Policy:")
print(matched_answer)

print("\nSimilarity Score:")
print(round(similarity_score, 3))

print("-----------------------------")