import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Generative AI HR Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Generative AI HR Assistant")

st.write(
    "Ask a question about company HR policies."
)


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/hr_faq.csv")


df = load_data()


# -----------------------------
# Prepare TF-IDF search
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(
    df["question"]
)


# -----------------------------
# Load FLAN-T5 model
# -----------------------------
@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------
# User input
# -----------------------------
user_question = st.text_input(
    "Enter your HR question:"
)


# -----------------------------
# Ask AI button
# -----------------------------
if st.button("Ask AI"):

    if user_question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        # Convert user question into vector
        user_vector = vectorizer.transform(
            [user_question]
        )

        # Compare question with dataset
        similarities = cosine_similarity(
            user_vector,
            question_vectors
        )

        # Find best match
        best_index = similarities.argmax()

        similarity_score = similarities[0][best_index]

        matched_question = df.iloc[best_index]["question"]

        matched_policy = df.iloc[best_index]["answer"]


        # -----------------------------
        # Check similarity
        # -----------------------------
        if similarity_score < 0.65:

            st.error(
                "I could not find a relevant HR policy. "
                "Please contact the HR department."
            )

        else:

            # Create prompt
            prompt = f"""
You are a helpful HR assistant.

Answer the employee question using only the company policy below.

Company Policy:
{matched_policy}

Employee Question:
{user_question}

Give a short, clear and professional answer.
"""

            # Convert prompt to tokens
            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

            # Generate AI answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=60
            )

            # Decode result
            ai_answer = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )


            # -----------------------------
            # Display results
            # -----------------------------
            st.subheader("AI Answer")

            st.success(ai_answer)


            st.subheader("Matched HR Question")

            st.write(matched_question)


            st.subheader("Matched HR Policy")

            st.info(matched_policy)


            st.subheader("Similarity Score")

            st.write(
                f"{similarity_score:.3f}"
            )