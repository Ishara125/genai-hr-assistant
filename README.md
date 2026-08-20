# Generative AI HR Assistant

A simple Generative AI application for Human Resources that helps employees get quick answers to common HR policy questions.

The system combines:

- **TF-IDF** for text representation
- **Cosine Similarity** for FAQ matching
- **Google FLAN-T5 Small** for Generative AI responses
- **Streamlit** for the user interface
- **Pandas** for dataset handling

---

## 1. Project Overview

Human Resources teams often receive the same employee questions repeatedly, such as:

- How many annual leave days do I get?
- Can I work from home?
- When is salary paid?
- What is the probation period?
- Do employees receive health insurance?

This project provides a simple AI-based HR assistant that searches a small HR FAQ dataset, retrieves the most relevant policy, and uses a Generative AI model to produce a short and natural answer.

---

## 2. Business Use Case

**Business Area:** Human Resources  
**Use Case:** Employee HR FAQ Assistant

The main goal is to reduce repetitive HR enquiries and provide employees with faster access to common company policy information.

The assistant can support questions related to:

- Annual leave
- Sick leave
- Remote work
- Working hours
- Probation
- Resignation
- Salary
- Overtime
- Health insurance
- Training
- Performance reviews
- HR contact information

---

## 3. Objectives

The project was developed to:

1. Build a simple Generative AI solution for an HR business use case.
2. Retrieve the most relevant HR policy from a structured FAQ dataset.
3. Generate a short natural-language answer using an open-source language model.
4. Provide a user-friendly web interface using Streamlit.
5. Handle unsupported questions safely instead of generating unrelated HR answers.

---

## 4. System Architecture

```text
Employee Question
        |
        v
Streamlit Web Interface
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity Search
        |
        v
Best Matching HR FAQ
        |
        v
Similarity Threshold Check
        |
        +-----------------------------+
        |                             |
        v                             v
Relevant Match                  Low Similarity
        |                             |
        v                             v
Matched HR Policy            Fallback Message
        |
        v
FLAN-T5 Generative AI Model
        |
        v
Generated HR Answer
```

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Load and manage the HR FAQ dataset |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Hugging Face Transformers | Load and run FLAN-T5 |
| PyTorch | Backend used by the language model |
| SentencePiece | Tokenization dependency for FLAN-T5 |
| Streamlit | Web-based user interface |
| CSV | HR FAQ dataset format |

---

## 6. Dataset

The project uses a custom HR FAQ dataset stored in:

```text
data/hr_faq.csv
```

The dataset contains HR-related questions and answers.

Example:

```csv
question,answer
How many annual leave days are available?,Employees are entitled to 14 days of annual leave per year.
Can employees work from home?,Employees may work from home up to two days per week with manager approval.
Do employees receive health insurance?,Eligible full-time employees are provided with company health insurance benefits.
```

The current prototype contains approximately 25 FAQ records.

---

## 7. Generative AI Model

The application uses:

```text
google/flan-t5-small
```

FLAN-T5 is an open-source instruction-tuned language model available through Hugging Face.

The model is used only after a relevant HR policy has been retrieved.

Example prompt:

```text
You are a helpful HR assistant.

Answer the employee question using only the company policy below.

Company Policy:
Employees are entitled to 14 days of annual leave per year.

Employee Question:
How many annual leave days do I get?

Give a short, clear and professional answer.
```

Example generated response:

```text
14 days
```

---

## 8. Information Retrieval Method

### TF-IDF

TF-IDF converts the HR FAQ questions into numerical vectors based on important words.

### Cosine Similarity

Cosine similarity compares the employee's question with all questions in the dataset and finds the closest match.

Example:

```text
User Question:
How many annual leave days do I get?

Matched HR Question:
How many annual leave days are available?

Similarity Score:
0.859
```

---

## 9. Similarity Threshold

A similarity threshold is used to prevent unrelated questions from being treated as valid HR policies.

Current threshold:

```python
if similarity_score < 0.65:
```

If the similarity score is below `0.65`, the system does not send the unrelated policy to the language model.

Instead, it displays:

```text
I could not find a relevant HR policy. Please contact the HR department.
```

This helps reduce incorrect or misleading responses.

---

## 10. Application Features

The Streamlit application provides:

- HR question input field
- Ask AI button
- AI-generated answer
- Matched HR FAQ question
- Matched HR policy
- Similarity score
- Empty-question validation
- Unknown-question fallback handling

---

## 11. Example Test Results

### Test 1 - Annual Leave

**Question**

```text
How many annual leave days do I get?
```

**AI Answer**

```text
14 days
```

**Matched Policy**

```text
Employees are entitled to 14 days of annual leave per year.
```

**Similarity Score**

```text
0.859
```

### Test 2 - Work From Home

**Question**

```text
Can I work from home?
```

**AI Answer**

```text
Yes
```

**Matched Policy**

```text
Employees may work from home up to two days per week with manager approval.
```

**Similarity Score**

```text
0.880
```

### Test 3 - Health Insurance

**Question**

```text
Do employees get health insurance?
```

**AI Answer**

```text
Eligible full-time employees are provided with company health insurance benefits.
```

**Similarity Score**

```text
0.844
```

### Test 4 - Unsupported Question

**Question**

```text
Does the company give free airline tickets?
```

**System Response**

```text
I could not find a relevant HR policy. Please contact the HR department.
```

This demonstrates that the system avoids generating an answer when no sufficiently relevant HR policy is found.

---

## 12. Project Structure

```text
genai-hr-assistant/
|
|-- app.py
|-- test_model.py
|-- test_search.py
|-- test_assistant.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- data/
|   `-- hr_faq.csv
|
`-- screenshots/
    |-- dashboard.png
    |-- annual_leave.png
    |-- work_from_home.png
    |-- health_insurance.png
    `-- unknown_question.png
```

---

## 13. Installation

### Step 1 - Open the project

Open the project folder in VS Code.

### Step 2 - Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3 - Activate the virtual environment

Windows:

```powershell
.venv\Scripts\activate
```

### Step 4 - Install required libraries

```powershell
pip install streamlit pandas scikit-learn transformers torch sentencepiece
```

---

## 14. How to Run the Application

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Run:

```powershell
streamlit run app.py
```

Streamlit will normally provide a local URL such as:

```text
http://localhost:8501
```

Open that address in a browser if it does not open automatically.

---

## 15. Testing

The project contains separate scripts for testing each part.

### Test FLAN-T5

```powershell
python test_model.py
```

### Test HR FAQ Search

```powershell
python test_search.py
```

### Test Search + Generative AI

```powershell
python test_assistant.py
```

### Run the Full Streamlit Application

```powershell
streamlit run app.py
```

---

## 16. Screenshots

Store demonstration screenshots inside the `screenshots` folder.

Suggested names:

```text
screenshots/dashboard.png
screenshots/annual_leave.png
screenshots/work_from_home.png
screenshots/health_insurance.png
screenshots/unknown_question.png
```

The screenshots can demonstrate:

- The Streamlit interface
- Correct HR answers
- Matched HR questions
- Matched HR policies
- Similarity scores
- Fallback handling for unsupported questions

---

## 17. Business Benefits

This solution can provide several benefits:

- Reduces repetitive questions handled manually by HR staff
- Provides employees with faster answers
- Improves consistency of common HR responses
- Provides basic HR information through a simple self-service interface
- Demonstrates how Generative AI can be connected to organizational knowledge
- Reduces unsupported answers by using a similarity threshold

---

## 18. Limitations

The current prototype has several limitations:

- It uses a small manually created HR FAQ dataset.
- TF-IDF mainly depends on word similarity and has limited semantic understanding.
- FLAN-T5 Small can sometimes produce very short responses.
- The similarity threshold may need adjustment when the dataset changes.
- The application does not currently support authentication.
- The application does not connect to a real company HR database.
- The model should not be used for important employment decisions.

---

## 19. Future Improvements

Possible future improvements include:

- Sentence-transformer embeddings
- Vector databases such as FAISS or Chroma
- Retrieval-Augmented Generation (RAG) using HR policy documents
- PDF HR policy upload
- Larger or more capable language models
- Conversation history
- Employee authentication
- Multilingual support
- Sinhala and Tamil HR questions
- Admin dashboard
- Feedback collection
- Real HR system integration

---

## 20. Responsible AI Considerations

The assistant should provide answers only when a sufficiently relevant company policy is available.

If no reliable policy is found, the application directs the employee to the HR department instead of generating an unsupported answer.

Sensitive employee information should not be entered unless suitable privacy, security, and access controls are implemented.

Important employment decisions should always involve qualified HR professionals.

---

## 21. Conclusion

The Generative AI HR Assistant demonstrates a simple business application of Generative AI in Human Resources.

The application combines information retrieval and an open-source language model to answer common employee questions. TF-IDF and cosine similarity are used to identify the most relevant HR FAQ, while FLAN-T5 generates a concise response based on the retrieved policy.

The Streamlit interface makes the system easy to demonstrate and use. The similarity threshold also provides a simple safety mechanism by rejecting unsupported questions and directing users to HR.

This prototype successfully demonstrates how Generative AI can be applied to a practical HR support use case while keeping the implementation simple, explainable, and suitable for a short academic demonstration.
