
# 🧠 RAG-Driven Question Answering API (TinyLlama + FAISS + LoRA)

## 🎯 Objective: Why Retrieval-Augmented Generation (RAG)?

Modern language models have strong reasoning skills, but limited access to **real-world or domain-specific knowledge**.
To solve this, **Retrieval-Augmented Generation (RAG)** combines:

> 🔹 *Information retrieval* (from documents)
> 🔹 *Generative reasoning* (via an LLM)

so the chatbot can answer based on **grounded, factual context**.

This project demonstrates how to build a **domain-adapted RAG pipeline** that can:

* Ingest PDFs and text files into a vector database.
* Retrieve semantically similar passages to user questions.
* Use a local language model to generate coherent, evidence-based answers.
* Optionally improve model understanding with **LoRA fine-tuning**.

The example use case focuses on **mental health and social policy in Indonesia**, using a curated dataset of PDF articles and text reports.

---

## 🧩 Architecture Overview


## ⚙️ Tech Stack

| Component           | Purpose                      | Framework/Library                                            |
| ------------------- | ---------------------------- | ------------------------------------------------------------ |
| **Backend API**     | Serve endpoints              | [FastAPI](https://fastapi.tiangolo.com/)                     |
| **Vector Store**    | Context retrieval            | [FAISS](https://github.com/facebookresearch/faiss)           |
| **Embeddings**      | Semantic similarity encoding | [HuggingFace Sentence Transformers](https://www.sbert.net/)  |
| **Model Runtime**   | Local generation             | [Transformers](https://huggingface.co/docs/transformers)     |
| **Fine-tuning**     | Lightweight adaptation       | [PEFT + LoRA](https://huggingface.co/docs/peft)              |
| **Docs Processing** | PDF & TXT parsing            | [LangChain Community Loaders](https://python.langchain.com/) |
---

## 🧱 Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py                # FastAPI server and routes
│   ├── rag_pipeline.py        # Core RAG logic (retriever + generator)
│   ├── indexing.py            # Builds FAISS vector index
│   ├── config.py              # Model and path configuration
│
├── finetune/
│   ├── train_lora.py          # LoRA fine-tuning script
│   └── adapter/               # Fine-tuned adapter weights (if available)
│
├── data/
│   ├── sample_docs/           # PDF/TXT source documents
│   └── mental_health_qa.jsonl # Fine-tuning dataset
│
├── requirements.txt
└── README.md
```

---
## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare the Documents

Add your `.pdf` and `.txt` files to:

```
data/sample_docs/
```

For example:

```
data/sample_docs/
├── Free from pasung.pdf
├── Barriers and facilitators.pdf
├── Cultural diversity in beliefs.pdf
├── Problems Among Indonesian Adolescents.pdf
├── First 1000 Days.pdf
└── Pasung.txt
```

### 5️⃣ Build the FAISS Index

```bash
python -m app.indexing
```

Expected output:

```
📚 Indexed 6 files → 477 chunks total.
💾 FAISS index saved to vector_index/
```

### 6️⃣ Run the API

```bash
uvicorn app.main:app --reload
```

Visit:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
to open the Swagger UI and test the `/ask` endpoint.

### 7️⃣ Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is pasung?"}'
```

Output:

```json
{
  "question": "What is pasung?",
  "answer": "Pasung is a long-standing custom in Indonesia that involves shackling individuals with mental illness.",
  "context_sources": [
    "Free from pasung.pdf (page 1)",
    "Pasung.txt (page ?)"
  ],
  "metadata": {
    "model": "tinyllama-LoRA",
    "retrieval_engine": "FAISS",
    "timestamp": "2025-10-19T09:00:00Z"
  }
}
```

---

## 🧠 Fine-Tuning with LoRA (Optional)

You can enhance the model’s factual alignment using your own Q&A dataset.

### Dataset Format

`data/mental_health_qa.jsonl`

```json
{"instruction": "What is pasung?", "output": "Pasung is a traditional practice in Indonesia where people with mental illness are restrained using shackles or wooden blocks."}
{"instruction": "What are the main barriers to mental health care?", "output": "Limited access, stigma, and insufficient health facilities are the main challenges."}
```

### Run Fine-Tuning

```bash
python finetune/train_lora.py
```

This creates LoRA adapter weights in:

```
finetune/adapter/
```

Once generated, the main pipeline automatically detects and loads them:

```
🧩 Found LoRA adapter. Loading adapted weights...
✅ RAG pipeline ready. Using model: tinyllama-LoRA
```

---

## 📊 Example Use Cases

* Summarizing insights from internal PDFs
* Answering domain-specific questions (e.g., policy, research)
* Training lightweight AI assistants with local documents
* Demonstrating fine-tuning workflows for small models

---

## 📈 Design Highlights

| Feature                     | Description                                       |
| --------------------------- | ------------------------------------------------- |
| **RAG with FAISS**          | Efficient semantic retrieval from document chunks |
| **HuggingFace Integration** | Local text generation using Transformers          |
| **Auto LoRA Loading**       | Detects and merges fine-tuned adapter weights     |
| **Explainable Responses**   | Includes document source references               |
| **Modular Components**      | Easy to extend with new models or datasets        |

---

## ⚙️ Environment Requirements

| Requirement  | Version |
| ------------ | ------- |
| Python       | ≥ 3.10  |
| torch        | ≥ 2.0   |
| transformers | ≥ 4.40  |
| fastapi      | latest  |
| uvicorn      | latest  |
| faiss-cpu    | latest  |

---

## 👤 Author

**Muhammad Zakaria Saputra**
TensorFlow Developer | AI Product Builder | Applied Research Enthusiast

💬 *“RAG bridges what models know and what they should know, making AI systems both factual and grounded.”*
