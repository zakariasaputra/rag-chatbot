import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import VECTOR_DB_PATH, EMBEDDING_MODEL, OLLAMA_MODEL
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def load_rag_pipeline():
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {VECTOR_DB_PATH}")
    print(f"📁 Loading FAISS index from: {VECTOR_DB_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print(f"🧠 Loading base model: {OLLAMA_MODEL}")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

    adapter_path = "finetune/adapter"
    adapter_file_bin = os.path.join(adapter_path, "adapter_model.bin")
    adapter_file_safe = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file_bin) or os.path.exists(adapter_file_safe):
        print("🧩 Found LoRA adapter. Loading adapted weights...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_name_display = f"{OLLAMA_MODEL}-LoRA"
    else:
        print("⚠️ No LoRA adapter found. Using base model only.")
        model = base_model
        model_name_display = OLLAMA_MODEL

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    PROMPT_TEMPLATE = (
        "You are a helpful assistant specializing in Indonesian mental health and policy.\n"
        "Using only the information in the provided context, answer factually and avoid adding outside knowledge.\n"
        "Make sure the response is complete and does not stop mid-sentence.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:\n"
    )

    def fetch_context(inputs):
        question = inputs["question"]
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            docs = retriever.get_relevant_documents(question)

        context_text = "\n\n".join(doc.page_content for doc in docs)
        context_sources = [
            f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
            for doc in docs
        ]

        formatted_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
        generated = generator(formatted_prompt, max_new_tokens=512)[0]["generated_text"]

        if "Answer:" in generated:
            answer = generated.split("Answer:")[-1].strip()
        else:
            answer = generated.strip()
        answer = answer.split("\n\n")[0].strip()

        return {
            "context": context_text,
            "context_sources": context_sources,
            "answer": answer,
        }

    print(f"✅ RAG pipeline ready. Using model: {model_name_display}\n")
    return fetch_context, model_name_display