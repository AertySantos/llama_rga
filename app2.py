from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
import os

DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_PATH = "models/llama-2-13b-chat.Q3_K_S.gguf"

# Template customizado para o prompt
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Carrega embeddings pré-treinados do Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Verifica a existência do FAISS database
if not os.path.exists(DB_FAISS_PATH):
    print(f"Erro: o caminho {DB_FAISS_PATH} não foi encontrado.")
    sys.exit(1)

# Carrega a base de dados FAISS
new_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Configurações do LLM
if not os.path.exists(MODEL_PATH):
    print(f"Erro: o modelo {MODEL_PATH} não foi encontrado.")
    sys.exit(1)

n_gpu_layers = -1
n_batch = 512

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    top_p=1,
    max_tokens=200,
    verbose=True,  
    n_ctx=4096,
)

# Configuração da chain de QA com RetrievalQA
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=new_db.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": set_custom_prompt()}
)

def main():
    
    while True:
        query = input("Input Prompt (Digite 'exit' para sair): ")
        if query.lower() == "exit":
            print("Exiting...")
            sys.exit(0)

        if not query.strip():
            print("Query vazia! Tente novamente.")
            continue

        try:
            result = retrieval_chain.run({"query": query})
            print("Resposta:", result)
        except Exception as e:
            print(f"Erro ao processar a consulta: {e}")

if __name__ == "__main__":
    main()
