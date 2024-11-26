from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import sys


DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
    
# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

new_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

n_gpu_layers = -1
n_batch = 512

llm = LlamaCpp(
    model_path="models/llama-2-13b-chat.Q3_K_S.gguf",
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    top_p=1,
    max_tokens=200,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096,
)

qa = ConversationalRetrievalChain.from_llm(
    llm, retriever = new_db.as_retriever(search_kwargs={'k': 3}))

while True:
    chat_history = []
    query = input(f"Input Prompt: ")
    if query == 'exit':

        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa.invoke({"question": query, "chat_history": chat_history})

    print("Response: ", result['answer'])
