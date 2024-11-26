# Importação de módulos necessários para processamento de linguagem natural
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Caminho para onde os dados processados serão salvos
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "dados/"

# Carregamento dos dados de um arquivo XML não estruturado
#loader = UnstructuredXMLLoader("data/ptwiki-20230920-pages-articles.xml")
loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)

data = loader.load()  # minimizar o problema  ctx512

print(data[:1])

# Divisão do texto em partes menores para facilitar o processamento
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Impressão do número de partes resultantes após a divisão do texto
print(len(text_chunks))

# Geração de embeddings usando o modelo Hugging Face 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

# Conversão dos pedaços de texto em embeddings e criação de um índice de pesquisa FAISS
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Salvamento dos dados processados, incluindo o índice FAISS, em um diretório específico
docsearch.save_local(DB_FAISS_PATH)
