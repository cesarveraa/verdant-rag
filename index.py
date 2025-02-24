import os
import time
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Configurar la clave de API y la región de Pinecone
api_key = "pcsk_5bpf1j_3NvVtLhnctqAMsivZy77kuDDdpoAa2CArjnFuNojf2xuX9ZABqResibrbExzPkz"
index_name = "agriculture-index"

# Inicializar el cliente de Pinecone
pc = Pinecone(api_key=api_key)

# Crear o conectar al índice en Pinecone
if index_name in pc.list_indexes():
    print(f"El índice '{index_name}' ya existe en Pinecone.")
    pinecone_index = pc.Index(index_name)
else:
    print(f"Creando un nuevo índice '{index_name}' en Pinecone...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    pinecone_index = pc.Index(index_name)
    print(f"Índice '{index_name}' creado correctamente en Pinecone.")

# Ruta al archivo PDF
pdf_path = 'libro1.pdf'

# Extraer texto del PDF
text = ""
with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""

# Dividir el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
    length_function=len,
)
chunks = text_splitter.split_text(text)

# Convertir los fragmentos a objetos Document
documents = [Document(page_content=chunk) for chunk in chunks]

# Cargar el modelo de embeddings locales con Sentence Transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Crear una clase de Embeddings compatible con Langchain
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embedding_model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        return embedding_model.encode([text], convert_to_tensor=False).tolist()[0]

embedding = LocalEmbeddings()

# Indexar los documentos en Pinecone
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embedding,
    text_key="text"
)

vectorstore.add_documents(documents)

print("Indexación con Pinecone completada correctamente usando embeddings locales.")
