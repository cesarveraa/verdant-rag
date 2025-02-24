import os
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Configurar la clave de API de SambaNova
os.environ["SAMBANOVA_API_KEY"] = "f3612483-e9ec-4409-a019-2e4081c0c575"

# Configurar la clave de API y la región de Pinecone
api_key = "pcsk_5bpf1j_3NvVtLhnctqAMsivZy77kuDDdpoAa2CArjnFuNojf2xuX9ZABqResibrbExzPkz"
index_name = "agriculture-index"

# Inicializar el cliente de Pinecone y conectar al índice
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index(index_name)

# Cargar el modelo de embeddings locales con Sentence Transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embedding_model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        return embedding_model.encode([text], convert_to_tensor=False).tolist()[0]

embedding = LocalEmbeddings()

# Cargar el vectorstore de Pinecone
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embedding,
    text_key="text"
)

# Configurar el recuperador semántico
semantic_chunk_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Configurar el cliente de SambaNova
client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

# Función para interactuar con Llama 3.3
def query_llama_sambanova(prompt: str, context: str = "") -> str:
    response = client.chat.completions.create(
        model="Meta-Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ],
        temperature=0.1,
        top_p=0.1
    )
    return response.choices[0].message.content

# Inicializar FastAPI
app = FastAPI()

class QuestionRequest(BaseModel):
    questions: List[str]

# Endpoint para consultas
@app.post("/ask")
def ask_questions(request: QuestionRequest):
    responses = []
    for question in request.questions:
        retrieved_docs = semantic_chunk_retriever.get_relevant_documents(question)
        
        if retrieved_docs:
            combined_context = "\n".join([doc.page_content for doc in retrieved_docs])
            result = query_llama_sambanova(question, combined_context)
            
            responses.append({
                "question": question,
                "response": result,
                "chunks": [doc.page_content for doc in retrieved_docs]
            })
        else:
            responses.append({
                "question": question,
                "response": "No se recuperaron chunks relevantes.",
                "chunks": []
            })
    
    return {"responses": responses}
