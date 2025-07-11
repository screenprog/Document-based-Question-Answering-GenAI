import os
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
import docx
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
load_dotenv()

client = chromadb.PersistentClient(path='Q&A')
collection_name = 'collection'
embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        model_name="models/embedding-001",
        api_key=GOOGLE_API_KEY
    )

def get_available_documents():
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    docs_metadata = collection.get()["metadatas"]
    if not docs_metadata:
        return []
    return set(meta["filename"] for meta in docs_metadata)

def create_vector_store_from_files(files: list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    metadatas = []
    for file in files:
        text = ""
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            pages = pdf_reader.pages
            for page in pages:
                text += page.extract_text()

        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            paragraphs = doc.paragraphs
            for paragraph in paragraphs:
                text += paragraph.text

        docs = text_splitter.split_text(text)
        for doc in docs:
            documents.append(doc)
            metadatas.append({"filename": file.name})


    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]
    for i in tqdm(
            range(0, len(documents), 512), desc="Adding documents", unit_scale=512
    ):
        collection.add(
            ids=ids[i: i + 512],
            documents=documents[i: i + 512],
            metadatas=metadatas[i: i + 512],
        )
    return collection.count() - count
