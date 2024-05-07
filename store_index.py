from src.helper import load_pdf,download_hugging_face_embeddings,text_split
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = 'medical-bot'

vectorstore = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)