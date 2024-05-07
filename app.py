from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings,format_docs
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from src.prompt import template
import os

load_dotenv()


# Flask initialization
app = Flask(__name__)

os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')

index_name = 'medical-bot'

# loding embedding model
embeddings = download_hugging_face_embeddings()

# loading existing index
vectorstore = PineconeVectorStore.from_existing_index(index_name,embeddings)

# Initializing LLM
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


custom_rag_prompt = PromptTemplate.from_template(template)

# retriever
retriever = vectorstore.as_retriever()

# RAG Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = rag_chain.invoke(input)
    print("Response : ", result)
    return str(result)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)





