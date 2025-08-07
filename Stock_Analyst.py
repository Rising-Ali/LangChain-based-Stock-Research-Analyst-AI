%pip install --upgrade --quiet  sentence_transformers langchain-community openai langchain unstructured
import os
import pickle
import langchain, langchain_community
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from google.colab import userdata

# Set OpenAI API Key
my_key = userdata.get("OpenAI_API_KEY")
os.environ["OPENAI_API_KEY"] = my_key

# Load data from URLs
loaders = UnstructuredURLLoader(urls = [
    "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
])
data = loaders.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=70
)
docs = text_splitter.split_documents(data)

# Create embeddings
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Create and save vector index
vectorindex_openai = FAISS.from_documents(docs, hf)
file_path = "vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vectorindex_openai, f)

# Load vector index
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorindex_openai = pickle.load(f)

# Initialize LLM and chain
llm = ChatOpenAI(
    model_name="deepseek/deepseek-chat-v3-0324:free",
    temperature=1
)
chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorindex_openai.as_retriever())

# Example query (you can change this)
query = "Which 2 Stocks have the most ROI based on Provided Data Set"
langchain.debug = True
result = chain({"question": query})
print(result)