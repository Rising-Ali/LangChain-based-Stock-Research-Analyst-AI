# LangChain-based-Stock-Research-Analyst-AI
A LangChain-powered AI Stock Research Analyst that fetches articles from URLs, chunks the text, creates embeddings using Hugging Face models, stores them in FAISS vector database, and answers questions based on the content.

 🚀 Features
- 📥 **URL Loader** – Fetches article content directly from provided links.  
- ✂️ **Text Chunking** – Splits large articles into smaller, LLM-friendly chunks.  
- 🧠 **Embeddings Generation** – Creates semantic embeddings using **Hugging Face models**.  
- 📦 **FAISS Vector Database** – Efficient similarity search for large text datasets.  
- 💬 **Question Answering** – Responds to queries based on the processed article data.  


 🛠️ Tech Stack
- **Python 3.9+**
- **LangChain** – Orchestration framework for LLM apps  
- **Hugging Face** – Free embedding models  
- **FAISS** – Fast Approximate Nearest Neighbor search  
- **BeautifulSoup4 / Requests** – Web scraping for articles  

 How It Works
Load URLs – The app takes your list of article links.
Extract & Clean – HTML is scraped and converted into clean text.
Chunking – Splits text into small, overlapping parts for better embedding accuracy.
Embeddings – Uses a Hugging Face transformer model to create embeddings.
Vector Storage – Saves these embeddings in a FAISS vector DB for fast retrieval.
Ask & Answer – You ask a question, the system finds the most relevant chunks, and the LLM generates an answer.
