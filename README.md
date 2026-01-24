📘 AI-Based Document Search and Knowledge Retrieval System
🔍 Project Overview

This project is an AI-powered document search and question-answering system built using Retrieval Augmented Generation (RAG).
It allows users to upload documents and ask questions based on their content. The system retrieves relevant information and generates accurate answers using Google Gemini AI.

🚀 Features

📂 Upload documents (PDF, PPT, TXT)

🔍 Semantic document search

🤖 AI-powered answers using Gemini

🧠 Vector-based similarity search

💬 Chat-style interface

⚡ Fast and user-friendly UI using Streamlit

🛠️ Technologies Used
Technology	Purpose
Python	Backend logic
Streamlit	Web interface
LangChain	RAG pipeline
Google Gemini API	AI model
FAISS / Vector Search	Similarity search
python-pptx	PPT reading
PyPDF	PDF text extraction
dotenv	Environment variable handling



📁 Project Structure
AI-Document-Search-RAG/


app.py     # Main application

requirements.txt         # Required libraries

.gitignore               # Ignored files

README.md                # Project documentation

.env (not uploaded)      # API key



⚙️ Installation & Setup



1️⃣ Clone the Repository
git clone https://github.com/lathika10/AI-Document-Search-RAG.git
cd AI-Document-Search-RAG

2️⃣ Create Virtual Environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Gemini API Key

Create a .env file:

GOOGLE_API_KEY=your_api_key_here


⚠️ Do NOT upload .env to GitHub.

5️⃣ Run the Application
streamlit run final_rag_project.py


Open browser:

http://localhost:8501

🧠 How It Works

User uploads a document

Text is extracted and split into chunks

Chunks are converted into embeddings

User asks a question

Relevant text is retrieved

Gemini AI generates the final answer

📌 This approach is called Retrieval-Augmented Generation (RAG).

💡 Example Use Cases

Academic document analysis

Research paper summarization

Study material question answering

Knowledge retrieval system

AI assistant for documents

🔐 Security

API keys are stored using .env

.env is excluded using .gitignore

No sensitive data uploaded to GitHub
