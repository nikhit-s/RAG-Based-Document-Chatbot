# Multi-Document Chatbot

An AI-powered chatbot that can process multiple documents and answer questions by retrieving information from the most relevant document(s).

## Features

- Upload multiple documents (PDF, DOCX, DOC, TXT)
- Chat interface for asking questions
- AI-powered responses based on document content
- Source document references for answers
- Conversation history

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```
4. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload one or more documents (PDF, DOCX, DOC, or TXT)
3. Click "Process Documents" to analyze the documents
4. Start asking questions in the chat interface

## How It Works

1. The application processes uploaded documents and splits them into manageable chunks
2. It creates vector embeddings for each chunk using OpenAI's embeddings
3. When you ask a question, the system:
   - Finds the most relevant document chunks
   - Uses GPT-3.5-turbo to generate an answer based on the relevant chunks
   - Returns the answer along with source references

## Note

- Your API key is only stored in the current session and is not saved anywhere
- The application processes documents locally in your browser
- For large documents, processing may take some time
