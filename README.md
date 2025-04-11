# 📚 MultiDoc-Bot 🤖

This project is a **Streamlit-based chatbot** that allows users to upload one or more PDF files and interactively ask questions about the content inside them. It leverages **Google's Gemini models**, **FAISS vector store**, and **LangChain** to extract, embed, and query information intelligently.

---

## 🚀 Features

- Upload multiple PDFs at once
- Extracts and processes all text from uploaded PDFs
- Uses chunking and embeddings to understand and store content
- Answers user questions using Google's Gemini models
- Stores processed PDFs in a FAISS vector store for efficient retrieval
- Simple and intuitive web interface with Streamlit

---

## 🧠 Tech Stack

| Technology         | Purpose                                  |
|--------------------|-------------------------------------------|
| **Streamlit**      | Frontend interface for interaction        |
| **PyPDF2**         | PDF parsing and text extraction          |
| **LangChain**      | Chaining components for LLMs and embeddings |
| **Google Generative AI** | Embeddings and Gemini model for answers |
| **FAISS**          | Vector storage for similarity search     |
| **dotenv**         | Secure API key management via `.env`     |

---

## 📂 Folder Structure

```bash
.
├── app.py              # Main Streamlit app
├── faiss_index/        # Vector index storage (auto-created)
├── .env                # Stores your GOOGLE_API_KEY
├── requirements.txt    # Python dependencies
└── README.md           # You're here!
```

---

## 🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/ajithkumarajii/MultiDoc-Bot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up API key**

Create a `.env` file and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🧪 How to Use

1. Run the app:

```bash
streamlit run app.py
```

2. In the web interface:
   - Upload one or more PDF files via the sidebar.
   - Click "Process PDFs" to extract and embed the content.
   - Ask any question related to the PDF content in the input box.

---

## ❓ How it Works

1. **PDF Upload**: User uploads PDFs which are read and parsed using `PyPDF2`.
2. **Chunking**: Text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**: Each chunk is converted to a vector using `GoogleGenerativeAIEmbeddings`.
4. **Storage**: The embeddings are stored in a FAISS vector store for fast retrieval.
5. **QA Chain**: When a user asks a question, relevant chunks are retrieved and passed to a Gemini model via LangChain to generate an answer.

---

## 🧑‍💻 Author

**Ajithhh**  
