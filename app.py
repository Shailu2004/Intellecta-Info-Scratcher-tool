import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to create a conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to fetch and clean text from the URL
def get_url(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        data = loader.load()
        cleaned_lines = []

        for ch in data:
            page_content = ch.page_content.strip()
            if page_content:
                lines = [line.strip() for line in page_content.splitlines() if line.strip()]
                cleaned_lines.extend(lines)

        text = " ".join(cleaned_lines)
        return text
    except Exception as e:
        st.error(f"Error loading content from the URL: {e}")
        return None


# Function to process user input and get a response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.header("Answer: ")
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error during question processing: {e}")



# Main function for the Streamlit app
def main():
    st.title("Intellecta: Info Scratcher Tool 📈")
    st.sidebar.title("Website URL's")
    
    url = st.sidebar.text_input("URL")
    question = st.text_input("Question")
    st.sidebar.header(" About Model")
    st.sidebar.success("""
                       Intellecta an AI-powered tool is designed to assist users in efficiently extracting 
                       information from any URL-based content. Built using LangChain, FAISS, and 
                       Google Generative AI, the model allows users to input a URL from various 
                       sources—news articles, blogs, research papers, or any web content—and ask 
                       detailed questions. """)
    st.sidebar.header("Features")
    st.sidebar.write(" Versatile URL Parsing")
    st.sidebar.write("Embeddings and Vector Search")
    st.sidebar.write("Conversational QA")
    st.sidebar.write("Text Chunking")
   
    # Add a button to submit URL and question
    if st.button("Submit"):
        if url and question:
            text = get_url(url)
            if text:
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                user_input(question)
        else:
            st.warning("Please enter both a URL and a question.")


# Run the Streamlit app
if __name__ == "__main__":
    main()
