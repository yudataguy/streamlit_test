from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from streamlit_chat import message

#test pr-commit 

pdf_loader = PyPDFLoader("./temporarystreetclosurefora.pdf")

pdf_file = pdf_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(pdf_file)

embeddings = OpenAIEmbeddings()

arcadia_search = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=arcadia_search.as_retriever(search_kwargs={"k": 1}),
)


def generate_response(prompt):
    """Generate a response from the user input."""
    return qa({"query": prompt})["result"]


st.title("ArcadiaGPT Test v1.0")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_input() -> str:
    """Get user input from the text input box."""
    input_text = st.text_input("What would you like to ask?", key="input")
    return input_text


user_input = get_input()


if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
