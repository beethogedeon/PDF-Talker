from datetime import datetime
import torch
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
import pinecone
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
import os
from typing import List, Union, Optional


def read_pdfs(pdf: str) -> List[Document]:
    """Extract the data from a list of URLs."""
    try:
        data = UnstructuredPDFLoader(pdf).load()

        return data
    except Exception as e:
        print("Could not load the data from the PDFs because : " + str(e))


def split_data(data: List[Document], chunk_size=1024, chunk_overlap=256) -> List[Document]:
    """Split the data into sentences."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        text_chunks = text_splitter.split_documents(data)

        return text_chunks
    except Exception as e:
        print("Could not split the data into sentences because : " + str(e))


def saving_in_vectorstore(data: List[Document], index_name: str | None = None, store="FAISS", embeddings_type="HF"):
    """Saving the data in the vectorstore."""
    try:
        if embeddings_type == "OPENAI":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()

        if store == "FAISS":
            vectorstore = FAISS.from_documents(data, embedding=embeddings)

            vectorstore.save_local("./vectordb/")

            return vectorstore
        elif store == "PINECONE":
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_api_env = os.environ.get('PINECONE_API_ENV')

            try:
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)
            except Exception as e:
                print("Could not init Pinecone because : " + str(e))

            try:
                # vectorstore = Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)
                vectorstore = Pinecone.from_documents(data, embeddings, index_name=index_name)

                return vectorstore
            except Exception as e:
                print("Could not create the vectorstore because : " + str(e))
        else:
            raise Exception("The vectorstore is not valid.")

    except Exception as e:
        print("Could not save the data in the vectorstore because : " + str(e))


def creating_chain(vectorstore: Union[FAISS, Pinecone], conversational_model="HuggingFace", model_temperature=0.7) -> ConversationalRetrievalChain:
    """Creating QA Chain."""
    try:

        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        qa_chain_prompt = PromptTemplate.from_template(template)

        if conversational_model == "OpenAI":
            llm = ChatOpenAI(temperature=model_temperature)
        else:
            llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.7, "max_length": 512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            condense_question_prompt=qa_chain_prompt
        )

        return conversation_chain

    except Exception as e:
        print("Error while creating QA Chain : " + str(e))


class Model:
    def __init__(self, pdfs_path: List[str] | None):
        self.urls = pdfs_path
        self.vectorstore: FAISS | Pinecone | None = None
        self.chain = None
        self.creation_date = None

    def train(self, urls: List[str] | None, store="FAISS"):
        """Train new Q/A chatbot with data from those urls"""

        # if os.environ['OPENAI_API_KEY'] is None:
        #    raise ValueError("OPENAI_API_KEY must be provided.")
        if store != "FAISS" and os.environ['PINECONE_API_KEY'] is None:
            raise ValueError("PINECONE_API_KEY must be provided.")
        else:
            try:
                if self.urls is None:
                    self.urls = urls

                if self.urls is None and urls is None:
                    raise ValueError("You must provide a list of URLs.")
                else:
                    data = read_pdfs(self.urls)
                    data = split_data(data)
                    if store == "FAISS":
                        self.vectorstore = saving_in_vectorstore(data)
                    else:
                        self.vectorstore = saving_in_vectorstore(data, index_name="chatbot", store=store)

                    self.chain = creating_chain(self.vectorstore)
                    self.creation_date = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

                    self.save()

            except Exception as e:
                print("Error while training the model : " + str(e))

    def answer(self, query: str) -> str:
        """Answering the question."""

        try:
            result = self.chain({"query": query}, return_only_outputs=True)
            result = result["result"]

            return result
        except Exception as e:
            print("Error while answering the question : " + str(e))

    def save(self):
        """Saving the model."""
        try:
            # Check if there's a previous latest model, and rename it
            if os.path.exists("models/latest.pt"):
                os.rename("models/latest.pt", f"models/model_{self.creation_date}.pt")
                
            # torch.save(self, f"models/latest.pt")
        except Exception as e:
            print("Error while saving the model : " + str(e))


def load_model(model_path="latest.pt"):
    try:
        model = torch.load(f"models/{model_path}")

        return model
    except Exception as e:
        print("Error while loading the model :" + str(e))


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)