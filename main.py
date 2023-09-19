import streamlit as st
from utils import Model, load_model

# Create a Streamlit app
st.title("PDF Chatbot")

# Sidebar to upload PDFs and configure the chatbot
st.sidebar.header("Chatbot Configuration")
load_existing_model = st.sidebar.checkbox("Load Existing Model")

# Initialize the chatbot model
model = None
existing_model_path = None

if load_existing_model:
    existing_model_path = st.sidebar.file_uploader("Select Existing Model File (.pt)", type=["pt"])

if existing_model_path:
    model = load_model(existing_model_path)
    st.sidebar.success("Existing model loaded successfully!")

if not load_existing_model:
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    store_type = st.sidebar.selectbox("Select Vector Store Type", ["FAISS", "PINECONE"])
    train_button = st.button("Train")

    if train_button and uploaded_pdfs:
        pdf_paths = [pdf.name for pdf in uploaded_pdfs]
        model = Model(pdf_paths)
        model.train(pdf_paths, store_type)
        st.sidebar.success("Chatbot model trained and ready!")

# Main content area
st.header("Ask the Chatbot")
user_question = st.text_input("Enter your question:")
if user_question and model:
    answer = model.answer(user_question)
    st.markdown(f"**Chatbot:** {answer}")

# Save and reload the model
if model:
    if st.sidebar.button("Save Model"):
        model.save()
        st.sidebar.success("Model saved successfully!")
