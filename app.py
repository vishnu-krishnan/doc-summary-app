import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from io import BytesIO

# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_file:
        text += page.get_text()
    return text

# Chunk large text into summarizable segments
def split_into_chunks(text, max_tokens=1000):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Streamlit UI
st.title("📄 Document Summarization Tool")
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Document Text")
    with st.expander("Show extracted text"):
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            chunks = split_into_chunks(text)
            summary = ""
            for chunk in chunks:
                summarized = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                summary += summarized[0]['summary_text'] + " "

            st.subheader("📝 Summary")
            st.write(summary.strip())
