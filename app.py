import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import re

# Load Pegasus summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/pegasus-xsum")

summarizer = load_summarizer()

# Clean up LaTeX-style or noisy text
def clean_text(text):
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)   # Remove LaTeX commands with braces
    text = re.sub(r'\\[a-zA-Z]+', '', text)            # Remove LaTeX commands without braces
    text = re.sub(r'\s{2,}', ' ', text)                # Replace multiple spaces with one
    return text.strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_file:
        text += page.get_text()
    return clean_text(text)

# Split text into manageable chunks
def split_into_chunks(text, max_tokens=400):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Streamlit App UI
st.title("📄 Document Summarization Tool")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📃 Extracted Document Text")
    with st.expander("Show extracted text"):
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

    if st.button("📝 Generate Summary"):
        with st.spinner("🧠 Generating summary..."):
            chunks = split_into_chunks(text)
            partial_summaries = []

            for chunk in chunks:
                try:
                    truncated = " ".join(chunk.split()[:400])  # hard cap for safety
                    summary = summarizer(truncated, max_length=200, min_length=60, do_sample=False)
                    partial_summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    st.warning(f"⚠️ Skipped a chunk due to error: {e}")

            combined_summary = " ".join(partial_summaries)

            # Optional: Final re-summarization to improve coherence
            final_summary = summarizer(
                combined_summary[:1024],
                max_length=200,
                min_length=80,
                do_sample=False
            )[0]['summary_text']

            st.subheader("✅ Final Summary")
            st.write(final_summary)

