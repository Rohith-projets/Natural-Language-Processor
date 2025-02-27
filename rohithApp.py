import streamlit as st
import pandas as pd
import chardet
from textblob import TextBlob
import nltk
import os
from textblob.download_corpora import download_all

# Ensure necessary corpora are downloaded
@st.cache_resource
def download_textblob_corpora():
    download_all()
download_textblob_corpora()

# Function to detect file encoding
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer after reading
    return result['encoding']

# Function to read uploaded file
def read_file(uploaded_file):
    encoding = detect_encoding(uploaded_file)
    filename = uploaded_file.name
    
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format")
        return None
    
    return df, filename

# Initialize session state if not exists
if "allData" not in st.session_state:
    st.session_state["allData"] = {}

# Sidebar file uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df, filename = read_file(uploaded_file)
        if df is not None:
            st.session_state["allData"][filename] = df

# Main UI
st.title("Text Processing with TextBlob")

if st.session_state["allData"]:
    selected_file = st.selectbox("Select a file", list(st.session_state["allData"].keys()))
    dataset = st.session_state["allData"].get(selected_file)
    st.write("### Preview Data")
    st.dataframe(dataset.head())
    
    col1, col2 = st.columns([1, 2])
    radio_options = col1.radio("Advanced Options", [
        "Extract Tags", "Extract Noun Phrases", "Sentiment Analysis", "Singularize",
        "Pluralize", "Lemmatize", "Definitions", "Spelling Correction", "Spell Check",
        "Word Counts", "N Grams", "Text Classification"
    ])
    selected_column = col2.selectbox("Select Column", dataset.columns)
    
    if col2.button("Apply", use_container_width=True, type='primary'):
        data = dataset.copy()
        
        if radio_options == "Extract Tags":
            data[f"{selected_column} (Tags)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).tags)
        elif radio_options == "Extract Noun Phrases":
            data[f"{selected_column} (Noun Phrases)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).noun_phrases)
        elif radio_options == "Sentiment Analysis":
            data[f"{selected_column} (Sentiment)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).sentiment)
        elif radio_options == "Singularize":
            data[f"{selected_column} (Singularized)"] = data[selected_column].apply(lambda x: ' '.join([word.singularize() for word in TextBlob(str(x)).words]))
        elif radio_options == "Pluralize":
            data[f"{selected_column} (Pluralized)"] = data[selected_column].apply(lambda x: ' '.join([word.pluralize() for word in TextBlob(str(x)).words]))
        elif radio_options == "Lemmatize":
            data[f"{selected_column} (Lemmatized)"] = data[selected_column].apply(lambda x: ' '.join([word.lemmatize() for word in TextBlob(str(x)).words]))
        elif radio_options == "Definitions":
            data[f"{selected_column} (Definitions)"] = data[selected_column].apply(lambda x: [word.definitions for word in TextBlob(str(x)).words])
        elif radio_options == "Spelling Correction":
            data[f"{selected_column} (Corrected)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).correct())
        elif radio_options == "Spell Check":
            data[f"{selected_column} (Spell Check)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).spellcheck())
        elif radio_options == "Word Counts":
            data[f"{selected_column} (Word Count)"] = data[selected_column].apply(lambda x: len(TextBlob(str(x)).words))
        elif radio_options == "N Grams":
            data[f"{selected_column} (Bigrams)"] = data[selected_column].apply(lambda x: list(TextBlob(str(x)).ngrams(2)))
        elif radio_options == "Text Classification":
            st.warning("Text classification requires a trained model in TextBlob, which is not included by default.")
        
        st.dataframe(data)
