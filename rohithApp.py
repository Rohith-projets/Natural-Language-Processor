import streamlit as st
import pandas as pd
import chardet
from textblob import TextBlob
import nltk
import os
from textblob.download_corpora import download_all
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier, MaxEntClassifier
import random

# Ensure necessary corpora are downloaded
if 'downloaded_corpora' not in st.session_state:
    @st.cache_resource
    def download_textblob_corpora():
        download_all()
        return True
    st.session_state['downloaded_corpora'] = download_textblob_corpora()

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
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = {}

# Sidebar file uploader
with st.sidebar:
    if st.session_state['downloaded_corpora']:
        st.success("TextBlob corpora loaded successfully!")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df, filename = read_file(uploaded_file)
        if df is not None:
            st.session_state["allData"][filename] = df
            st.success(f"File {filename} loaded successfully!")

# Main UI
st.title("Text Processing with TextBlob")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Operations", "View Data", "Delete Data"])

with tab1:
    if st.session_state["allData"]:
        selected_file = st.selectbox("Select a file", list(st.session_state["allData"].keys()))
        dataset = st.session_state["allData"].get(selected_file)
        
        st.subheader("Select Operation And Perform", divider='blue')
        col1, col2 = st.columns([1, 2], border=True)
        radio_options = col1.radio("Advanced Options", [
            "Extract Tags", "Extract Noun Phrases", "Sentiment Analysis", "Singularize",
            "Pluralize", "Lemmatize", "Definitions", "Spelling Correction", "Spell Check",
            "Word Counts", "N Grams", "Text Classification"
        ])
        
        selected_column = col2.selectbox("Select Column", dataset.columns)
        
        if radio_options == "N Grams":
            n_value = col2.number_input("Enter N value for N-grams", min_value=2, max_value=5, value=2)
        
        if radio_options == "Text Classification":
            st.warning("For text classification, your data should have at least 2 columns: one for text and one for labels.")
            label_column = col2.selectbox("Select Label Column", [col for col in dataset.columns if col != selected_column])
            train_size = col2.slider("Training set size (%)", 10, 90, 70)
            classifier_type = col2.selectbox("Select Classifier", ["Naive Bayes", "Decision Tree", "Maximum Entropy"])
            train_model = col2.checkbox("Train model")
            
            if train_model and col2.button("Train and Evaluate"):
                # Prepare data
                data = list(zip(dataset[selected_column].astype(str), dataset[label_column]))
                random.shuffle(data)
                split_idx = int(len(data) * train_size / 100)
                train_data = data[:split_idx]
                test_data = data[split_idx:]
                
                # Train classifier
                if classifier_type == "Naive Bayes":
                    classifier = NaiveBayesClassifier(train_data)
                elif classifier_type == "Decision Tree":
                    classifier = DecisionTreeClassifier(train_data)
                elif classifier_type == "Maximum Entropy":
                    classifier = MaxEntClassifier(train_data)
                
                # Store classifier in session state
                st.session_state['classifier'] = classifier
                
                # Evaluate
                accuracy = classifier.accuracy(test_data)
                col2.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Show informative features
                if classifier_type == "Naive Bayes":
                    col2.write("Most informative features:")
                    col2.write(classifier.show_informative_features(5))
            
            if 'classifier' in st.session_state:
                test_text = col2.text_area("Enter text to classify")
                if col2.button("Predict"):
                    if test_text:
                        result = st.session_state['classifier'].classify(test_text)
                        col2.success(f"Prediction: {result}")
        
        if col2.button("Apply", use_container_width=True, type='primary') and radio_options != "Text Classification":
            data = dataset.copy()
            with col2:
                if radio_options == "Extract Tags":
                    data[f"{selected_column} (Tags)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).tags)
                    st.session_state["processed_data"]["tags"] = data
                elif radio_options == "Extract Noun Phrases":
                    data[f"{selected_column} (Noun Phrases)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).noun_phrases)
                    st.session_state["processed_data"]["noun_phrases"] = data
                elif radio_options == "Sentiment Analysis":
                    data[f"{selected_column} (Sentiment)"] = data[selected_column].apply(lambda x: TextBlob(str(x)).sentiment)
                    st.session_state["processed_data"]["sentiment"] = data
                elif radio_options == "Singularize":
                    data[f"{selected_column} (Singularized)"] = data[selected_column].apply(
                        lambda x: ' '.join([word.singularize() for word in TextBlob(str(x)).words]))
                    st.session_state["processed_data"]["singularized"] = data
                elif radio_options == "Pluralize":
                    data[f"{selected_column} (Pluralized)"] = data[selected_column].apply(
                        lambda x: ' '.join([word.pluralize() for word in TextBlob(str(x)).words]))
                    st.session_state["processed_data"]["pluralized"] = data
                elif radio_options == "Lemmatize":
                    data[f"{selected_column} (Lemmatized)"] = data[selected_column].apply(
                        lambda x: ' '.join([word.lemmatize() for word in TextBlob(str(x)).words]))
                    st.session_state["processed_data"]["lemmatized"] = data
                elif radio_options == "Definitions":
                    data[f"{selected_column} (Definitions)"] = data[selected_column].apply(
                        lambda x: [word.definitions for word in TextBlob(str(x)).words])
                    st.session_state["processed_data"]["definitions"] = data
                elif radio_options == "Spelling Correction":
                    data[f"{selected_column} (Corrected)"] = data[selected_column].apply(
                        lambda x: str(TextBlob(str(x)).correct()))
                    st.session_state["processed_data"]["corrected"] = data
                elif radio_options == "Spell Check":
                    data[f"{selected_column} (Spell Check)"] = data[selected_column].apply(
                        lambda x: TextBlob(str(x)).spellcheck())
                    st.session_state["processed_data"]["spellcheck"] = data
                elif radio_options == "Word Counts":
                    data[f"{selected_column} (Word Count)"] = data[selected_column].apply(
                        lambda x: len(TextBlob(str(x)).words))
                    st.session_state["processed_data"]["word_counts"] = data
                elif radio_options == "N Grams":
                    data[f"{selected_column} ({n_value}-grams)"] = data[selected_column].apply(
                        lambda x: list(TextBlob(str(x)).ngrams(n_value)))
                    st.session_state["processed_data"]["ngrams"] = data
                
                st.subheader("Result", divider='blue')
                st.dataframe(data)

with tab2:
    if st.session_state["processed_data"]:
        selected_processed = st.selectbox("Select processed data to view", 
                                        list(st.session_state["processed_data"].keys()))
        st.dataframe(st.session_state["processed_data"][selected_processed])
    else:
        st.info("No processed data available. Perform operations in the 'Operations' tab first.")

with tab3:
    if st.session_state["allData"]:
        file_to_delete = st.selectbox("Select file to delete", 
                                    list(st.session_state["allData"].keys()))
        if st.button("Delete Selected File"):
            del st.session_state["allData"][file_to_delete]
            st.success(f"File {file_to_delete} deleted successfully!")
            # Also delete any processed data from this file
            keys_to_delete = [k for k in st.session_state["processed_data"] 
                            if k.startswith(file_to_delete)]
            for k in keys_to_delete:
                del st.session_state["processed_data"][k]
    else:
        st.info("No files available to delete.")
