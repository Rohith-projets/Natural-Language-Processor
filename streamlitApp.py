col1, col2 = st.columns([1, 2],border=True)
    
      radio_options = col1.radio("Advanced Options", [
          "Extract Tags", "Extract Noun Phrases", "Sentiment Analysis", "Singularize",
          "Pluralize", "Lemmatize", "Definitions", "Spelling Correction", "Spell Check",
          "Word Counts", "N Grams", "Text Classification"
      ])
      
      selected_column = col2.selectbox("Select Column", self.dataset.columns)
      
      if col2.button("Apply", use_container_width=True, type='primary'):
          data = self.dataset.copy()
          
          if radio_options == "Extract Tags":
              data[f"{selected_column} (Tags)"] = data[selected_column].apply(lambda x: TextBlob(x).tags)
          
          elif radio_options == "Extract Noun Phrases":
              data[f"{selected_column} (Noun Phrases)"] = data[selected_column].apply(lambda x: TextBlob(x).noun_phrases)
          
          elif radio_options == "Sentiment Analysis":
              data[f"{selected_column} (Sentiment)"] = data[selected_column].apply(lambda x: TextBlob(x).sentiment)
          
          elif radio_options == "Singularize":
              data[f"{selected_column} (Singularized)"] = data[selected_column].apply(lambda x: ' '.join([word.singularize() for word in TextBlob(x).words]))
          
          elif radio_options == "Pluralize":
              data[f"{selected_column} (Pluralized)"] = data[selected_column].apply(lambda x: ' '.join([word.pluralize() for word in TextBlob(x).words]))
          
          elif radio_options == "Lemmatize":
              data[f"{selected_column} (Lemmatized)"] = data[selected_column].apply(lambda x: ' '.join([word.lemmatize() for word in TextBlob(x).words]))
          
          elif radio_options == "Definitions":
              data[f"{selected_column} (Definitions)"] = data[selected_column].apply(lambda x: [word.definitions for word in TextBlob(x).words])
          
          elif radio_options == "Spelling Correction":
              data[f"{selected_column} (Corrected)"] = data[selected_column].apply(lambda x: TextBlob(x).correct())
          
          elif radio_options == "Spell Check":
              data[f"{selected_column} (Spell Check)"] = data[selected_column].apply(lambda x: TextBlob(x).spellcheck())
          
          elif radio_options == "Word Counts":
              data[f"{selected_column} (Word Count)"] = data[selected_column].apply(lambda x: len(TextBlob(x).words))
          
          elif radio_options == "N Grams":
              data[f"{selected_column} (Bigrams)"] = data[selected_column].apply(lambda x: list(TextBlob(x).ngrams(2)))
          
          elif radio_options == "Text Classification":
              st.warning("Text classification requires a trained model in TextBlob, which is not included by default.")
          
          col2.dataframe(data)
