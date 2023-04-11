import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI: Categorizing Essays", page_icon="📈")

st.title("Categorize Your Essays")
st.write("Categorize essays with my AI model, originally developed for tuition centers to categorize well-written student essays for easier reference.")
st.subheader("How It Works")
st.write(" It utilizes both LDA topic modelling and multi-class text classification techniques to accurately predict the category of an essay based on its title and content. The LDA topic modelling algorithm analyzes the key themes and topics present in the essay, while the multi-class text classification algorithm determines the most appropriate category for the essay based on these topics.")
st.write("This model has been trained and tuned to find the best performance based on a large dataset of essays from various fields.  ")

form = st.form(key="form1")
titleHere = form.text_input(label="Title")
essayHere = form.text_input(label="Essay  ")
submit = form.form_submit_button(label="Make Prediction")

from streamlit_app_modules import *
from gensim.corpora import Dictionary

@st.cache
def make_prediction(title, essay):
    essays = pd.DataFrame({
        'essay': [essay],
        'title': [title]
    })

    # clean data
    essays['num_words'] = essays['essay'].apply(lambda x: len(x.split()))
    essays['essay'] = essays['essay'].apply(lambda x: clean_text(x))
    essays['title'] = essays['title'].apply(lambda x: clean_text(x))

    # add nouns column
    essays['nouns'] = essays['essay'].apply(lambda x: [token for sent in x for token, pos in sent if pos in ['NN', 'NNS', 'NNP','NNPS']])

    # get only the words for the title column
    essays['title'] = essays['title'].apply(lambda x: [token for sent in x for token, pos in sent])
    essays['title'] = essays['title'].apply(lambda x: ' '.join(x))

    dictionary = Dictionary(essays['nouns'])
    topic_distributions = [lda_model.get_document_topics(dictionary.doc2bow(text)) for text in essays['nouns']]
    most_probable_topics = [max(topic_distributions[i], key=lambda x: x[1])[0] for i in range(len(topic_distributions))]
    essays['topic'] = most_probable_topics

    # get predictions
    essays['nouns'] = essays['nouns'].apply(lambda x: ' '.join(x))
    predictions = lr_model.predict(essays[['nouns', 'topic', 'title', 'num_words']])

    # get predictions category from df_mapping
    df_mapping = pd.DataFrame({
        'category': [0, 1, 2, 3, 4, 5, 6],
        'category_name': ['Art, Culture and Religion', 'Business and Economics', 'Health and Nutrition', 'History', 'Law, Crime and Punishment', 'Literature, Social Issues and Life', 'Science and Environment']
    })
    predictions = [df_mapping[df_mapping['category'] == prediction]['category_name'].values[0] for prediction in predictions]
    return predictions[0]

if submit:
    # make prediction from the input text
    category = make_prediction(titleHere, essayHere)
 
    # Display results of the NLP task
    # add space
    st.write("")
    st.write("")
    st.write("Your essay is most likely in:")
    st.subheader(f"{category}.")

# run this file with streamlit run streamlit_app.py
# C:\Users\jolen\anaconda3\python.exe -m streamlit run streamlit_app.py