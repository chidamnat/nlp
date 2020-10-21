import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
import time

st.title('Sample topic modeling app')
DATA_URL = 'https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json'
st.sidebar.header("User Input Parameters")
n_rows = st.sidebar.slider("number of records to read", 100, 100000)
num_of_topics = st.sidebar.slider("Number of topics", 1, 50)
column_to_be_used = None

# w = st.file_uploader("Upload a dataset", type="csv")


@st.cache
def load_data(nrows, file_path):
# def load_data(nrows, file_path=DATA_URL):

    if file_path:
        return pd.read_csv(file_path, nrows=nrows)
    # extension = file_path.split('.')[-1]
    # if extension == 'json':
    #     df = pd.read_json(file_path)
    #     return df.head(nrows)
    # return pd.read_csv(file_path, nrows=nrows)

    # st.write(df.target_names.unique().tolist())


def get_data():
    # time.sleep(10)
    w = st.file_uploader("Upload a dataset", type="csv")
    data_load_state = st.text('loading data ...')
    # import pdb; pdb.set_trace()
    data = load_data(n_rows, w)
    # data = load_data(n_rows)
    data_load_state = st.text('Done! (using st.cache)')

    st.subheader('Raw data')
    st.write(f'number of records = {len(data)}')
    # st.write(data.head())
    return data


def preprocess_data(raw_data, column):
    processed_data = raw_data[column]
    my_stop_words = ENGLISH_STOP_WORDS.union(['from', 'subject', 're', 'edu', 'use'])
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2),
                                 # stop_words='english',
                                 stop_words=my_stop_words,
                                 min_df=10,
                                 max_df=0.5)
    tfidf = tfidf_vect.fit_transform(processed_data)
    return tfidf, tfidf_vect


def build_model(X_train, num_of_topics, model_name='LatentDirichletAllocation'):
    if model_name == 'LatentDirichletAllocation':
        lda = LatentDirichletAllocation(n_components=num_of_topics,
                                        learning_method="batch",
                                        random_state=0)
        lda.fit_transform(X_train)
        return lda


def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        st.write("\nTopic #%d:" % topic_idx)
        st.write(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))





def main(column_to_be_used):
    if not column_to_be_used:
        raw_data = get_data()
    # import pdb; pdb.set_trace()
    column_to_be_used = st.sidebar.selectbox("Column to be used", raw_data.columns[raw_data.dtypes == np.object])
    st.write(raw_data.head())
    if st.button('Train model'):
        prepared_data, tfidf_vect = preprocess_data(raw_data, column_to_be_used)
        lda = build_model(prepared_data, num_of_topics)
        print_topics(lda, tfidf_vect, n_top_words=10)
        vis = pyLDAvis.sklearn.prepare(lda, prepared_data, tfidf_vect, mds='tsne')
        if st.button('View Topics'):
            pyLDAvis.show(vis)
    # st.pyplot(vis)
    #
    # st.write(lda_result)


if __name__ == '__main__':
    main(column_to_be_used)