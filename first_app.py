import nltk
from textblob import TextBlob
import cleantext
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud
from rake_nltk import Rake
from nltk import ne_chunk, pos_tag, word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tree import Tree
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import langdetect
from langdetect import detect,LangDetectException
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
def custom_visualization(option, df):
    sentiment_counts = df['analysis'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    if option == "Bar Chart":
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                     labels={'Sentiment': 'Sentiment', 'Count': 'Number of Tweets'},
                     title='Sentiment Analysis Summary')
    elif option == "Pie Chart":
        fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Distribution')
    elif option == "Histogram":
        fig = px.histogram(df, x='score', nbins=30, title='Sentiment Score Distribution')
    elif option == "Word Cloud":
        wordcloud_text = ' '.join(df['tweets'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        return
    elif option == "Keyword Distribution":
        all_keywords = df['tweets'].apply(extract_keywords)
        flat_keywords = [item for sublist in all_keywords for item in sublist]
        keyword_freq = pd.Series(flat_keywords).value_counts().head(10)
        st.write(keyword_freq)
        st.bar_chart(keyword_freq)
        return
    elif option == "Box Plot":
        fig = px.box(df, y='score', title='Sentiment Score Box Plot')
    elif option == "Line Chart":
        df['index'] = df.index
        fig = px.line(df, x='index', y='score', title='Sentiment Score Line Chart')
    elif option == "Heatmap":
        correlation = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation, annot=True, ax=ax)
        st.pyplot(fig)
        return
    elif option == "Scatter Plot":
        fig = px.scatter(df, x='score', y='analysis', title='Sentiment Score Scatter Plot')

    st.plotly_chart(fig)

def detect_topics(text_data, num_topics=5, num_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    data_vectorized = vectorizer.fit_transform(text_data)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(data_vectorized)
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [words[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {topic_idx+1}: {', '.join(topic_words)}")
    return topics


st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analyzer</h1>", unsafe_allow_html=True)
with st.expander("**Analyse the Text**"):
    text = st.text_input("# Enter the Text to Analyse ",placeholder="Some Text must be Entered")
    blob = TextBlob(text)
    st.divider()
    st.write("**Polarity Of The Entered Statement** ", round(blob.sentiment.polarity,2))
    st.divider()
    st.write("**Subjectivity Of The Entered Statement** ", round(blob.sentiment.subjectivity, 2))
    st.divider()

    try:
        lang = detect(text)
        st.write(f"**Language Of The Entered Text**: {lang}")
    except LangDetectException:
        st.write("**Language Of The Entered Text**: Could not detect the language")


    prep = st.text_input("**Enter the Clean Text to analyse**")
    if prep:
        st.write(cleantext.clean(prep, clean_all=False,extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True))

    if text:
        st.subheader("Additional Features")


        st.write("**Tokenization**")
        st.write("**Words**: ", pd.DataFrame(word_tokenize(text)))
        st.write("**Sentences**: ", pd.DataFrame(sent_tokenize(text)))


        st.write("**Part-of-Speech Tagging(POS)**")
        st.write(pos_tag(word_tokenize(text)))


        st.write("**Stemming**")
        stemmer = PorterStemmer()
        st.write([stemmer.stem(word) for word in word_tokenize(text)])


        st.write("**Lemmatization**")
        lemmatizer = WordNetLemmatizer()
        st.write([lemmatizer.lemmatize(word) for word in word_tokenize(text)])


        st.write("**Sentiment Analysis using VADER**")
        vader = SentimentIntensityAnalyzer()
        st.write(vader.polarity_scores(text))

        st.write("**Topic Modeling**")
        topics = detect_topics([text])
        for topic in topics:
            st.write(topic)



with st.expander("# Analyze a Excel (xlsx) file"):
    upload = st.file_uploader("Upload a File")

    def score(x):
        b1 = TextBlob(x)
        return round(b1.sentiment.polarity,2)

    def analyse(x):
        if x>=0.5:
            return "Positive Text 😊"
        elif x<= -0.5:
            return "Negative Text 😠"
        else:
            return "Neutral Text 😐"


    def extract_keywords(text):
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()


    def extract_entities(text):
        entities = []
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)
        tree = nltk.ne_chunk(tags)
        for subtree in tree:
            if isinstance(subtree, Tree):
                entity = " ".join([token for token, pos in subtree.leaves()])
                entities.append(entity)
        return entities

    if upload:
        with st.spinner("Analysing the File "):
            df = pd.read_excel(upload)
            if 'tweets' in df.columns:
                df['tweets'] = df['tweets'].astype(str).fillna('')
                df['score'] = df['tweets'].apply(score)
                df['analysis'] = df['score'].apply(analyse)
                st.write(df.head(10))
                st.divider()

                visual_option = st.selectbox(
                    "Select Custom Visualization",
                    ["Bar Chart", "Pie Chart", "Histogram", "Word Cloud", "Keyword Distribution", "Box Plot",
                     "Line Chart", "Heatmap", "Scatter Plot"]
                )
                custom_visualization(visual_option, df)


            @st.cache_data
            def conversion_df(df):
                return df.to_csv().encode('utf-8')

            csv = conversion_df(df)

            st.download_button(
                label='Download As CSV',
                data=csv,
                file_name="Sentiment.csv",
                mime = 'text/csv'

            )

with st.expander("Named Entity Recognition (NER)"):
    st.subheader("Named Entity Recognition (NER)")
    ner_text = st.text_area("Enter text for NER analysis", height=150,placeholder="Enter some text For NER")
    if st.button("Extract Entities"):
        entities = extract_entities(ner_text)
        st.write("Entities found:", pd.DataFrame(entities))
