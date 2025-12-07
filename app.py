import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""  

# ----------------------------
# Standard imports
# ----------------------------
import io
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# Keras/TF (used for your local BiLSTM model)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# NLP utils
from nltk.stem import PorterStemmer

# Transformers (PyTorch-only because TRANSFORMERS_NO_TF=1)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ----------------------------
# Custom Attention Layer
# ----------------------------
class attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(attention, self).get_config()
        config.update({'return_sequences': self.return_sequences})
        return config

# ----------------------------
# Preprocessing helpers
# ----------------------------
space_pattern = r'\s+'
giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                   r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = r'@[\w\-]+'
emoji_regex = r'&#[0-9]{4,6};'

def preprocess(text_string: str) -> str:
    if not isinstance(text_string, str):
        return ""
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(r'\bRT\b', '', parsed_text)
    parsed_text = re.sub(emoji_regex, '', parsed_text)
    parsed_text = re.sub('…', '', parsed_text)
    return parsed_text.strip()

def preprocess_clean(text_string: str, remove_hashtags=True, remove_special_chars=True) -> str:
    text_string = preprocess(text_string)
    parsed_text = text_string.lower()
    parsed_text = re.sub("'", "", parsed_text)
    parsed_text = re.sub(":", "", parsed_text)
    parsed_text = re.sub(",", "", parsed_text)
    parsed_text = re.sub("&amp;", "", parsed_text)
    if remove_hashtags:
        parsed_text = re.sub(r'#[\w\-]+', '', parsed_text)
    if remove_special_chars:
        parsed_text = re.sub(r'(\!|\?)+', '', parsed_text)
    return parsed_text.strip()

def strip_hashtags(text: str) -> str:
    text_proc = preprocess_clean(text, remove_hashtags=False, remove_special_chars=True)
    hashtags = re.findall(r'#[\w\-]+', text_proc)
    for tag in hashtags:
        cleantag = tag[1:]
        text_proc = re.sub(re.escape(tag), cleantag, text_proc)
    return text_proc

# Stemming
stemmer = PorterStemmer()
def stemming(text: str):
    if not text:
        return []
    return [stemmer.stem(t) for t in text.split()]

# ----------------------------
# App settings & dataset
# ----------------------------
st.set_page_config(page_title="Tweet Tone Triage (4T)", layout="wide")
DATA_PATH = Path("labeled_data.csv")

@st.cache_data
def load_dataset(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read dataset {path}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Dataset not found at {path}. Some pages will show limited information.")
        return pd.DataFrame()

df = load_dataset(DATA_PATH)

# ----------------------------
# Load HuggingFace pipeline (PyTorch)
# ----------------------------
@st.cache_resource
def load_hf_pipeline(model_name="ctoraman/hate-speech-bert"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
        # Try to extract id2label mapping if present
        id2label = getattr(model.config, "id2label", None)
        return pipe, id2label
    except Exception as e:
        # Return error details so caller can show message
        return None, str(e)

hf_pipe, hf_id2label = load_hf_pipeline()

# ----------------------------
# Load local models (cached)
# ----------------------------
@st.cache_resource
def load_local_models():
    messages = []
    SFD_model = None
    LR_model = None
    RF_model = None
    DT_model = None
    SVM_model = None

    keras_path = Path("One-layer_BiLSTM_without_dropout.keras")
    lr_path = Path("LR_model.pkl")
    rf_path = Path("Random_Forest_Model.pkl")
    dt_path = Path("Decision_Tree_Model.pkl")
    svm_path = Path("SVM_model.pkl")

    # Keras model
    if keras_path.exists():
        try:
            SFD_model = load_model(str(keras_path), custom_objects={'attention': attention})
        except Exception as e:
            messages.append(f"Failed to load Keras model: {e}")
    else:
        messages.append(f"Keras model file not found: {keras_path}")

    # Pickled sklearn models
    for p, name in [(lr_path, "LR"), (rf_path, "RF"), (dt_path, "DT"), (svm_path, "SVM")]:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                if name == "LR":
                    LR_model = obj
                elif name == "RF":
                    RF_model = obj
                elif name == "DT":
                    DT_model = obj
                elif name == "SVM":
                    SVM_model = obj
            except Exception as e:
                messages.append(f"Failed to load {name} model from {p}: {e}")
        else:
            messages.append(f"{name} model file not found: {p}")

    return SFD_model, LR_model, RF_model, DT_model, SVM_model, messages

SFD_model, LR_model, RF_model, DT_model, SVM_model, model_load_messages = load_local_models()
if model_load_messages:
    st.sidebar.warning("Model load notes: " + "; ".join(model_load_messages[:4]))

# ----------------------------
# Sidebar menu (Option A: selectbox style)
# ----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Tweet Tone Triage Technique (4T): A Secured Federated Deep Learning Approach",
        options=["Data Acquisition", "Data Exploration", "Data Classes Balancing", "Data Preparation",
                 "ML Model Selection", "Try The Model", "About", "Contact"],
        icons=["house","cloud", "list", "gear", "graph-up", "briefcase", "bxs-robot", "info","envelope"],
        menu_icon="cast",
        default_index=5,
        orientation="vertical"
    )

# ----------------------------
# Pages
# ----------------------------
if selected == "Data Acquisition":
    st.title("Hate Speech and Offensive Language Dataset")
    st.write("""
    This dataset contains data related to hate speech and offensive language.
    Davidson et al. (2017) introduced a crowdsourced dataset classifying tweets into:
    0 = hate speech, 1 = offensive language, 2 = neither.
    """)
    st.markdown("---")

elif selected == "Data Exploration":
    st.title("Loading and Previewing the Dataset")
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Information", "Dataset Description", "Dataset Overview", "Missing values"])
    with tab1:
        st.subheader('Dataset Information')
        buffer = io.StringIO()
        if not df.empty:
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        else:
            st.info("No dataset loaded.")
    with tab2:
        st.subheader('Dataset Columns Description')
        if not df.empty:
            st.write(df.describe(include='all'))
        else:
            st.info("No dataset loaded.")
    with tab3:
        st.subheader('Dataset Overview (Before Preprocessing)')
        if not df.empty:
            st.write(df.head(10))
        else:
            st.info("No dataset loaded.")
    with tab4:
        st.subheader("Missing values in each column:")
        if not df.empty:
            st.write(df.isnull().sum())
        else:
            st.info("No dataset loaded.")
    st.markdown("---")

elif selected == "Data Classes Balancing":
    st.title("Understanding Class Distribution")
    if 'class' in df.columns:
        df_fig = df['class']
        class_labels = ['Hate Speech', 'Offensive Language', 'Neither']
        class_counts = df_fig.value_counts().reindex([0,1,2], fill_value=0)
        tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
        with tab1:
            st.subheader('Distribution of Classes (Bar Chart)')
            bar_fig = px.bar(x=class_labels, y=class_counts.values,
                             labels={'x':'Class','y':'Frequency'}, title='Distribution of Classes')
            st.plotly_chart(bar_fig, use_container_width=True)
        with tab2:
            st.subheader('Proportion of Classes (Pie Chart)')
            pie_fig = go.Figure(data=[go.Pie(labels=class_labels, values=class_counts.values, hole=0.3,
                                            pull=[0, 0.1, 0], textinfo='label+percent')])
            pie_fig.update_layout(title_text="Distribution of Classes (Pie Chart)", showlegend=True)
            st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("Column 'class' not found in dataset.")
    st.markdown("---")

elif selected == "Data Preparation":
    st.title("Dataset Preprocessing")
    st.write("""
    Preprocessing summary:
    - Remove URLs, mentions, RT tokens, emojis.
    - Lowercasing and light punctuation removal.
    - Normalize hashtags into words.
    - Stemming.
    - Tokenizing and padding to max length (100).
    """)
    tab1, tab2, tab3, tab4 = st.tabs(["Tweets Before Preprocessing", "Cleaned Tweets", "Stemmed Tweets", "Tokenized Tweets"])
    with tab1:
        st.subheader('Tweets Before Preprocessing')
        if 'tweet' in df.columns:
            st.write(df['tweet'].head(50))
        else:
            st.info("No 'tweet' column found.")
    with tab2:
        st.subheader('Tweets After Cleaning')
        if Path("cleaned_tweets.csv").exists():
            st.write(pd.read_csv("cleaned_tweets.csv").head(50))
        else:
            st.info("cleaned_tweets.csv not found.")
    with tab3:
        st.subheader('Tweets After Stemming')
        if Path("stemmed_tweets.csv").exists():
            st.write(pd.read_csv("stemmed_tweets.csv").head(50))
        else:
            st.info("stemmed_tweets.csv not found.")
    with tab4:
        st.subheader('Tweets After Tokenization')
        if Path("Tokenized_Padded_tweets.csv").exists():
            st.write(pd.read_csv("Tokenized_Padded_tweets.csv").head(50))
        else:
            st.info("Tokenized_Padded_tweets.csv not found.")
    st.markdown("---")

elif selected == "ML Model Selection":
    st.title("Model Selection")
    st.write("""Details about classifiers and cross-validation results.""")
    tab1, tab2 = st.tabs(["Classification Results", "Display Results Figures"])
    with tab1:
        st.subheader('Table I. Classification Results')
        data = {
            'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
            'Precision': ['0.83 ± 0.04', '0.77 ± 0.06', '0.77 ± 0.06', '0.71 ± 0.07', '0.79 ± 0.05', '0.78 ± 0.05'],
            'Recall': ['0.96 ± 0.02', '1.00 ± 0.01', '1.00 ± 0.01', '0.96 ± 0.02', '0.90 ± 0.03', '1.00 ± 0.01'],
            'F1-Score': ['0.88 ± 0.02', '0.87 ± 0.03', '0.87 ± 0.03', '0.81 ± 0.04', '0.84 ± 0.04', '0.87 ± 0.03']
        }
        df_results = pd.DataFrame(data)
        st.table(df_results)
    with tab2:
        st.subheader('Display Results Figures')
        data = {
            'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
            'Precision': [0.83, 0.77, 0.77, 0.71, 0.79, 0.78],
            'Recall': [0.96, 1.00, 1.00, 0.96, 0.90, 1.00],
            'F1-Score': [0.88, 0.87, 0.87, 0.81, 0.84, 0.87]
        }
        df_fig = pd.DataFrame(data)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Precision'], name='Precision'))
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Recall'], name='Recall'))
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['F1-Score'], name='F1-Score'))
        fig.update_layout(title='Classification Results', xaxis_title='Algorithm', yaxis_title='Score', barmode='group', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

elif selected == "Try The Model":
    st.title("Tweet Tone Triage Application")

    model_choice = st.selectbox("Select model", (
        "4T Model",
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "SVM - SVC",
    ))

    user_input = st.text_area("Enter the tweet:", value="!!!!! RT @mleew17: boy dats cold...tyga dwn bad for cuffin dat hoe in the 1st place!!", height=150)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter text for prediction.")
            st.stop()

        # Preprocess pipeline
        preprocessed_tweet = preprocess(user_input)
        clean_tweet = preprocess_clean(preprocessed_tweet)
        stripped_tweet = strip_hashtags(clean_tweet)
        stemmed_tokens = stemming(stripped_tweet)
        stemmed_tweet = " ".join(stemmed_tokens)

        # Local tokenization & padding (for your Keras model)
        tokenizer_local = Tokenizer()
        tokenizer_local.fit_on_texts([stemmed_tweet])
        encoded_docs = tokenizer_local.texts_to_sequences([stemmed_tweet])[0]
        max_length = 100
        padded_docs = pad_sequences([encoded_docs], maxlen=max_length, padding='post')

        label_map = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        label_map_4T = {"LABEL_0": "Neutral", "LABEL_1": "Offensive", "LABEL_2": "Hate"}
        # Show preprocessing
        st.markdown("#### Preprocessing")
        st.write("Preprocessed:", preprocessed_tweet)
        st.write("Cleaned:", clean_tweet)
        st.write("Hashtag-stripped:", stripped_tweet)
        st.write("Stemmed tokens:", stemmed_tokens)
        st.write("Padded tokens shape:", padded_docs.shape)
        st.markdown("---")

        # HuggingFace option:
        if model_choice == "4T Model":
            if hf_pipe is None:
                st.error(f"4T Model failed to load: {hf_id2label}")
            else:
                with st.spinner("Analyzing with 4T model..."):
                    try:
                        # hf_pipe returns list of dicts, e.g. [{"label": "LABEL_1", "score": 0.987}]
                        hf_result = hf_pipe(user_input, truncation=True)
                        st.subheader("4T Model Output")
                        #st.write(hf_result[0]["label"])
                        raw_label = hf_result[0]["label"]
                        readable_label = label_map.get(raw_label, raw_label)
                        st.write(f"**Class:** {readable_label}")

                    except Exception as e:
                        st.error(f"4T Model error")

        else:
            # Ensure local models loaded
            if any(m is None for m in (LR_model, RF_model, DT_model, SVM_model)):
                st.sidebar.warning("One or more local models failed to load; check sidebar messages.")
           
            # Logistic Regression
            if model_choice == "Logistic Regression":
                if LR_model is None:
                    st.error("LR model not loaded.")
                else:
                    try:
                        y_pred = LR_model.predict(padded_docs)
                        st.subheader("Logistic Regression Result")
                        st.write(f"Prediction: {label_map.get(int(y_pred[0]), str(y_pred[0]))}")
                    except Exception as e:
                        st.error(f"LR model prediction error: {e}")

            # Random Forest
            elif model_choice == "Random Forest":
                if RF_model is None:
                    st.error("Random Forest model not loaded.")
                else:
                    try:
                        y_pred = RF_model.predict(padded_docs)
                        st.subheader("Random Forest Result")
                        st.write(f"Prediction: {label_map.get(int(y_pred[0]), str(y_pred[0]))}")
                    except Exception as e:
                        st.error(f"Random Forest prediction error: {e}")

            # Decision Tree
            elif model_choice == "Decision Tree":
                if DT_model is None:
                    st.error("Decision Tree model not loaded.")
                else:
                    try:
                        y_pred = DT_model.predict(padded_docs)
                        st.subheader("Decision Tree Result")
                        st.write(f"Prediction: {label_map.get(int(y_pred[0]), str(y_pred[0]))}")
                    except Exception as e:
                        st.error(f"Decision Tree prediction error: {e}")

            # SVM - SVC
            elif model_choice == "SVM - SVC":
                if SVM_model is None:
                    st.error("SVM model not loaded.")
                else:
                    try:
                        y_pred = SVM_model.predict(padded_docs)
                        st.subheader("SVM - SVC Result")
                        st.write(f"Prediction: {label_map.get(int(y_pred[0]), str(y_pred[0]))}")
                    except Exception as e:
                        st.error(f"SVM prediction error: {e}")

    st.markdown("---")

elif selected == "About":
    st.title("About This App")
    st.write("""
    This application is designed for the analysis of hate speech and offensive language in tweets. 
    It provides several functionalities, including:
    
    - Loading and exploring the dataset
    - Understanding class distribution of hate speech, offensive language, and neutral content
    - Preprocessing tweets (removing URLs, mentions, emojis, and special characters)
    - Tokenizing and padding tweet sequences for machine learning models
    - Model selection and classification of tweets using traditional machine learning classifiers
    - Testing a trained model for real-time predictions of tweet sentiment or class
    
    **Key Features:**
    
    - Utilizes a crowdsourced dataset from Davidson et al. (2017)
    - Supports preprocessing steps like stemming and tokenization
    - Provides an interactive interface for exploring dataset attributes, class distributions, and preprocessing steps
    - Enables users to test machine learning models on custom tweets
    
    **References:**
    
    - Dataset Source: Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language.
    - Available on Kaggle: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
    """)
    st.markdown("---")

elif selected == "Contact":
    st.title("Supervisors")
    st.write("This application was designed and deployed by **Tharwat El-Sayed Ismail**, under the supervision of:")
    def safe_image(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    ayman_image = safe_image("Ayman Elsayed.jpg")
    abdallah_image = safe_image("Abdullah-N-Moustafa.png")
    tharwat_image = safe_image("Tharwat Elsayed Ismail.JPG")
    if ayman_image:
        st.subheader("Prof. Dr. Ayman EL-Sayed")
        st.image(ayman_image, width=200)
        st.write("[ayman.elsayed@el-eng.menofia.edu.eg](mailto:ayman.elsayed@el-eng.menofia.edu.eg)")
    if abdallah_image:
        st.subheader("Dr. Abdallah Moustafa Nabil")
        st.image(abdallah_image, width=200)
        st.write("[abdalla.moustafa@ejust.edu.eg](mailto:abdalla.moustafa@ejust.edu.eg)")
    if tharwat_image:
        st.subheader("Eng. Tharwat El-Sayed Ismail")
        st.image(tharwat_image, width=200)
        st.write("[tharwat.elsayed@el-eng.menofia.edu.eg](mailto:tharwat.elsayed@el-eng.menofia.edu.eg)")
    st.markdown("---")
    st.title("Contact Me")
    st.write("**Email:** tharwat_uss89@hotmail.com")
    st.markdown("---")
