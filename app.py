!pip install nltk

import streamlit as st
import pickle
import string 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Load the pre-trained model and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

try:
    model = pickle.load(open('model.pkl', 'rb'))
except ValueError:
    # Handle version mismatch or other loading issues
    st.error("Error loading the model. Please check the scikit-learn version.")
    model = None

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Set background color and padding
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a header with some style
st.markdown(
    """
    <div class="main">
        <h2 style='text-align: center; color: #3498db; font-size: 2.5em; font-weight: bold; margin-bottom: 20px;'>Email Phishing Detection</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Text input for the user to enter the email
input_email = st.text_area("Enter the email text")
st.markdown(
    """
    <style>
        div[data-baseweb="textarea"] {
            width: 100%;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            background-color: #fff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Button to trigger the prediction
if st.button('Predict'):
    st.markdown(
        """
        <style>
            div[data-baseweb="button"] {
                padding: 15px;
                background-color: #e74c3c;
                color: #fff;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1.2em;
                font-weight: bold;
            }

            .result {
                text-align: center;
                font-size: 24px;
                margin-top: 12px;
                padding: 8px;
                border-radius: 8px;
                font-weight: bold;
            }

            .phishing {
                background-color: #e74c3c;
                color: #fff;
            }

            .safe {
                background-color: #2ecc71;
                color: #fff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if model is not None:
        transformed_email = transform_text(input_email)
        vector_input = tfidf.transform([transformed_email])
        result = model.predict(vector_input)[0]

        result_text = "Phishing Email" if result == 1 else "Safe Email"
        result_color = "phishing" if result == 1 else "safe"

        st.markdown(f'<p class="result {result_color}">{result_text}</p>', unsafe_allow_html=True)
    else:
        st.warning("Model is not loaded. Please check the model file.")
