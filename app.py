import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import PyPDF2
import io

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Function to transform text
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


# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon=":email:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .css-1v3fvcr {background-color: #3f2a4e !important;}
    .css-1aumxhk {color: white !important;}
    </style>
    """,
    unsafe_allow_html=True
)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with io.BytesIO(pdf_file.read()) as f:  # Read the content of the uploaded PDF file as bytes
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


# Sidebar with navigation bar
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio(
    "",
    ["Home", "About", "Contact", "Upload PDF"]
)

# Main content area
if nav_selection == "Home":
    st.title("Email/SMS Spam Detector - By ANIK CHOWDHURY")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):

        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])
        # Predict using the model
        result = model.predict(vector_input)[0]
        # Display the result
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")

elif nav_selection == "About":
    st.title("About")
    st.write("This is a simple email/SMS spam classifier built using Streamlit and a trained machine learning model.")

elif nav_selection == "Contact":
    st.title("Contact")
    st.write("For any inquiries, please contact us at example@example.com")

elif nav_selection == "Upload PDF":
    st.title("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        st.write("Extracted Text:")
        st.write(text)

        transformed_text = transform_text(text)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
