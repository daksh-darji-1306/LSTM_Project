# --- Sarcasm Detection Streamlit Web App (Enhanced UI) ---
# This script loads a pre-trained LSTM model to create an interactive,
# visually appealing web application for sarcasm detection.

import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Sarcasm Detector",
    page_icon="😏",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
# This injects CSS to style the result cards and buttons for a more polished look.
st.markdown("""
<style>
.stButton>button {
    border-radius: 15px;
    border: 1px solid #f63366;
    color: #f63366;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    border-color: #f63366;
    color: white;
    background-color: #f63366;
}
.result-card {
    padding: 25px;
    border-radius: 10px;
    color: white;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
.sarcastic { background: linear-gradient(to right, #ff416c, #ff4b2b); }
.not-sarcastic { background: linear-gradient(to right, #28b485, #8ceabb); }
.unsure { background: linear-gradient(to right, #ffc107, #ff9800); }
.result-text {
    font-size: 28px;
    font-weight: bold;
}
.confidence-text {
    font-size: 18px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# --- Load Model and Tokenizer (Cached) ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and tokenizer from disk."""
    try:
        model = load_model("sarcasm_model.h5")
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.error("Please ensure 'sarcasm_model.h5' and 'tokenizer.json' are in the same directory.")
        st.stop()

model, tokenizer = load_assets()

# --- Prediction Function ---
def predict_sarcasm(text, model, tokenizer):
    """Predicts sarcasm for a given text sentence."""
    max_length = 40
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence, verbose=0)
    return prediction[0][0]

# --- UI Layout ---

# Sidebar Content
st.sidebar.title("About the App")
st.sidebar.info(
    """
    This app uses a **Long Short-Term Memory (LSTM)** neural network to analyze
    text and predict whether it's sarcastic.
    
    The model was trained on the "News Headlines Dataset for Sarcasm Detection" from Kaggle,
    which contains over 28,000 headlines from various sources.
    """
)
st.sidebar.markdown("---")

# Main Page Title
st.title("AI Sarcasm Detector")
st.markdown("### Is your text sarcastic, or are you just kidding? Let's find out! 😏")

# Expander for instructions and details
with st.expander("🤔 How does this work?"):
    st.write("""
        This tool uses a deep learning model called an LSTM (Long Short-Term Memory network). Here’s a simple breakdown:
        1.  **Training:** The model was trained on thousands of news headlines, learning the patterns, word choices, and contexts that distinguish sarcastic text from non-sarcastic text.
        2.  **Input:** When you enter a sentence, it's converted into a sequence of numbers that the model can understand.
        3.  **Prediction:** The LSTM processes this sequence, remembering the context, and outputs a "sarcasm score" between 0 (not sarcastic) and 1 (sarcastic).
    """)

st.markdown("---")

# User Input and Examples
st.subheader("Analyze Your Text")

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Example buttons to provide easy testing
st.write("Or try one of these examples:")
examples = [
    "I absolutely love being stuck in traffic for hours.",
    "Wow, another software update that fixed nothing.",
    "This is a fantastic day for our country.",
    "Scientists have discovered a new species of fish."
]
col1, col2 = st.columns(2)
if col1.button(examples[0]):
    st.session_state.user_input = examples[0]
if col1.button(examples[1]):
    st.session_state.user_input = examples[1]
if col2.button(examples[2]):
    st.session_state.user_input = examples[2]
if col2.button(examples[3]):
    st.session_state.user_input = examples[3]

# Main text area for user input
user_input = st.text_area(
    "Enter your sentence here:",
    st.session_state.user_input,
    height=100,
    placeholder="e.g., Oh great, another meeting that could have been an email."
)

# Analyze button
if st.button("Analyze Text", type="primary"):
    if user_input:
        with st.spinner("🤖 The AI is thinking..."):
            score = predict_sarcasm(user_input, model, tokenizer)
            
            # Display result in a styled card based on the score
            if score > 0.6:
                st.markdown(f'<div class="result-card sarcastic"><div class="result-text">😏 Sarcastic</div><div class="confidence-text">Model Confidence: {score:.2%}</div></div>', unsafe_allow_html=True)
            elif score < 0.4:
                st.markdown(f'<div class="result-card not-sarcastic"><div class="result-text">😊 Not Sarcastic</div><div class="confidence-text">Model Confidence: {(1-score):.2%}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card unsure"><div class="result-text">🤔 Unsure</div><div class="confidence-text">The model finds this text ambiguous (Sarcasm Score: {score:.2%})</div></div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

