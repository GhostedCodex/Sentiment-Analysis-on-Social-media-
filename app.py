import streamlit as st
from transformers import pipeline

# --- 1. Setup the Page ---
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖")

# Add a cool title and description
st.title("🤖 AI Sentiment Analyzer")
st.markdown(
    "Type a sentence below, and the AI will determine if it is **Positive** or **Negative**.")

# --- 2. Load the Model (Cached) ---
# We use @st.cache_resource so the model only loads once (it's heavy!)


@st.cache_resource
def load_model():
    # This downloads the DistilBERT model finetuned for sentiment
    return pipeline("sentiment-analysis")


# Show a spinner while the model loads
with st.spinner("Loading AI Model..."):
    sentiment_pipeline = load_model()

# --- 3. User Input ---
user_input = st.text_area("Enter your text here:",
                          placeholder="e.g., I loved the movie, but the popcorn was stale.")

# --- 4. The "Analyze" Button ---
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        # Get prediction
        result = sentiment_pipeline(user_input)[0]
        label = result['label']
        score = result['score']

        # Display Result with specific colors/emojis
        if label == 'POSITIVE':
            st.success(f"**Sentiment:** {label} 😃")
            st.metric("Confidence Score", f"{score:.2%}")
        else:
            st.error(f"**Sentiment:** {label} 😔")
            st.metric("Confidence Score", f"{score:.2%}")

    else:
        st.warning("Please enter some text first!")

# --- 5. Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a **DistilBERT** model from Hugging Face to analyze the emotional tone of text.")
