import streamlit as st
import requests
import plotly.express as px

API_URL = "http://localhost:8000"
#API_URL = "https://manted50.pythonanywhere.com"

# Sidebar content
st.sidebar.title("ℹ️ About")
st.sidebar.write("This app analyzes the sentiment of a given text (positive or negative) using a machine learning model.")
st.sidebar.write("You can also get an explanation of the model's decision using LIME.")

st.sidebar.write("### Examples:")
example_tweets = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The service was okay, not great but not terrible.",
    "I'm so happy with the results!",
    "I regret buying this, it's awful."
]

selected_example = st.sidebar.selectbox("Choose an example:", [""] + example_tweets)
if selected_example:
    st.session_state.tweet_text = selected_example

# Main content
st.title("📝 Sentiment Analyzer")

if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""

tweet_text = st.text_area(
    "Entrez votre texte (max 280 caractères) :",
    value=st.session_state.tweet_text,
    max_chars=280,
    placeholder="Tapez votre texte ici..."
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_btn = st.button("🎯 Prédire le Sentiment")

with col2:
    explain_btn = st.button("🔍 Expliquer avec LIME")

with col3:
    cancel_btn = st.button("❌ Annuler")

# Appel API
def call_prediction_api(text):
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=30
    )
    return response

def call_explain_api(text):
    response = requests.post(
        f"{API_URL}/explain",
        json={"text": text},
        timeout=30
    )
    return response

# Affichage des résultats
def display_prediction(data):
    sentiment = data.get("sentiment", "")
    confidence = data.get("confidence", 0)
    prob_neg = data.get("probability_negative", 0)
    prob_pos = data.get("probability_positive", 0)

    if sentiment == "positive":
        st.success(f"😊 POSITIVE ({confidence:.1%})")
    else:
        st.error(f"😞 NEGATIVE ({confidence:.1%})")

    fig = px.bar(
        x=["Negative", "Positive"],
        y=[prob_neg, prob_pos],
        labels={"x": "Sentiment", "y": "Probability"},
        title="Sentiment Probabilities"
    )
    st.plotly_chart(fig)

def display_explanation(data):
    st.write(f"### Sentiment: {data.get('sentiment', '').capitalize()}")

    important_words = data.get("explanation", [])
    st.write("### Mots importants :")
    for item in important_words:
        st.write(f"**{item['word']}** : {item['importance']:.4f}")

    html_expl = data.get("html_explanation", None)
    if html_expl:
        st.write("### Explication LIME (HTML)")
        st.components.v1.html(html_expl, height=400, scrolling=True)

# Buttons actions
if predict_btn:
    if not tweet_text:
        st.warning("Veuillez entrer du texte.")
    else:
        response = call_prediction_api(tweet_text)
        if response.status_code == 200:
            data = response.json()
            display_prediction(data)
        else:
            st.error(f"Erreur API : {response.status_code}")

if explain_btn:
    if not tweet_text:
        st.warning("Veuillez entrer du texte.")
    else:
        response = call_explain_api(tweet_text)
        if response.status_code == 200:
            data = response.json()
            display_explanation(data)
        else:
            st.error(f"Erreur API : {response.status_code}")

if cancel_btn:
    st.session_state.tweet_text = ""
    st.experimental_rerun()
