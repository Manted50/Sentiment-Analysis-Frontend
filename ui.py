import streamlit as st
import requests
import plotly.express as px
import time

API_URL = "https://sentiment-analysis-backend-g4h6.onrender.com/"

# Configuration de la page et du thème
st.set_page_config(
    page_title="Analyseur de Sentiment",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Personnalisation CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #dff0d8 !important;
        color: #3c763d !important;
    }
    .stError {
        background-color: #f2dede !important;
        color: #a94442 !important;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# Gestion de l'état de session
if "_tweet_text" in st.session_state:
    st.session_state.tweet_text = st.session_state._tweet_text
    del st.session_state._tweet_text

if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("ℹ️ Informations")
st.sidebar.write("Cette application analyse le sentiment d'un texte donné (positif ou négatif) à l'aide d'un modèle d'apprentissage automatique.")
st.sidebar.write("Vous pouvez également obtenir une explication de la décision du modèle à l'aide de LIME.")
st.sidebar.markdown("### Exemples de texte")
example_tweets = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The service was okay, not great but not terrible.",
    "I'm so happy with the results!",
    "I regret buying this, it's awful."
]
for i, example in enumerate(example_tweets):
    if st.sidebar.button(f"📝 Exemple {i + 1}", key=f"ex_{i}"):
        st.session_state.tweet_text = example

# Titre principal
st.title("📝 Analyseur de sentiment")

# Zone de texte avec compteur de caractères
text_input = st.text_area(
    "Entrez votre texte (max 280 caractères) :",
    max_chars=280,
    placeholder="Tapez votre texte ici...",
    key="tweet_text"
)
st.write(f"{len(text_input)}/280 caractères")

# Boutons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    predict_btn = st.button("🎯 Prédire le Sentiment")
with col2:
    explain_btn = st.button("🔍 Expliquer avec LIME")
with col3:
    delete_btn = st.button("❌ Effacer")

if delete_btn:
    st.session_state._tweet_text = ""
    st.rerun()

# Fonctions utilitaires
def call_prediction_api(text):
    with st.spinner("Analyse en cours..."):
        time.sleep(0.5)  # Simule un délai pour le spinner
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
    return response

def call_explain_api(text):
    with st.spinner("Génération de l'explication..."):
        time.sleep(0.5)
        response = requests.post(
            f"{API_URL}/explain",
            json={"text": text},
            timeout=30
        )
    return response

def display_prediction(data):
    st.markdown("---")
    with st.container():
        st.markdown("<h3 style='text-align: center;'>Résultat de l'analyse</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            sentiment = data.get("sentiment", "")
            confidence = data.get("confidence", 0)
            if sentiment == "positive":
                st.success(f"😊 **POSITIF** ({confidence:.1%})")
            else:
                st.error(f"😞 **NÉGATIF** ({confidence:.1%})")
        with col2:
            fig = px.bar(
                x=["Négatif", "Positif"],
                y=[data.get("probability_negative", 0), data.get("probability_positive", 0)],
                labels={"x": "Sentiment", "y": "Probabilité"},
                title="Probabilités",
                color=["#d62728", "#2ca02c"]
            )
            st.plotly_chart(fig, use_container_width=True)

def display_explanation(data):
    st.markdown("---")
    st.write(f"### Sentiment: {data.get('sentiment', '').upper()}")
    st.write("### Mots importants :")
    for item in data.get("explanation", []):
        st.write(f"**{item['word']}** : {item['importance']:.4f}")
    st.components.v1.html(data.get("html_explanation", ""), height=400, scrolling=True)

# Gestion des boutons
if predict_btn:
    if not st.session_state.tweet_text:
        st.warning("Veuillez entrer du texte.")
    else:
        response = call_prediction_api(st.session_state.tweet_text)
        if response.status_code == 200:
            data = response.json()
            display_prediction(data)
            st.session_state.history.append({
                "text": st.session_state.tweet_text,
                "sentiment": data["sentiment"],
                "confidence": data["confidence"]
            })
        else:
            st.error(f"Erreur API : {response.status_code}")

if explain_btn:
    if not st.session_state.tweet_text:
        st.warning("Veuillez entrer du texte.")
    else:
        response = call_explain_api(st.session_state.tweet_text)
        if response.status_code == 200:
            data = response.json()
            display_explanation(data)
        else:
            st.error(f"Erreur API : {response.status_code}")

# Historique
if st.session_state.history:
    st.markdown("### Historique des analyses")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Analyse {len(st.session_state.history)-i}: {item['sentiment'].upper()} ({item['confidence']:.1%})"):
            st.write(item["text"])
