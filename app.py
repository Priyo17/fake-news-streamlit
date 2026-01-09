import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

# -------------------------------
# Text Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -------------------------------
# Load NRC Emotion Lexicon
# -------------------------------
def load_nrc(path):
    lexicon = {}
    emotions = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, value = line.strip().split('\t')
            if int(value) == 1:
                lexicon.setdefault(word, []).append(emotion)
                emotions.add(emotion)

    return {
        "lexicon": lexicon,
        "emotions": sorted(list(emotions))
    }

# -------------------------------
# Emotion Extraction
# -------------------------------
def extract_emotions(text, emotion_dict):
    emotion_counts = {e: 0 for e in emotion_dict["emotions"]}

    for word in text.split():
        if word in emotion_dict["lexicon"]:
            for emo in emotion_dict["lexicon"][word]:
                emotion_counts[emo] += 1

    return emotion_counts

# -------------------------------
# Load Models & Objects
# -------------------------------
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("emotion_list.pkl", "rb") as f:
    emotion_list = pickle.load(f)

emotion_dict = load_nrc("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion-Sensitive Fake News Detection")
st.title("📰 Emotion-Sensitive Fake News Detection")

st.write(
    "This system detects fake news using **emotion-aware NLP features** "
    "combined with classical machine learning."
)

user_text = st.text_area("Enter News Text")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_text)

        emotion_scores = extract_emotions(clean_text, emotion_dict)
        emotion_vector = [emotion_scores[e] for e in emotion_list]

        tfidf_vector = tfidf.transform([clean_text])
        final_vector = hstack([tfidf_vector, emotion_vector])

        prediction = rf_model.predict(final_vector)[0]

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News Detected")

        st.subheader("Emotion Profile")
        st.bar_chart(dict(zip(emotion_list, emotion_vector)))
