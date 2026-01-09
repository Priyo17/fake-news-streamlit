import streamlit as st
import pickle
import re
import string
from scipy.sparse import hstack, csr_matrix

# -------------------------------
# Stopwords (simple list to avoid NLTK downloads)
# -------------------------------
stop_words = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her
here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves
""".split())

# -------------------------------
# Simple Tokenizer
# -------------------------------
def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = simple_tokenize(text)
    return " ".join(tokens)

# -------------------------------
# Load NRC Emotion Lexicon
# -------------------------------
@st.cache_data
def load_nrc(path):
    lexicon = {}
    emotions = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, value = line.strip().split("\t")
            if int(value) == 1:
                lexicon.setdefault(word, []).append(emotion)
                emotions.add(emotion)
    return {"lexicon": lexicon, "emotions": sorted(list(emotions))}

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
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    emotion_list = pickle.load(open("emotion_list.pkl", "rb"))
    return rf_model, tfidf, emotion_list

rf_model, tfidf, emotion_list = load_models()
emotion_dict = load_nrc("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion-Sensitive Fake News Detection")
st.title("📰 Emotion-Sensitive Fake News Detection")

st.write(
    "This system detects fake news using **emotion-aware NLP features** "
    "combined with a Random Forest classifier."
)

user_text = st.text_area("Enter News Text")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_text)

        # -------------------
        # Emotion features
        # -------------------
        emotion_scores = extract_emotions(clean_text, emotion_dict)
        # Align emotion vector with training order
        emotion_vector = [emotion_scores.get(e, 0) for e in emotion_list]
        emotion_vector_sparse = csr_matrix([emotion_vector])

        # -------------------
        # TF-IDF features
        # -------------------
        tfidf_vector = tfidf.transform([clean_text])

        # -------------------
        # Combine features
        # -------------------
        final_vector = hstack([tfidf_vector, emotion_vector_sparse])

        # -------------------
        # Prediction
        # -------------------
        prediction = rf_model.predict(final_vector)[0]
        prediction_proba = rf_model.predict_proba(final_vector)[0]

        # Display prediction
        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News Detected")

        # Show probabilities
        st.write(f"Prediction probabilities → Fake: {prediction_proba[0]:.2f}, Real: {prediction_proba[1]:.2f}")

        # -------------------
        # Display emotion chart
        # -------------------
        st.subheader("Emotion Profile")
        st.bar_chart(emotion_scores)

        # -------------------
        # Optional debug info
        # -------------------
        st.write("Cleaned text:", clean_text)
        st.write("TF-IDF vector shape:", tfidf_vector.shape)
        st.write("Emotion vector:", emotion_vector)
        st.write("Final feature shape:", final_vector.shape)
