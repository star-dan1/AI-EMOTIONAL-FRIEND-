
import streamlit as st
import joblib
import numpy as np
import unicodedata

# ---------------------------
# Load saved models
# ---------------------------
lang_pipe = joblib.load(r"C:\Users\User PC\Documents\Emmotional AI\models\lang_pipeline.joblib")
emo_pipe  = joblib.load(r"C:\Users\User PC\Documents\Emmotional AI\models\emo_pipeline.joblib")

# ---------------------------
# Text normalization
# ---------------------------
def normalize_text(s):
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    return s

# ---------------------------
# Response map
# ---------------------------
RESPONSE_MAP = {
    "Yoruba": {
        "Happy":   "Inu mi dun pe o n rẹrin!",
        "Sad":     "Má bà a ní lokan, ohun gbogbo máa dáa.",
        "Angry":   "Jọwọ, gbiyanju lati tú ọkan rẹ soke — a le sọrọ.",
        "Fear":    "Má ṣe yọ ara rẹ lẹnu, emi wà n'íbẹ̀ fún ọ.",
        "Love":    "O ṣeun, inu mi dun.",
        "Surprise":"Oh! Iyanu ni yẹn!",
        "Neutral": "O ṣeun, mo ti gbọ́ ọ."
    },
    "Hausa": {
        "Happy":   "Ina farin ciki da jin haka!",
        "Sad":     "Ina tausaya maka, za mu iya magana idan kana so.",
        "Angry":   "Ka kwantar da hankalinka; muna iya tattauna shi.",
        "Fear":    "Kar ka ji tsoro, zan taimaka in dai zan iya.",
        "Love":    "Na gode — hakan ya faranta min rai.",
        "Surprise":"Abin mamaki ne!",
        "Neutral": "Na gode, na fahimta."
    },
    "Igbo": {
        "Happy":   "Obi dị m ụtọ ịnụ nke ahụ!",
        "Sad":     "Ekwela ka obi daa gi, ihe niile ga-adịrị mma.",
        "Angry":   "Kwusi iwe—ka anyị kwurịta ya nwayọọ.",
        "Fear":    "Ejila egwu jide gị, m nọ ebe a.",
        "Love":    "Ekele m, nke ahụ gbaara m obi umeala.",
        "Surprise":"Nnọọ! Nke ahụ juputara m n'ịtụnanya.",
        "Neutral": "Daalụ, ewezuga nke ahụ."
    }
}

# ---------------------------
# Function to classify
# ---------------------------
def classify_and_respond(text):
    t = normalize_text(text)

    # language prediction
    lang_probs = lang_pipe.predict_proba([t])[0]
    lang_idx = np.argmax(lang_probs)
    lang_label = lang_pipe.classes_[lang_idx]
    lang_conf = float(lang_probs[lang_idx])

    # emotion prediction
    emo_probs = emo_pipe.predict_proba([t])[0]
    emo_idx = np.argmax(emo_probs)
    emo_label = emo_pipe.classes_[emo_idx]
    emo_conf = float(emo_probs[emo_idx])

    # response
    response = "Thanks — I'm here for you."
    if lang_label in RESPONSE_MAP and emo_label in RESPONSE_MAP[lang_label]:
        response = RESPONSE_MAP[lang_label][emo_label]

    return lang_label, lang_conf, emo_label, emo_conf, response

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Multilingual Emotion Chatbot", page_icon="💬", layout="centered")

st.title("💬 Multilingual Emotion Chatbot")
st.write("Chat in **Yoruba**, **Hausa**, or **Igbo** and get an emotional response!")

user_input = st.text_input("Type your message:")

if user_input:
    lang, lconf, emo, econf, resp = classify_and_respond(user_input)
    st.markdown(f"**Detected Language:** {lang} ({lconf:.2f})")
    st.markdown(f"**Detected Emotion:** {emo} ({econf:.2f})")
    st.success(f"**Bot:** {resp}")
