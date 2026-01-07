import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Mental Health Support System",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Custom CSS (UI Polish)
# -----------------------------
st.markdown("""
<style>
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Mental Health Support System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Awareness-based, safe & explainable mental health guidance</p>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("drugs_side_effects_drugs_com.csv")
    df.columns = df.columns.str.lower()
    return df

df = load_data()

# -----------------------------
# Filter Mental Health Data
# -----------------------------
mental_keywords = [
    "depression", "anxiety", "bipolar", "panic",
    "schizophrenia", "adhd", "insomnia"
]

mental_df = df[
    df['medical_condition'].str.lower().str.contains('|'.join(mental_keywords), na=False)
].copy()

mental_df['medical_condition'] = mental_df['medical_condition'].str.lower()
mental_df = mental_df.reset_index(drop=True)
# -----------------------------
# NLP: Condition Similarity
# -----------------------------
vectorizer = CountVectorizer(stop_words='english')
condition_vectors = vectorizer.fit_transform(mental_df['medical_condition'])
similarity_matrix = cosine_similarity(condition_vectors)

# -----------------------------
# Helper Functions
# -----------------------------
def condition_exists(condition):
    return condition in mental_df['medical_condition'].values


def find_closest_condition(user_input):
    corpus = mental_df['medical_condition'].unique().tolist()
    vectors = vectorizer.fit_transform(corpus + [user_input])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    return corpus[similarity.argmax()]


def get_similar_conditions(condition, top_n=3):
    idx = mental_df[mental_df['medical_condition'] == condition].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    similar = []
    for i, score in scores:
        cond = mental_df.iloc[i]['medical_condition']
        if cond != condition:
            similar.append(cond)
        if len(similar) == top_n:
            break
    return list(set(similar))


def get_common_side_effects(condition, top_n=5):
    effects = mental_df[
        mental_df['medical_condition'] == condition
    ]['side_effects']

    effects = effects.dropna().astype(str)
    all_effects = ",".join(effects).split(",")

    common = Counter(all_effects).most_common(top_n)
    return [e[0].strip() for e in common]


# -----------------------------
# Simplified Treatment Categories
# -----------------------------
category_mapping = {
    "selective serotonin reuptake inhibitors": 
        "Take such Antidepressants that helps improve your mood and reduce anxiety by increasing serotonin in your brain)",

    "serotonin-norepinephrine reuptake inhibitors": 
        "Take such Antidepressants that helps improve mood and energy by balancing serotonin and norepinephrine in your brain)",

    "tricyclic antidepressants": 
        "Take older-Antidepressants that boosts mood and can help with sleep or pain by increasing brain chemical levels",

    "monoamine oxidase inhibitors": 
       "Take special Antidepressants that helps lift your mood by keeping certain brain chemicals active longer(used with care)",

    "atypical antipsychotics": 
        "Take Mood-stabilizing medications",

    "benzodiazepines": 
        "Take Anti-anxiety medicines (short-term use)",

    "cns stimulants": 
        "Take focus enhnacing medicine that helps increase alertness, focus, and energy by stimulating the brain" ,

    "Dopaminergic antiparkinsonism agents":
        "Take such medicines that helps with Parkinson’s by improving movement and reducing stiffness",
    
    "Mood stabilizers":
        "Take mood stabilizers that helps keep your mood steady and prevent extreme highs or lows"

    
}

def get_treatment_categories(condition):
    categories = mental_df[
        mental_df['medical_condition'] == condition
    ]['drug_classes']

    simplified = set()
    for cat in categories.dropna():
        for key in category_mapping:
            if key in cat.lower():
                simplified.add(category_mapping[key])

    return list(simplified)

# -----------------------------
# Main Recommender
# -----------------------------
def mental_health_recommender(user_input):
    try:
        user_input = str(user_input).lower().strip()

        note = "Input processed"
        condition = user_input

        if not condition_exists(user_input):
            condition = find_closest_condition(user_input)
            note = f"Input '{user_input}' mapped to '{condition}'"
        else:
            note = "Exact condition matched"

        return {
            "note": note,
            "similar_conditions": get_similar_conditions(condition),
            "treatments": get_treatment_categories(condition),
            "side_effects": get_common_side_effects(condition)
        }

    except Exception as e:
        # FAIL-SAFE RETURN (VERY IMPORTANT)
        return {
            "note": "We couldn't fully process the input, showing general guidance.",
            "similar_conditions": [],
            "treatments": [],
            "side_effects": []
        }



# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("📝 MENTAL CONDITION ")
user_condition = st.sidebar.text_input(
    "Enter your concern (e.g. stress, anxiety, depression)"
)
submit = st.sidebar.button("Get Support 💙")

# -----------------------------
# Output Section
# -----------------------------
if submit and user_condition.strip() != "":
    result = mental_health_recommender(user_condition)

    st.subheader("🔍 Understanding Your Input")
    st.info(result.get("note", "Input processed safely"))

    st.subheader("🧠 Similar Conditions")
    for c in result.get("similar_conditions", []):
        st.markdown(f"• **{c.title()}**")

    st.subheader("💊 Treatment Categories (Awareness Only)")
    treatments = result.get("treatments", [])
    if treatments:
        for t in treatments:
            st.markdown(f"• {t}")
    else:
        st.markdown("• General mental health support & therapy")

    st.subheader("⚠️ Common Side Effects (Awareness)")
    for s in result.get("side_effects", []):
        st.markdown(f"• {s}")

    st.warning(
        "⚠️ This system is for educational purposes only and does not diagnose or prescribe medication."
    )
