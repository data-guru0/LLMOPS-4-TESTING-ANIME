import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

# ✅ This must be the very first Streamlit command
st.set_page_config(page_title="Anime Recommender", layout="wide")

load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()

pipeline = init_pipeline()

st.title("🎌 Anime Recommender System")

query = st.text_input("Enter your anime preferences (e.g., 'light-hearted comedy with high school setting'):")

if query:
    with st.spinner("Fetching recommendations..."):
        response = pipeline.recommend(query)
        st.markdown("### 🎯 Recommendations")
        st.write(response)
