import streamlit as st
import pickle
import gdown
import os
import pandas as pd

# Load model dari Google Drive
@st.cache_resource
def load_model_from_gdrive(file_id, output_path="model.pkl"):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        model = pickle.load(f)
    
    return model

# Load dataset lengkap dari CSV
@st.cache_data
def load_full_dataset():
    df = pd.read_csv("netflix_preprocessed.csv")  # Ganti path jika perlu
    return df

# Load model dan data
file_id = "1LZQVpBg9ZsCweR6FA_ug1EjMauTDAaNw"
model_data = load_model_from_gdrive(file_id)

netflix_title_series = model_data["netflix_title"]  # Series of titles
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]

# load function recommender
with open('recommender.pkl', 'rb') as f:
    content_recommender = pickle.load(f)

# content_recommender = model_data["content_recommender"]

full_df = load_full_dataset()

# UI Streamlit
st.title("Netflix Movie Recommender")
st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")

if search_clicked and title:
    # Cek apakah title ada di netflix_title_series
    if title in set(netflix_title_series):
        # Rekomendasi
        st.subheader("Recommended Titles:")
        recommendations = content_recommender(
            title
        )

    else:
        st.error("❌ Movie title not found in model title list.")
