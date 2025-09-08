import streamlit as st
import pandas as pd
import json
import base64
import os
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Human Evaluation for Peer Reviews",
    layout="wide"
)

# --- Data Loading (Cached for Performance) ---
@st.cache_data
def load_data(data_folder):
    """
    Loads all necessary CSV and JSON files from the data folder.
    Returns a tuple of dataframes and dictionaries.
    """
    try:
        user_df = pd.read_csv(Path(data_folder) / "user.csv")
        mapping_df = pd.read_csv(Path(data_folder) / "mapping.csv")
        with open(Path(data_folder) / "sampled_papers_rouge2_t2_llama_gnn_5_3.json", 'r') as f:
            reviews_5_3 = json.load(f)
        with open(Path(data_folder) / "sampled_papers_rouge2_t2_llama_gnn_5_5.json", 'r') as f:
            reviews_5_5 = json.load(f)
        return user_df, mapping_df, reviews_5_3, reviews_5_5
    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please check your './data' directory. Details: {e}")
        return None, None, None, None

# --- PDF Display Function ---
def display_pdf(pdf_path):
    """Displays a PDF file in the Streamlit app."""
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at path: {pdf_path}")
        return
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        # Embedding PDF in an iframe
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to display PDF. Reason: {e}")


# --- Main Application ---
st.title("📄 Human Evaluation Interface")

# Define paths
DATA_FOLDER = "./data"
PDF_FOLDER = "./pdfs"

# Load all data
user_df, mapping_df, reviews_5_3, reviews_5_5 = load_data(DATA_FOLDER)

if user_df is None:
    st.stop()

# --- User Selection ---
st.sidebar.header("👤 Annotator Selection")
print(user_df)
users = ["--- Select User ---"] + user_df["User"].tolist()
selected_user = st.sidebar.selectbox("Select your username:", users)

# --- Paper Selection (dynamically updated) ---
if selected_user != "--- Select User ---":
    st.sidebar.header("📜 Paper Selection")
    
    # Determine which papers to show in the dropdown
    if selected_user == 'master':
        # Master user sees all papers
        paper_ids = ["--- Select Paper ID ---"] + list(reviews_5_3.keys())
    else:
        # Regular user sees only their assigned papers
        user_papers = mapping_df[mapping_df['user'] == selected_user]
        if not user_papers.empty:
            paper_1 = user_papers['paper_1'].iloc[0]
            paper_2 = user_papers['paper_2'].iloc[0]
            paper_ids = ["--- Select Paper ID ---", paper_1, paper_2]
        else:
            st.sidebar.warning("No papers assigned to this user.")
            paper_ids = ["--- Select Paper ID ---"]
            
    selected_paper_id = st.sidebar.selectbox("Select a Paper ID to review:", paper_ids)
    
    # --- Display Area ---
    if selected_paper_id != "--- Select Paper ID ---":
        st.header(f"Reviewing Paper: `{selected_paper_id}`")
        
        # Define review data
        gold_review = reviews_5_3.get(selected_paper_id, {}).get("gold_review", "Not Available")
        review_5_3 = reviews_5_3.get(selected_paper_id, {}).get("inference_review", "Not Available")
        review_5_5 = reviews_5_5.get(selected_paper_id, {}).get("inference_review", "Not Available")
        
        # Create two main columns: one for the PDF, one for the reviews
        col_pdf, col_reviews = st.columns([1, 1])

        with col_pdf:
            st.subheader("📄 Original Paper")
            pdf_file_path = Path(PDF_FOLDER) / f"{selected_paper_id}.pdf"
            display_pdf(pdf_file_path)

        with col_reviews:
            st.subheader("📝 Reviews")
            
            st.markdown("#### ✨ Ground Truth Review")
            st.text_area("Ground Truth", value=gold_review, height=250, disabled=True)

            st.markdown("#### 🤖 Generated Review (Model 5_3)")
            st.text_area("Model 5_3", value=review_5_3, height=250, disabled=True)
            
            st.markdown("#### 🚀 Generated Review (Model 5_5)")
            st.text_area("Model 5_5", value=review_5_5, height=250, disabled=True)

else:
    st.info("Please select a user from the sidebar to begin.")
