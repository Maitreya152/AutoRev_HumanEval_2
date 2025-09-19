import streamlit as st
import pandas as pd
import json
import os
import re
import random
from pathlib import Path
from datetime import datetime

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
        user_df.columns = user_df.columns.str.strip()
        mapping_df = pd.read_csv(Path(data_folder) / "mapping.csv")
        mapping_df.columns = mapping_df.columns.str.strip()
        with open(Path(data_folder) / "sampled_papers_rouge2_t2_llama_gnn_5_3.json", 'r', encoding='utf-8') as f:
            reviews_5_3 = json.load(f)
        with open(Path(data_folder) / "sampled_papers_rouge2_t2_llama_gnn_5_5.json", 'r', encoding='utf-8') as f:
            reviews_5_5 = json.load(f)
        return user_df, mapping_df, reviews_5_3, reviews_5_5
    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please check your './data' directory. Details: {e}")
        return None, None, None, None

# --- Review Parsing Function ---
def parse_review(review_text):
    """
    Parses a review string into a dictionary of sections and bullet points.
    """
    if not isinstance(review_text, str):
        return {
            "Summary": ["Not Available"], "Strengths": [], "Weaknesses": [], "Questions": []
        }

    sections = {
        "Summary": [], "Strengths": [], "Weaknesses": [], "Questions": []
    }
    
    summary_match = re.search(r'\*\*Summary\*\*(.*?)(?=\*\*Strengths\*\*|\Z)', review_text, re.DOTALL)
    strengths_match = re.search(r'\*\*Strengths\*\*(.*?)(?=\*\*Weaknesses\*\*|\Z)', review_text, re.DOTALL)
    weaknesses_match = re.search(r'\*\*Weaknesses\*\*(.*?)(?=\*\*Questions\*\*|\Z)', review_text, re.DOTALL)
    questions_match = re.search(r'\*\*Questions\*\*(.*)', review_text, re.DOTALL)

    if summary_match and summary_match.group(1).strip():
        sections["Summary"].append(summary_match.group(1).strip())

    for match, section_name in [(strengths_match, "Strengths"), (weaknesses_match, "Weaknesses"), (questions_match, "Questions")]:
        if match:
            content = match.group(1).strip()
            # Split by newline and hyphen, then clean up each point
            raw_points = content.split('\n-')
            cleaned_points = []
            for point in raw_points:
                cleaned = point.strip()
                # Remove a leading hyphen if it exists from the split
                if cleaned.startswith('-'):
                    cleaned = cleaned[1:].strip()
                # Only add non-empty points
                if cleaned:
                    cleaned_points.append(cleaned)
            sections[section_name] = cleaned_points
            
    return sections

# --- Display Functions ---
def display_pdf(pdf_path):
    """Displays a PDF file and provides a download button."""
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at path: {pdf_path}")
        return
    try:
        # Provide a download button for the PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )
        
        # Use st.pdf() which is more robust and efficient for large files.
        st.pdf(pdf_path, height=1000)

    except Exception as e:
        st.error(f"Failed to display or provide download for PDF. Reason: {e}")


def display_review_form(title, review_data, review_type):
    """Displays a parsed review as a form with rating dropdowns."""
    st.markdown(f"#### {title}")
    rating_options = ["--- Select ---", "Agree", "Mostly Agree", "Mostly Disagree", "Disagree"]
    with st.container(border=True):
        if review_data.get("Summary"):
            st.markdown("**Summary**")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(review_data["Summary"][0])
            with col2:
                st.selectbox(
                    "Rating", 
                    options=rating_options, 
                    key=f"{review_type}_Summary_0",
                    label_visibility="collapsed"
                )

        for section_name in ["Strengths", "Weaknesses", "Questions"]:
            if review_data.get(section_name):
                st.markdown(f"**{section_name}**")
                for i, point in enumerate(review_data[section_name]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"- {point}")
                    with col2:
                        st.selectbox(
                            "Rating", 
                            options=rating_options, 
                            key=f"{review_type}_{section_name}_{i}",
                            label_visibility="collapsed"
                        )

def save_results(results_path, records):
    """Saves evaluation records to a CSV file."""
    new_data = pd.DataFrame(records)
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        combined_df = pd.concat([results_df, new_data], ignore_index=True)
        combined_df.to_csv(results_path, index=False)
    else:
        new_data.to_csv(results_path, index=False)

def check_if_rated(results_path, user, paper_id):
    """Checks if a user has already rated a specific paper."""
    if not os.path.exists(results_path):
        return False
    results_df = pd.read_csv(results_path)
    return not results_df[(results_df['user'] == user) & (results_df['paper_id'] == paper_id)].empty


# --- Main Application ---
st.title("📄 Human Evaluation Interface")

DATA_FOLDER = Path("./data")
PDF_FOLDER = Path("./pdfs")
RESULTS_CSV_PATH = DATA_FOLDER / 'evaluation_results.csv'

user_df, mapping_df, reviews_5_3, reviews_5_5 = load_data(DATA_FOLDER)

if user_df is None:
    st.stop()

st.sidebar.header("👤 Annotator Selection")
users = ["--- Select User ---"] + user_df['User'].tolist()
selected_user = st.sidebar.selectbox("Select your username:", users)

if selected_user != "--- Select User ---":
    st.sidebar.header("📜 Paper Selection")
    
    paper_ids = ["--- Select Paper ID ---"]
    if selected_user == 'master':
        paper_ids += sorted(list(reviews_5_3.keys()))
    else:
        user_papers = mapping_df[mapping_df['user'] == selected_user]
        if not user_papers.empty:
            paper_ids += sorted([user_papers['paper_1'].iloc[0], user_papers['paper_2'].iloc[0]])
        else:
            st.sidebar.warning("No papers assigned to this user.")
            
    selected_paper_id = st.sidebar.selectbox("Select a Paper ID to review:", paper_ids, key=f"paper_select_{selected_user}")
    
    if selected_paper_id != "--- Select Paper ID ---":
        st.header(f"Reviewing Paper: `{selected_paper_id}`")
        
        already_rated = check_if_rated(RESULTS_CSV_PATH, selected_user, selected_paper_id)
        if already_rated:
            st.success(f"✅ You have already submitted ratings for this paper ({selected_paper_id}).")

        # --- PDF Viewer Section ---
        st.subheader("📄 Original Paper")
        pdf_file_path = PDF_FOLDER / f"{selected_paper_id}.pdf"
        display_pdf(pdf_file_path)
        st.divider()

        # --- Reviews Section ---
        st.subheader("📝 Rate the Reviews")
        
        gold_review = reviews_5_3.get(selected_paper_id, {}).get("gold_review", "Not Available")
        review_5_3 = reviews_5_3.get(selected_paper_id, {}).get("inference_review", "Not Available")
        review_5_5 = reviews_5_5.get(selected_paper_id, {}).get("inference_review", "Not Available")
        
        parsed_reviews = {
            "gold": parse_review(gold_review),
            "model_5_3": parse_review(review_5_3),
            "model_5_5": parse_review(review_5_5)
        }

        # --- Randomization Logic ---
        session_key = f'review_order_{selected_user}_{selected_paper_id}'
        if session_key not in st.session_state:
            reviews_to_display = [
                {"original_key": "gold", "data": parsed_reviews["gold"]},
                {"original_key": "model_5_3", "data": parsed_reviews["model_5_3"]},
                {"original_key": "model_5_5", "data": parsed_reviews["model_5_5"]}
            ]
            random.shuffle(reviews_to_display)
            st.session_state[session_key] = reviews_to_display
        
        shuffled_reviews = st.session_state[session_key]
        
        with st.form(key=f"review_form_{selected_paper_id}"):
            for i, review_item in enumerate(shuffled_reviews):
                display_review_form(f"Review {chr(65+i)}", review_item["data"], review_item["original_key"])
                if i < len(shuffled_reviews) - 1: # Add a divider between reviews
                    st.divider()

            submitted = st.form_submit_button("Submit Ratings", disabled=already_rated)

            if submitted:
                all_fields_rated = True
                for review_item in shuffled_reviews:
                    review_type = review_item["original_key"]
                    parsed_data = review_item["data"]
                    if not all_fields_rated: break
                    if parsed_data.get("Summary"):
                        if st.session_state[f"{review_type}_Summary_0"] == "--- Select ---":
                            all_fields_rated = False
                    
                    for section in ["Strengths", "Weaknesses", "Questions"]:
                        if not all_fields_rated: break
                        for i in range(len(parsed_data.get(section, []))):
                            if st.session_state[f"{review_type}_{section}_{i}"] == "--- Select ---":
                                all_fields_rated = False
                                break
                
                if not all_fields_rated:
                    st.error("Validation Failed: Please rate all points for all reviews before submitting.")
                else:
                    records = []
                    for review_item in shuffled_reviews:
                        review_type = review_item["original_key"]
                        parsed_data = review_item["data"]

                        if parsed_data.get("Summary"):
                            records.append({
                                "timestamp": datetime.now().isoformat(), "user": selected_user,
                                "paper_id": selected_paper_id, "review_type": review_type,
                                "section": "Summary", "point_index": 0,
                                "point_text": parsed_data["Summary"][0],
                                "rating": st.session_state[f"{review_type}_Summary_0"]
                            })
                        
                        for section in ["Strengths", "Weaknesses", "Questions"]:
                            for i, point_text in enumerate(parsed_data.get(section, [])):
                                rating = st.session_state[f"{review_type}_{section}_{i}"]
                                records.append({
                                    "timestamp": datetime.now().isoformat(), "user": selected_user,
                                    "paper_id": selected_paper_id, "review_type": review_type,
                                    "section": section, "point_index": i,
                                    "point_text": point_text, "rating": rating
                                })
                    
                    save_results(RESULTS_CSV_PATH, records)
                    st.success("Your ratings have been saved successfully!")
                    # Clear the session state for this paper to allow for re-randomization if visited again
                    del st.session_state[session_key]
                    st.rerun()

else:
    st.info("Please select a user from the sidebar to begin.")