import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Simplified clean_text without NLTK
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stopwords = set([
        "the", "and", "is", "in", "to", "of", "for", "on", "with", "as", "by",
        "at", "this", "that", "a", "an", "be", "or", "it", "from"
    ])
    filtered = [word for word in words if word not in stopwords]
    return " ".join(filtered)

@st.cache_data
def load_data():
    courses_df = pd.read_csv("courses.csv")
    courses_df["Clean_Course"] = (
        courses_df["Required_Subjects"] + " " +
        courses_df["Skills_Taught"] + " " +
        courses_df["Description"]
    ).apply(clean_text)
    return courses_df

def recommend_courses(subjects, grade, interest, goal, courses_df, top_n=3):
    student_profile = clean_text(subjects + " " + interest + " " + goal)

    vectorizer = TfidfVectorizer()
    combined_text = pd.concat([pd.Series([student_profile]), courses_df["Clean_Course"]])
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    student_vec = tfidf_matrix[0]
    course_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(student_vec, course_vecs).flatten()
    boost_factor = 1.3

    # Only eligible courses
    eligible = courses_df[courses_df["Min_Grade"] <= grade].copy()
    eligible["similarity"] = similarities[eligible.index]
    eligible.loc[eligible["Field"] == interest, "similarity"] *= boost_factor

    top_courses = eligible.sort_values("similarity", ascending=False).head(top_n)
    return top_courses[["Course_Name", "Description", "Min_Grade", "similarity"]]

# Streamlit UI
st.set_page_config(page_title="Kenya University Course Recommender")
st.title("🎓 University Course Recommender - Kenya")

st.markdown("""
Enter your profile below to discover suitable university courses based on your subjects, interests, and grades.
""")

# Inputs
subjects = st.text_input("📘 Subjects (comma-separated):", "Math, Physics, Chemistry")
grade = st.slider("📊 Mean Grade / Cluster Points:", 26.0, 48.0, 38.0)
interest = st.selectbox("🎯 Preferred Field:", [
    "Engineering", "IT", "Health Sciences", "Business", "Agriculture", "Education",
    "Math & Stats", "Law", "Environmental Science", "Social Sciences"
])
goal = st.text_input("💡 Career Goal:", "Engineer")

# Recommend
courses_df = load_data()
if st.button("🔍 Recommend Courses"):
    recs = recommend_courses(subjects, grade, interest, goal, courses_df)
    st.success(f"Top {len(recs)} course matches found:")
    st.dataframe(recs.reset_index(drop=True))
