import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Job Recommendation System",
    layout="centered"
)

# --------------------------------------------------
# CSS Styling
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1rem;
}

.card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    margin: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    color: black;
    text-align: center;
}

.top-job {
    border-left: 6px solid #ffcc00;
    background-color: #fff9e6;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 6px;
    height: 38px;
    width: 220px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title (ONLY ONE TITLE)
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center; color:white; display:flex; justify-content:center; align-items:center; gap:12px;'>
<img src="https://cdn-icons-png.flaticon.com/512/3135/3135755.png" width="45" alt="graduation cap emoji">
AI Job Recommendation System
</h1>
""", unsafe_allow_html=True)


st.markdown("<p style='text-align:center; color:white;'>Select your degree and skills</p>", unsafe_allow_html=True)

# --------------------------------------------------
# Degree Selection
# --------------------------------------------------
degree = st.selectbox(
    "Select Your Degree",
    ["BBA", "BCA", "BA", "BCom"]
)

# --------------------------------------------------
# Degree Data
# --------------------------------------------------
degree_data = {
    "BBA": {
        "jobs": ["Marketing Executive", "HR Executive", "Financial Analyst", "Business Development Executive"],
        "skills": [
            "marketing sales communication branding",
            "recruitment hr management leadership",
            "finance accounting excel analysis",
            "business strategy negotiation communication"
        ],
        "salary": ["3-5 LPA", "3-4 LPA", "4-6 LPA", "3-5 LPA"],
        "growth": [
            "Marketing Manager → Brand Manager → CMO",
            "HR Manager → HR Head → HR Director",
            "Senior Analyst → Finance Manager → CFO",
            "Sales Manager → Regional Head → Director"
        ]
    },
    "BCA": {
        "jobs": ["Web Developer", "Software Developer", "System Administrator", "Data Analyst"],
        "skills": [
            "html css javascript react",
            "java python c++ coding",
            "linux networking troubleshooting",
            "python sql excel data analysis"
        ],
        "salary": ["3-6 LPA", "4-8 LPA", "3-5 LPA", "4-7 LPA"],
        "growth": [
            "Senior Developer → Tech Lead → Engineering Manager",
            "Senior Developer → Architect → CTO",
            "IT Manager → Infrastructure Head",
            "Senior Analyst → Data Scientist → AI Engineer"
        ]
    },
    "BA": {
        "jobs": ["Content Writer", "Social Media Manager", "Public Relations Officer", "Journalist"],
        "skills": [
            "writing communication creativity research",
            "social media marketing communication",
            "public speaking communication management",
            "reporting writing research"
        ],
        "salary": ["2-4 LPA", "3-5 LPA", "3-6 LPA", "3-5 LPA"],
        "growth": [
            "Senior Writer → Content Head",
            "Digital Marketing Manager → Brand Head",
            "PR Manager → Communications Director",
            "Editor → Bureau Chief"
        ]
    },
    "BCom": {
        "jobs": ["Accountant", "Banking Associate", "Tax Consultant", "Auditor"],
        "skills": [
            "accounting tally gst finance",
            "banking finance customer service",
            "taxation gst filing accounting",
            "auditing accounting compliance"
        ],
        "salary": ["3-5 LPA", "3-6 LPA", "4-7 LPA", "4-6 LPA"],
        "growth": [
            "Senior Accountant → Finance Manager",
            "Branch Manager → Regional Manager",
            "Senior Consultant → Tax Advisor",
            "Senior Auditor → Audit Manager"
        ]
    }
}

# --------------------------------------------------
# Job Images
# --------------------------------------------------
job_images = {
    "Marketing Executive": "https://cdn-icons-png.flaticon.com/512/1995/1995574.png",
    "HR Executive": "https://cdn-icons-png.flaticon.com/512/1077/1077012.png",
    "Financial Analyst": "https://cdn-icons-png.flaticon.com/512/2331/2331970.png",
    "Business Development Executive": "https://cdn-icons-png.flaticon.com/512/3062/3062634.png",

    "Web Developer": "https://cdn-icons-png.flaticon.com/512/2721/2721297.png",
    "Software Developer": "https://cdn-icons-png.flaticon.com/512/6062/6062646.png",
    "System Administrator": "https://cdn-icons-png.flaticon.com/512/4248/4248443.png",
    "Data Analyst": "https://cdn-icons-png.flaticon.com/512/4149/4149678.png",

    "Content Writer": "https://cdn-icons-png.flaticon.com/512/2991/2991148.png",
    "Social Media Manager": "https://cdn-icons-png.flaticon.com/512/1384/1384060.png",
    "Public Relations Officer": "https://cdn-icons-png.flaticon.com/512/2920/2920254.png",
    "Journalist": "https://cdn-icons-png.flaticon.com/512/3011/3011270.png",

    "Accountant": "https://cdn-icons-png.flaticon.com/512/2920/2920256.png",
    "Banking Associate": "https://cdn-icons-png.flaticon.com/512/2830/2830284.png",
    "Tax Consultant": "https://cdn-icons-png.flaticon.com/512/4256/4256900.png",
    "Auditor": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
}

# --------------------------------------------------
# Skill Selection
# --------------------------------------------------
available_skills = list(set(" ".join(degree_data[degree]["skills"]).split()))
selected_skills = st.multiselect("Select Your Skills", available_skills)

# --------------------------------------------------
# Button
# --------------------------------------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    recommend_btn = st.button("Get Recommendations")

# --------------------------------------------------
# Recommendation Logic
# --------------------------------------------------
if recommend_btn:

    if selected_skills:

        user_input = " ".join(selected_skills)

        df = pd.DataFrame({
            "Job Title": degree_data[degree]["jobs"],
            "Skills": degree_data[degree]["skills"],
            "Salary": degree_data[degree]["salary"],
            "Growth": degree_data[degree]["growth"]
        })

        skills_data = df["Skills"].tolist()
        skills_data.append(user_input)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(skills_data)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        scores = similarity.flatten()

        df["Similarity"] = scores
        recommended_jobs = df.sort_values(by="Similarity", ascending=False)

        st.subheader("Top Recommended Jobs")

        top_3 = recommended_jobs.head(3).reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for index, row in top_3.iterrows():

            score_percent = round(row['Similarity'] * 100, 1)
            highlight_class = "card top-job" if index == 0 else "card"
            image_url = job_images.get(row['Job Title'], "")

            with cols[index]:
                st.markdown(f"""
                <div class="{highlight_class}">
                    <img src="{image_url}" width="60"><br>
                    <h3>{row['Job Title']}</h3>
                    <p><b>Match Score:</b> {score_percent}%</p>
                    <p><b>Average Salary:</b> {row['Salary']}</p>
                    <p><b>Career Growth:</b> {row['Growth']}</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("Please select at least one skill")