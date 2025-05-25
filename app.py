import streamlit as st
from PyPDF2 import PdfReader
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Critique Pro", layout="wide")

# Load models once
@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return  sentence_model

model = load_models()

# Extract text from uploaded resume
def extract_resume_text(uploaded_file):
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or '' for page in reader.pages)
        elif uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format. Please upload a PDF or TXT file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Grammar score with TextBlob (reverted to original)
def grammar_score(text):
    try:
        blob = TextBlob(text)
        num_sentences = len(blob.sentences)
        num_errors = 0
        suggestions = []

        for sentence in blob.sentences:
            corrected = sentence.correct()
            if sentence != corrected:
                suggestions.append(f"Original: {sentence}\nSuggestion: {corrected}")
                if abs(len(sentence.words) - len(corrected.words)) > 1:
                    num_errors += 1

        grammar_accuracy = max(0, 100 - (num_errors / num_sentences) * 100) if num_sentences else 100
        return round(grammar_accuracy, 2), num_errors, suggestions
    except Exception as e:
        st.error(f"Error in grammar checking: {str(e)}")
        return 100, 0, ["Unable to perform grammar check."]

# Readability score using Flesch-Kincaid
def readability_score(text):
    try:
        return round(textstat.flesch_kincaid_grade(text), 2)
    except:
        return None

# Sentiment analysis for tone
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Keyword extraction and NER using spaCy

# Semantic similarity score
def semantic_similarity_score(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity[0][0]) * 100, 2)

# Generate improvement suggestions using Phi-3-mini
def generate_suggestions(resume_text, job_description, grammar_suggestions, semantic_score, readability, sentiment):
    try:
        client = InferenceClient(
            provider="nebius",
            api_key="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",
        )
        prompt = f"""
        Given a resume with the following metrics:
        - Grammar Accuracy: {grammar_suggestions[0]}% ({grammar_suggestions[1]} errors)
        - Semantic Fit Score: {semantic_score}%
        - Readability (Flesch-Kincaid Grade): {readability}
        - Sentiment: {sentiment[0]} (Polarity: {sentiment[1]})
        - Job Description: {job_description[:500]}...

        Provide 3-5 specific, actionable suggestions to improve the resume for better alignment with the job description. Keep each suggestion concise and practical.
        """
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200
        )
        suggestions = completion.choices[0].message.content.split("\n")
        return [s.strip() for s in suggestions if s.strip()][:5]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return ["Unable to generate suggestions due to an error."]


def generate_radar_chart(grammar_accuracy, semantic_score, readability, sentiment_polarity):
    categories = ['Grammar', 'Job Fit', 'flesch Kincaid Grade Score', 'Sentiment']
    values = [
        grammar_accuracy / 100,
        semantic_score / 100,
        min(readability / 12, 1.0),  
        (sentiment_polarity + 1) / 2  
    ]
    values += values[:1] 
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Resume Analysis Radar", size=20, color='blue', y=1.1)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return buf.getvalue()

# Generate PDF report
def generate_pdf_report(resume_text, job_description, grammar_accuracy, num_errors, semantic_score, readability, sentiment, suggestions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "AI Resume Critique Report")
    c.setFont("Helvetica", 10)
    
    y = 700
    c.drawString(50, y, f"Job Title: {st.session_state.role_input}")
    y -= 20
    c.drawString(50, y, "Summary:")
    y -= 20
    c.drawString(50, y, f"Grammar Accuracy: {grammar_accuracy}% ({num_errors} errors)")
    y -= 20
    c.drawString(50, y, f"Semantic Fit Score: {semantic_score}%")
    y -= 20
    c.drawString(50, y, f"Readability (Flesch-Kincaid): {readability}")
    y -= 20
    c.drawString(50, y, f"Sentiment: {sentiment[0]} (Polarity: {round(sentiment[1], 2)})")
    y -= 30
    c.drawString(50, y, "Improvement Suggestions:")
    for i, suggestion in enumerate(suggestions[:5], 1):
        y -= 15
        c.drawString(50, y, f"{i}. {suggestion}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


st.title("📄 AI Resume Critique Pro")
st.markdown("Upload your resume and enter a job title to receive an in-depth critique, including grammar, readability, sentiment, and job fit analysis.")

# File uploader and job title input
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
with col2:
    role_input = st.text_input("Job title (e.g., ASIC Design Engineer):")

if uploaded_file and role_input:
    st.session_state.role_input = role_input
    st.success("✅ Resume and job title received.")

    if st.button("🔍 Analyze Resume"):
        resume_text = extract_resume_text(uploaded_file)
        if not resume_text:
            st.stop()

        
        progress = st.progress(0)
        progress.progress(10)

      
        with st.spinner("Generating job description..."):
            try:
                client = InferenceClient(
                    provider="nebius",
                    api_key="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",
                )
                completion = client.chat.completions.create(
                    model="microsoft/Phi-3-mini-4k-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": f"What skills, experiences, projects, and educational qualifications are essential for a {role_input}?"
                        }
                    ],
                )
                job_description = completion.choices[0].message.content
                progress.progress(30)
            except Exception as e:
                st.error(f"Error generating job description: {str(e)}")
                st.stop()

        st.markdown(f"**🧑‍💼 Generated Job Description for `{role_input}`:**\n\n{job_description}")

        # Perform analyses
        grammar_accuracy, num_errors, grammar_suggestions = grammar_score(resume_text)
        progress.progress(50)
        semantic_score = semantic_similarity_score(resume_text, job_description)
        progress.progress(70)
        readability = readability_score(resume_text) or "N/A"
        sentiment = sentiment_analysis(resume_text)
        progress.progress(90)
        suggestions = generate_suggestions(resume_text, job_description, (grammar_accuracy, num_errors, grammar_suggestions), semantic_score, readability, sentiment)
        progress.progress(100)

        
        final_score = round((grammar_accuracy * 0.25 + semantic_score * 0.6 + (12 - min(readability, 12)) / 12 * 100 * 0.1 + (sentiment[1] + 1) / 2 * 100 * 0.05), 2) if readability != "N/A" else round((grammar_accuracy * 0.3 + semantic_score * 0.7), 2)

       
        st.subheader("📊 Resume Critique Summary")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Grammar Accuracy", f"{grammar_accuracy}%", f"{num_errors} errors")
            st.metric("Semantic Fit Score", f"{semantic_score}%")
            st.metric("Final Resume Fit Score", f"{final_score}/100")
        with col2:
            st.metric("Readability (Flesch-Kincaid Grade [0-18])", f"{readability}" if readability != "N/A" else "N/A")
            st.metric("Sentiment", f"{sentiment[0]}", f"Polarity: {round(sentiment[1], 2)}")

        
        radar_image = generate_radar_chart(grammar_accuracy, semantic_score, readability if readability != "N/A" else 12, sentiment[1])
        st.image(radar_image, caption="Resume Metrics Visualization")

        
        if final_score > 80:
            st.success("✅ Excellent resume! You're highly aligned for this role.")
        elif final_score > 60:
            st.warning("⚠️ Good match, but consider refining grammar, readability, or keywords.")
        else:
            st.error("❌ Needs improvement — tailor your resume to better match the job.")

        
        with st.expander("🔍 Detailed Analysis"):
            st.markdown("**Improvement Suggestions:**")
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")

        
        pdf_buffer = generate_pdf_report(resume_text, job_description, grammar_accuracy, num_errors, semantic_score, readability, sentiment, suggestions)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name="resume_critique_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Please upload a resume and enter a job title to begin.")