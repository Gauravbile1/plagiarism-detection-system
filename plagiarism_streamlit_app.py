import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

# Load dataset
dataset_path = "C:/Users/Gaurav Bile/Videos/1Study/SKY internship/Plagiarism Detection System/plagiarism_dataset.txt"
with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = file.readlines()

# Preprocess input text
def preprocess_input_file(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Plagiarism calculation
def calculate_plagiarism(input_text, dataset):
    vectorizer = TfidfVectorizer()
    all_texts = dataset + [input_text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    plagiarism_score = max(cosine_sim_matrix[0]) * 100
    return plagiarism_score, cosine_sim_matrix[0]

# Highlight text
def highlight_text(input_text, cosine_sim_scores, threshold=0.2):
    sentences = input_text.split('. ')
    highlighted_sentences = []
    for i, sentence in enumerate(sentences):
        is_plagiarized = cosine_sim_scores[i] > threshold if i < len(cosine_sim_scores) else False
        highlighted_sentences.append((sentence, is_plagiarized))
    return highlighted_sentences

# PDF Report
def generate_pdf_report(score, highlighted_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph(f"<b>Plagiarism Report</b>", styles['Title'])
    score_para = Paragraph(f"<b>Plagiarism Score:</b> {score:.2f}%", styles['Normal'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    elements.append(score_para)
    elements.append(Spacer(1, 12))

    for sentence, is_plagiarized in highlighted_text:
        color = "red" if is_plagiarized else "green"
        formatted_sentence = f'<font color="{color}">{sentence}</font>'
        elements.append(Paragraph(formatted_sentence, styles['Normal']))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="Plagiarism Detector", page_icon="🧠", layout="wide")
st.title("🧠 Plagiarism Detection System")

st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload your .txt file for plagiarism check", type="txt")

# Sidebar
st.sidebar.title("🔍 Legend & Tools")
st.sidebar.markdown("""
### 🔴 Red  
Plagiarized sentences (similar to dataset)

### 🟢 Green  
Original sentences (no significant match)

---

### ⚙️ Technologies  
- Cosine Similarity  
- TF-IDF  
- Streamlit  
- PDF ReportLab  
""")

# Main logic
if uploaded_file is not None:
    input_text = preprocess_input_file(uploaded_file)
    plagiarism_score, cosine_sim_scores = calculate_plagiarism(input_text, dataset)
    highlighted_sentences = highlight_text(input_text, cosine_sim_scores)

    st.subheader(f"📊 Plagiarism Score: **{plagiarism_score:.2f}%**")
    st.markdown("---")
    st.subheader("🔎 Highlighted Sentences")

    for sentence, is_plagiarized in highlighted_sentences:
        color = "red" if is_plagiarized else "green"
        st.markdown(f'<p style="color:{color}; font-size:16px;">{sentence}</p>', unsafe_allow_html=True)

    st.markdown("---")
    pdf_buffer = generate_pdf_report(plagiarism_score, highlighted_sentences)
    st.download_button("⬇️ Download PDF Report", data=pdf_buffer, file_name="plagiarism_report.pdf", mime="application/pdf")
