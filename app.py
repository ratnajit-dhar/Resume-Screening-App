import streamlit as st
import pickle 
import re
import nltk
import docx
import PyPDF2
import string

model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl','rb'))

def CleanResume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    
    # Remove mentions (e.g., @username)
    txt = re.sub(r'@\S+', ' ', txt)
    
    # Remove hashtags (keeping words)
    txt = re.sub(r'#\S+', ' ', txt)
    
    # Remove RT (retweets) and common Twitter elements
    txt = re.sub(r'\bRT\b|cc', ' ', txt)
    
    # Remove special characters, punctuations
    txt = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', txt)
    
    # Remove numbers
    txt = re.sub(r'\d+', ' ', txt)
    
    # Remove extra whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    
    # Convert to lowercase
    txt = txt.lower()
    
    return txt
    
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

def handle_file_upload(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext == 'pdf':
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_ext == 'docx':
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error('Invalid file format. Please upload a PDF or DOCX file.')
    return resume_text

def pred(input_resume):
    input_resume = CleanResume(input_resume)
    input_resume = tfidf.transform([input_resume])
    input_resume = input_resume.toarray()
    pred = model.predict(input_resume)
    return le.inverse_transform(pred)[0]



def main():
    st.set_page_config(page_title='Resume Screening App', page_icon='📄', layout='wide')

    st.title('Resume Screening App')
    st.markdown('Upload a resume in PDF or DOCX format and get the predicted job category.')
    st.markdown('---')

    uploaded_file = st.file_uploader('Upload Resume', type=['pdf', 'docx'])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write('Successfully extracted the text from the uploaded resume')

            if st.checkbox('Show Resume Text', False):
                st.text_area('Resume Text', resume_text, height=400)

            if st.button('Submit'):
                st.subheader('Predicted Category')
                category = pred(resume_text)
                st.markdown(f'''
                    <div style="background-color: #000000; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong style="font-size: 24px;">{category}</strong>
                    </div>
                ''', unsafe_allow_html=True)

        except Exception as e:
            st.error(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
