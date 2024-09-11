from flask import Flask, request, render_template, redirect, url_for
import pickle
import re
import nltk
from io import BytesIO
from PyPDF2 import PdfReader

# Initialize Flask app
app = Flask(__name__)

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('resume')
        results = []

        for file in files:
            if file.filename.endswith('.pdf'):
                resume_text = read_pdf(file)
            else:
                resume_text = file.read().decode('utf-8', errors='ignore')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                0: "Advocate",
                1: "Arts",
                2: "Automation Testing",
                3: "Blockchain",
                4: "Business Analyst",
                5: "Civil Engineer",
                6: "Data Science",
                7: "Database",
                8: "DevOps Engineer",
                9: "DotNet Developer",
                10: "ETL Developer",
                11: "Electrical Engineering",
                12: "HR",
                13: "Hadoop",
                14: "Health and fitness",
                15: "Java Developer",
                16: "Mechanical Engineer",
                17: "Network Security Engineer",
                18: "Operations Manager",
                19: "PMO",
                20: "Python Developer",
                21: "SAP Developer",
                22: "Sales",
                23: "Testing",
                24: "Web Designing",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")
            results.append({
                'filename': file.filename,
                'category': category_name
            })

        return render_template('result.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
