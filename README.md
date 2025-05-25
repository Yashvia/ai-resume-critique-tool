AI Resume Critique Pro
A Streamlit-based web application that analyzes resumes for grammar, readability, sentiment, and job fit using AI-powered tools. Upload a resume (PDF or TXT) and provide a job title to receive a detailed critique, including a radar chart visualization and a downloadable PDF report.
Features

Grammar Analysis: Evaluates resume text for grammatical errors using TextBlob.
Readability Score: Calculates the Flesch-Kincaid grade level for text clarity.
Sentiment Analysis: Assesses the tone of the resume (positive, neutral, negative).
Job Fit Score: Measures semantic similarity between the resume and a generated job description using Sentence Transformers.
Improvement Suggestions: Provides actionable suggestions using the Phi-3-mini model from Hugging Face.
Visualization: Generates a radar chart to visualize resume metrics.
PDF Report: Downloads a detailed report summarizing the analysis.

Installation

Clone the Repository:
git clone https://github.com/your-username/resume-critique-pro.git
cd resume-critique-pro


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Set Up Environment Variables:

Create a .env file in the project root.
Add your Hugging Face API key: HUGGINGFACE_API_KEY=your_huggingface_api_key


Obtain an API key from Hugging Face.


Run the Application:
streamlit run app.py



Usage

Open the app in your browser (typically at http://localhost:8501).
Upload a resume in PDF or TXT format.
Enter a job title (e.g., "ASIC Design Engineer").
Click "Analyze Resume" to generate the critique.
View the analysis results, including metrics, a radar chart, and improvement suggestions.
Download the PDF report for a detailed summary.

Project Structure
resume-critique-pro/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .gitignore          # Files to ignore in version control
├── .env               # Environment variables (not tracked)
└── README.md           # Project documentation

Dependencies
See requirements.txt for the full list of Python packages. Key dependencies include:

Streamlit: Web app framework
PyPDF2: PDF text extraction
TextBlob: Grammar and sentiment analysis
Sentence Transformers: Semantic similarity scoring
Hugging Face Hub: AI model for suggestions
Textstat: Readability scoring
Matplotlib & Seaborn: Visualization
ReportLab: PDF generation
python-dotenv: Environment variable management

Deployment
To deploy on Streamlit Cloud:

Connect your GitHub repository to Streamlit Cloud.
Select the repository and branch (e.g., main).
Specify app.py as the main file.
Add the HUGGINGFACE_API_KEY as a secret in Streamlit Cloud’s settings.
Deploy the app.

Security Notes

The .env file contains sensitive information (API key) and is excluded from version control via .gitignore.
Never commit the .env file to GitHub. If exposed, regenerate your API key on Hugging Face.

Contributing
Feel free to submit issues or pull requests via GitHub. Ensure any changes include updated documentation and tests where applicable.


