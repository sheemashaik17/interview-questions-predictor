# app.py — InterviewIQ with Real ML Model
# ─────────────────────────────────────────────────────────────
# The ML model (ml/model.pkl) is loaded at startup.
# When a user uploads a resume, the model PREDICTS the job category
# from the resume text, then generates matching questions.
# ─────────────────────────────────────────────────────────────

import os, json, re
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist(): return datetime.now(IST).replace(tzinfo=None)
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
except: PyPDF2 = None

try:
    from docx import Document
except: Document = None

# ── App Setup ─────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'interviewiq-ml-secret'

# ── MySQL Config ───────────────────────────────────────────────
# Format: mysql+pymysql://username:password@host:port/database_name
MYSQL_USER     = "root"               # ← your MySQL username
MYSQL_PASSWORD = "170904"      # ← your MySQL password
MYSQL_HOST     = "localhost"
MYSQL_PORT     = "3306"
MYSQL_DB       = "interviewiq"        # ← database name (create this first)

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 280,   # reconnect before MySQL's wait_timeout
    'pool_pre_ping': True  # test connection before using it
}

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Groq API Config ────────────────────────────────────────────
GROQ_API_KEY = "gsk_bEgPcRgsJgUAn7dTzyKmWGdyb3FYBZWJyyUbU9cRITEe8u5gfDk1python app.py"   # ← paste your key here
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
app.jinja_env.globals['enumerate'] = enumerate

# ── Load Trained ML Model ─────────────────────────────────────
# The model was trained by ml/train_model.py and saved as model.pkl
# It predicts job category from resume text
MODEL_PATH = 'ml/model.pkl'

try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✅ ML model loaded from {MODEL_PATH}")
except FileNotFoundError:
    ml_model = None
    print(f"⚠️  Model not found. Run: python ml/train_model.py")

# ── Database Models ───────────────────────────────────────────
class User(UserMixin, db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100))
    email        = db.Column(db.String(150), unique=True)
    password     = db.Column(db.String(300))
    sessions     = db.relationship('Session', backref='user', lazy=True, cascade='all,delete-orphan')

class Session(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'))
    job_role         = db.Column(db.String(200))  # User entered role
    predicted_role   = db.Column(db.String(200))  # ML predicted role from resume
    confidence       = db.Column(db.Float)         # Model confidence %
    job_desc         = db.Column(db.Text)
    skills           = db.Column(db.Text)          # JSON list
    questions        = db.Column(db.Text)          # JSON list
    created_at       = db.Column(db.DateTime, default=now_ist)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

# ── Resume Reading ────────────────────────────────────────────
def read_resume(path):
    ext = path.rsplit('.', 1)[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            # pdfplumber preserves line breaks much better than PyPDF2
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
            except:
                if PyPDF2:
                    with open(path, 'rb') as f2:
                        reader = PyPDF2.PdfReader(f2)
                        for page in reader.pages:
                            text += (page.extract_text() or "") + "\n"
        elif ext == 'docx' and Document:
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
    except:
        text = ""
    return text

# ── ML Prediction: Predict job role from resume ───────────────
def predict_job_role(resume_text):
    """
    Uses the trained ML model to predict the job category
    from the resume text and returns:
    - predicted category (string)
    - confidence score (float 0-1)
    - all probabilities (dict)
    """
    if ml_model is None:
        return "Unknown", 0.0, {}

    predicted = ml_model.predict([resume_text])[0]
    probabilities = ml_model.predict_proba([resume_text])[0]
    classes = ml_model.classes_

    prob_dict = {cls: round(float(prob) * 100, 1)
                 for cls, prob in zip(classes, probabilities)}

    confidence = max(probabilities)
    return predicted, confidence, prob_dict

# ── Skill Detection ───────────────────────────────────────────
SKILLS = [
    "python","java","javascript","typescript","c++","c#","react","angular","vue",
    "nodejs","django","flask","spring","html","css","sql","mysql","postgresql",
    "mongodb","redis","docker","kubernetes","git","aws","azure","gcp","linux",
    "machine learning","deep learning","tensorflow","pytorch","scikit-learn",
    "pandas","numpy","rest api","graphql","agile","scrum","tableau",
    "data science","nlp","devops","ci/cd","kotlin","swift","flutter","dart",
    "fastapi","express","next.js","tailwind","bootstrap","firebase","hadoop","spark"
]

def find_skills(text):
    text = text.lower()
    return [s for s in SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text)]



# ── Groq AI Question Generation ────────────────────────────────
def generate_questions_groq(resume_text, job_role, job_desc, skills, predicted_role):
    """
    Sends resume + job role to Groq LLaMA3 and returns
    30 fully personalized interview questions as a list.
    """
    skills_str = ", ".join(skills) if skills else "not specified"

    system_prompt = """You are a senior technical interviewer with 15 years of experience.
Your job is to generate deep, specific, personalized interview questions by carefully reading the candidate's resume.
Never ask generic questions. Every question must be specific to what is written in the resume.
You must return ONLY a valid JSON array. No explanation, no markdown, no text before or after the JSON."""

    user_prompt = f"""Read this resume carefully and generate 30 interview questions for the role of {job_role}.

=== RESUME ===
{resume_text[:3500]}

=== JOB DESCRIPTION ===
{job_desc[:400] if job_desc else "Not provided"}

=== INSTRUCTIONS ===
Generate questions in this exact JSON format:
[
  {{"type": "Technical", "question": "your question here"}},
  ...
]

STRICT RULES:
1. Exactly 15 Technical questions — ask about specific technologies, frameworks, and concepts from THEIR resume (e.g. if they used Flask + MySQL, ask about Flask routing, SQLAlchemy, etc.)
2. Exactly 5 Project questions — mention the ACTUAL project names from their resume (e.g. "In your SNM Application, how did you implement OTP email verification?")
3. Exactly 5 Conceptual questions — CS fundamentals relevant to their role
4. Exactly 5 HR questions — behavioral questions relevant to their background
5. Total must be exactly 30 questions
6. Do NOT ask "Explain your experience with X" — ask real interview questions like "How does X work?" or "What challenges did you face with X?"
7. Return ONLY the JSON array, nothing else
"""

    try:
        print(f"Calling Groq API for {job_role}...")
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 4000
            },
            timeout=60
        )

        print(f"Groq status: {response.status_code}")

        if response.status_code == 200:
            raw = response.json()["choices"][0]["message"]["content"].strip()
            print(f"Groq raw response (first 200 chars): {raw[:200]}")
            # Strip markdown code fences if present
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            # Sometimes model adds text before the JSON — find the array
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            questions = json.loads(raw)
            if isinstance(questions, list) and len(questions) >= 20:
                print(f"✅ Groq returned {len(questions)} questions")
                return questions[:30]
            else:
                print(f"⚠️ Groq returned invalid list: {questions[:2]}")
        else:
            print(f"❌ Groq error {response.status_code}: {response.text[:300]}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e} — raw was: {raw[:300]}")
    except Exception as e:
        print(f"❌ Groq call failed: {e}")

    # This should never show — only if Groq completely fails
    print("⚠️ Using fallback questions")
    return [
        {"type": "Technical",  "question": f"Walk me through how you have used {skills[0] if skills else job_role} in a real project."},
        {"type": "Technical",  "question": f"What are the core concepts of {job_role} you rely on daily?"},
        {"type": "Technical",  "question": "Explain RESTful API design principles."},
        {"type": "Technical",  "question": "How do you handle errors and exceptions in your code?"},
        {"type": "Technical",  "question": "What is the difference between SQL and NoSQL databases?"},
        {"type": "Technical",  "question": "Explain the MVC design pattern with an example."},
        {"type": "Technical",  "question": "What is version control and how do you use Git in your workflow?"},
        {"type": "Technical",  "question": "How do you ensure your code is secure?"},
        {"type": "Technical",  "question": "What is the difference between authentication and authorization?"},
        {"type": "Technical",  "question": "Explain how HTTP requests and responses work."},
        {"type": "Technical",  "question": "What tools do you use for debugging?"},
        {"type": "Technical",  "question": "How do you optimize database queries?"},
        {"type": "Technical",  "question": "What is dependency injection and why is it useful?"},
        {"type": "Technical",  "question": "Explain the concept of middleware in web frameworks."},
        {"type": "Technical",  "question": "What is containerization and how does Docker help?"},
        {"type": "Project",    "question": "Walk me through your most complex project end to end."},
        {"type": "Project",    "question": "What was the biggest technical challenge you faced in your projects?"},
        {"type": "Project",    "question": "How did you handle user authentication in your projects?"},
        {"type": "Project",    "question": "What database design decisions did you make and why?"},
        {"type": "Project",    "question": "How did you deploy your projects and what tools did you use?"},
        {"type": "Conceptual", "question": "Explain OOP principles with real examples from your code."},
        {"type": "Conceptual", "question": "What is time and space complexity? Give an example."},
        {"type": "Conceptual", "question": "Explain the difference between synchronous and asynchronous programming."},
        {"type": "Conceptual", "question": "What are design patterns and which ones have you used?"},
        {"type": "Conceptual", "question": "What is Agile methodology and how have you applied it?"},
        {"type": "HR",         "question": "Tell me about yourself and your journey as a developer."},
        {"type": "HR",         "question": "Why are you applying for this role specifically?"},
        {"type": "HR",         "question": "Describe a time you had to learn a new technology quickly."},
        {"type": "HR",         "question": "How do you handle constructive criticism of your code?"},
        {"type": "HR",         "question": "Where do you see yourself growing in the next 2 years?"},
    ]

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name  = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        pwd   = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'error')
        elif len(pwd) < 6:
            flash('Password must be at least 6 characters.', 'error')
        else:
            db.session.add(User(name=name, email=email, password=generate_password_hash(pwd)))
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email'].strip().lower()).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    sessions = Session.query.filter_by(user_id=current_user.id).order_by(Session.created_at.desc()).limit(5).all()
    total    = Session.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', sessions=sessions, total=total)

@app.route('/prepare', methods=['GET','POST'])
@login_required
def prepare():
    if request.method == 'POST':
        job_role = request.form.get('job_role','').strip()
        job_desc = request.form.get('job_desc','').strip()
        file     = request.files.get('resume')

        if not job_role:
            flash('Please enter a job role.', 'error')
            return render_template('prepare.html')
        if not file or not file.filename.endswith(('.pdf','.docx')):
            flash('Please upload a PDF or DOCX resume.', 'error')
            return render_template('prepare.html')

        # Save resume file
        fname = secure_filename(f"{current_user.id}_{int(datetime.now().timestamp())}_{file.filename}")
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        # Extract text from resume
        resume_text = read_resume(fpath)

        # ── ML PREDICTION ──────────────────────────────────────
        # The trained model predicts job category from resume text
        predicted_role, confidence, all_probs = predict_job_role(resume_text)

        # Detect skills
        skills = find_skills(resume_text + ' ' + job_desc)

        # Generate 30 questions using Groq AI (falls back to rule-based if API fails)
        questions = generate_questions_groq(resume_text, job_role, job_desc, skills, predicted_role)

        # Save session
        s = Session(
            user_id        = current_user.id,
            job_role       = job_role,
            predicted_role = predicted_role,
            confidence     = round(confidence * 100, 1),
            job_desc       = job_desc[:300],
            skills         = json.dumps(skills),
            questions      = json.dumps(questions),
        )
        db.session.add(s)
        db.session.commit()
        return redirect(url_for('result', sid=s.id))

    return render_template('prepare.html')

@app.route('/result/<int:sid>')
@login_required
def result(sid):
    s = Session.query.get_or_404(sid)
    if s.user_id != current_user.id: return redirect(url_for('dashboard'))
    questions = json.loads(s.questions)
    skills    = json.loads(s.skills)
    return render_template('result.html', s=s, questions=questions, skills=skills)

@app.route('/history')
@login_required
def history():
    sessions = Session.query.filter_by(user_id=current_user.id).order_by(Session.created_at.desc()).all()
    return render_template('history.html', sessions=sessions)

@app.route('/history/<int:sid>/questions')
@login_required
def get_questions(sid):
    s = Session.query.get_or_404(sid)
    if s.user_id != current_user.id: return jsonify({}), 403
    return jsonify({'questions': json.loads(s.questions), 'job_role': s.job_role})

@app.route('/history/<int:sid>/delete', methods=['POST'])
@login_required
def delete_session(sid):
    s = Session.query.get_or_404(sid)
    if s.user_id == current_user.id:
        db.session.delete(s)
        db.session.commit()
        flash('Session deleted.', 'success')
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)