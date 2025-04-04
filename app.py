import os
import re
import PyPDF2
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['STATIC_FOLDER'] = 'static'

# Authentication Credentials
VALID_USERNAME = "1234"
VALID_PASSWORD = "5678"

def extract_marks_from_pdf(pdf_path):
    """Extract academic data from marksheet PDFs with enhanced parsing and error handling"""
    data = {
        'marks': {},
        'credits': {},
        'grades': {},
        'grade_points': {},
        'sgpa': None,
        'cgpa': None,
        'semester': None,
        'result_status': 'PASS'
    }

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])

            # Extract semester information
            semester_match = re.search(r'Semester\s*:\s*([IVXLCDM]+)', text, re.I)
            if semester_match:
                data['semester'] = semester_match.group(1)

            # Extract SGPA and CGPA with safety checks
            sgpa_match = re.search(r'S\.G\.P\.A\s+(\d+\.\d+)', text)
            if sgpa_match:
                try:
                    data['sgpa'] = float(sgpa_match.group(1))
                except (ValueError, TypeError):
                    pass

            cgpa_match = re.search(r'C\.G\.P\.A\s+(\d+\.\d+)', text)
            if cgpa_match:
                try:
                    data['cgpa'] = float(cgpa_match.group(1))
                except (ValueError, TypeError):
                    pass

            data['result_status'] = 'DISTINCTION' if 'DISTINCTION' in text else 'PASS'

            # Enhanced subject pattern with error handling
            subject_pattern = re.compile(
                r'(\d+)\.\s+([A-Z][A-Z\s\&\-]+?)\s+([A-Z0-9]+)\s+(?:\d+\s+){7}(\d+)\s+([A-Z\+]+)\s+(\d+)',
                re.DOTALL
            )
            
            for match in subject_pattern.finditer(text):
                try:
                    subject = re.sub(r'\s+', ' ', match.group(2).strip())
                    data['marks'][subject] = int(match.group(4))
                    data['credits'][subject] = int(match.group(5))
                    data['grades'][subject] = match.group(6)
                    data['grade_points'][subject] = int(match.group(7))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing subject data: {str(e)}")
                    continue

    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
    
    return data

def generate_performance():
    """Process all marksheets and generate performance data with comprehensive error handling"""
    pdf_dir = os.path.join(app.config['STATIC_FOLDER'], 'pdfs')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        return {
            'performance_data': {},
            'cgpa_history': {},
            'latest_semester': None,
            'error': "No marksheets found in the pdfs directory"
        }

    try:
        # Get and sort PDF files by semester number
        pdf_files = []
        for f in os.listdir(pdf_dir):
            if f.lower().endswith('.pdf'):
                match = re.search(r'semester(\d+)', f.lower())
                if match:
                    try:
                        sem_num = int(match.group(1))
                        pdf_files.append((sem_num, f))
                    except (ValueError, AttributeError):
                        continue

        if not pdf_files:
            return {
                'performance_data': {},
                'cgpa_history': {},
                'latest_semester': None,
                'error': "No valid semester files found"
            }

        pdf_files.sort(key=lambda x: x[0])
        pdf_files = [f[1] for f in pdf_files]

        performance_data = {}
        cgpa_history = {}

        for file in pdf_files:
            try:
                sem_num = re.search(r'semester(\d+)', file.lower()).group(1)
                semester = f"Semester {sem_num}"
                pdf_path = os.path.join(pdf_dir, file)
                extracted_data = extract_marks_from_pdf(pdf_path)
                
                if extracted_data['marks']:
                    performance_data[semester] = {
                        'marks': extracted_data['marks'],
                        'credits': extracted_data['credits'],
                        'grades': extracted_data['grades'],
                        'grade_points': extracted_data['grade_points'],
                        'sgpa': extracted_data['sgpa']
                    }
                    if extracted_data['cgpa'] is not None:
                        cgpa_history[semester] = extracted_data['cgpa']
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        latest_semester = None
        if pdf_files:
            try:
                latest_semester = extract_marks_from_pdf(os.path.join(pdf_dir, pdf_files[-1]))
            except Exception as e:
                print(f"Error processing latest semester: {str(e)}")

        return {
            'performance_data': performance_data,
            'cgpa_history': cgpa_history,
            'latest_semester': latest_semester,
            'error': None
        }

    except Exception as e:
        print(f"Error generating performance data: {str(e)}")
        return {
            'performance_data': {},
            'cgpa_history': {},
            'latest_semester': None,
            'error': "Error processing marksheets"
        }

def analyze_performance(performance_data, cgpa_history):
    """Generate dynamic insights and recommendations with comprehensive error handling"""
    analysis = {
        'strengths': [],
        'weaknesses': [],
        'recommendations': [],
        'prediction': None,
        'subject_trends': {},
        'credit_summary': {'earned': 0, 'total': 0},
        'error': None
    }

    if not performance_data or not isinstance(performance_data, dict):
        analysis['error'] = "No performance data available"
        return analysis

    try:
        # Calculate credit totals from the latest semester
        latest_sem = next(reversed(performance_data.values())) if performance_data else None
        if latest_sem and 'credits' in latest_sem and isinstance(latest_sem['credits'], dict):
            earned = sum(latest_sem['credits'].values())
            analysis['credit_summary'] = {
                'earned': earned,
                'total': earned * 2  # Assuming max credits are double of earned
            }
        
        # Track subject performance across semesters
        subject_history = defaultdict(list)
        for sem, data in performance_data.items():
            if not isinstance(data, dict) or 'marks' not in data:
                continue
                
            for subj, marks in data['marks'].items():
                subject_history[subj].append({
                    'semester': sem,
                    'marks': marks,
                    'grade': data['grades'].get(subj, 'N/A'),
                    'credits': data['credits'].get(subj, 0)
                })

        # Analyze each subject
        for subj, records in subject_history.items():
            if not records:
                continue
                
            latest = records[-1]
            trend = 'stable'
            
            if len(records) > 1:
                try:
                    trend = 'improving' if latest['marks'] > records[0]['marks'] else \
                           'declining' if latest['marks'] < records[0]['marks'] else 'stable'
                except (KeyError, TypeError):
                    trend = 'stable'
            
            analysis['subject_trends'][subj] = {
                'current': latest.get('marks', 0),
                'trend': trend,
                'grade': latest.get('grade', 'N/A'),
                'credits': latest.get('credits', 0)
            }

            # Classify as strength or weakness
            try:
                if latest['marks'] >= 75:
                    analysis['strengths'].append({
                        'subject': subj,
                        'score': latest['marks'],
                        'trend': trend,
                        'credits': latest.get('credits', 0)
                    })
                elif latest['marks'] < 50:
                    analysis['weaknesses'].append({
                        'subject': subj,
                        'score': latest['marks'],
                        'trend': trend,
                        'credits': latest.get('credits', 0)
                    })
            except (KeyError, TypeError):
                continue

        # Generate recommendations
        if analysis['strengths']:
            try:
                top_strength = max(analysis['strengths'], key=lambda x: x['score'])
                analysis['recommendations'].append(
                    f"Excellent performance in {top_strength['subject']} ({top_strength['score']}%). "
                    "Consider advanced courses or research opportunities in this area."
                )
            except (KeyError, ValueError):
                pass

        if analysis['weaknesses']:
            for weak in analysis['weaknesses']:
                try:
                    analysis['recommendations'].append(
                        f"Needs improvement in {weak['subject']} ({weak['score']}%). "
                        "Schedule faculty consultations and additional practice."
                    )
                except (KeyError, ValueError):
                    continue

        # CGPA prediction
        if len(cgpa_history) >= 2 and isinstance(cgpa_history, dict):
            try:
                semesters = sorted(cgpa_history.keys())
                X = np.array([int(s.split()[-1]) for s in semesters]).reshape(-1, 1)
                y = np.array([cgpa_history[sem] for sem in semesters])

                model = LinearRegression()
                model.fit(X, y)
                
                next_sem = len(cgpa_history) + 1
                predicted = max(0, min(10.0, model.predict([[next_sem]])[0]))
                
                analysis['prediction'] = {
                    'next_semester': next_sem,
                    'predicted_cgpa': round(predicted, 2),
                    'trend': 'improving' if model.coef_[0] > 0 else 'declining',
                    'confidence': round(model.score(X, y), 2)
                }

                # Add prediction-based recommendations
                if analysis['prediction']['trend'] == 'declining':
                    analysis['recommendations'].append(
                        "Your CGPA trend is declining. Meet with an academic advisor "
                        "to discuss improvement strategies."
                    )
                else:
                    analysis['recommendations'].append(
                        f"On track to achieve {analysis['prediction']['predicted_cgpa']} CGPA. "
                        "Maintain consistent study habits."
                    )
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                analysis['error'] = "Could not generate predictions"

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        analysis['error'] = "Error analyzing performance data"
    
    return analysis

def generate_cgpa_graph(cgpa_history, prediction):
    """Generate CGPA trend graph with comprehensive error handling"""
    plt.switch_backend('Agg')
    
    if not cgpa_history or len(cgpa_history) < 2 or not isinstance(cgpa_history, dict):
        return False

    try:
        semesters = sorted(cgpa_history.keys())
        sem_nums = []
        cgpa_values = []
        
        for sem in semesters:
            try:
                sem_nums.append(int(sem.split()[-1]))
                cgpa_values.append(cgpa_history[sem])
            except (ValueError, AttributeError, KeyError):
                continue

        if len(sem_nums) < 2:
            return False

        plt.figure(figsize=(10, 5))
        plt.plot(sem_nums, cgpa_values, 'o-', color='#4361ee', 
                linewidth=2.5, markersize=8, label='Actual CGPA')

        if prediction and isinstance(prediction, dict):
            try:
                x_pred = [sem_nums[-1], prediction['next_semester']]
                y_pred = [cgpa_values[-1], prediction['predicted_cgpa']]
                plt.plot(x_pred, y_pred, '--', color='#f72585', 
                        linewidth=2, label='Predicted Trend')
                plt.scatter([prediction['next_semester']], [prediction['predicted_cgpa']], 
                           color='#f72585', s=120, label=f"Predicted: {prediction['predicted_cgpa']}")
            except (KeyError, ValueError):
                pass

        plt.xlabel('Semester', fontsize=12)
        plt.ylabel('CGPA', fontsize=12)
        plt.title('CGPA Progression Trend', fontsize=14, pad=20)
        plt.xticks(range(1, max(sem_nums) + (2 if prediction else 1)))
        plt.ylim(0, 10.5)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()

        graph_path = os.path.join(app.config['STATIC_FOLDER'], 'cgpa_graph.png')
        plt.savefig(graph_path, dpi=100, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Graph generation error: {str(e)}")
        return False

@app.route('/performance')
def performance():
    """Main performance dashboard route with comprehensive error handling"""
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        data = generate_performance()
        if data.get('error'):
            return render_template("performance.html", 
                                error=data['error'],
                                performance_data={},
                                cgpa_history={},
                                latest_semester=None,
                                analysis=None)

        analysis = analyze_performance(data['performance_data'], data['cgpa_history'])
        
        # Ensure credit_summary exists with proper structure
        if 'credit_summary' not in analysis or not isinstance(analysis['credit_summary'], dict):
            analysis['credit_summary'] = {'earned': 0, 'total': 0}
        else:
            analysis['credit_summary']['earned'] = analysis['credit_summary'].get('earned', 0)
            analysis['credit_summary']['total'] = analysis['credit_summary'].get('total', 0)
        
        # Generate graph if we have enough data
        if data['cgpa_history'] and len(data['cgpa_history']) >= 2:
            generate_cgpa_graph(data['cgpa_history'], analysis.get('prediction'))

        return render_template("performance.html",
                            error=None,
                            performance_data=data['performance_data'],
                            cgpa_history=data['cgpa_history'],
                            latest_semester=data['latest_semester'],
                            analysis=analysis)

    except Exception as e:
        print(f"Performance route error: {str(e)}")
        return render_template("performance.html",
                            error="An error occurred while processing your request",
                            performance_data={},
                            cgpa_history={},
                            latest_semester=None,
                            analysis=None)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['user'] = username
            return redirect(url_for('welcome'))
        else:
            return render_template('login.html', error="Invalid credentials. Only numbers allowed.")
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('welcome.html')

@app.route('/student_info')
def student_info():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('student_info.html')

@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))



@app.route('/examination')
def examination():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('examination.html')

@app.route('/exam_schedule')
def exam_schedule():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('exam_schedule.html')

@app.route('/marks_obtained')
def marks_obtained():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('marks_obtained.html')

@app.route('/result')
def result():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    student_data = {
        'name': 'Vilma Xavier',
        'id': '225025',
        'academic_year': '2024-2025',
        'year_group': 'Third Year-A'
    }

    return render_template('result.html', student_data=student_data)

@app.route('/registration')
def registration():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('registration.html')

@app.route('/directory')
def directory():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('directory.html')

@app.route('/navigation')
def navigation():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('navigation.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)