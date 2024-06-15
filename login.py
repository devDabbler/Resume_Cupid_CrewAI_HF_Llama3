from flask import Blueprint, Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from dotenv import load_dotenv
import os
import threading

# Load environment variables from .env file
load_dotenv()

def create_login_blueprint():
    login_bp = Blueprint('login', __name__)

    # Configuring secret key and session type
    login_bp.config = {
        'SECRET_KEY': os.getenv('SECRET_KEY'),
        'SESSION_TYPE': 'filesystem'
    }

    # Environment variables for username and password
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')

    @login_bp.route('/')
    def home():
        return render_template('login.html')

    @login_bp.route('/login', methods=['POST'])
    def login():
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('login.streamlit_redirect'))
        else:
            return 'Invalid Credentials. Please try again.'

    @login_bp.route('/dashboard')
    def dashboard():
        if 'logged_in' in session and session['logged_in']:
            return 'Welcome to the dashboard!'
        else:
            return redirect(url_for('login.home'))

    @login_bp.route('/logout')
    def logout():
        session.pop('logged_in', None)
        return redirect(url_for('login.home'))

    @login_bp.route('/streamlit_redirect')
    def streamlit_redirect():
        if 'logged_in' in session and session['logged_in']:
            return redirect('http://127.0.0.1:8501')  # URL of the Streamlit app
        else:
            return redirect(url_for('login.home'))

    return login_bp

# Ensure session is initialized with app context
def init_app(app):
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)

if __name__ == '__main__':
    app = Flask(__name__)
    init_app(app)
    app.register_blueprint(create_login_blueprint(), url_prefix='/')
    if threading.current_thread() is threading.main_thread():
        app.run()
