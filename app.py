from flask import Flask
from login import create_login_blueprint, init_app as init_login
from resume_calibrator import create_resume_calibrator_blueprint

app = Flask(__name__)

# Initialize login session
init_login(app)

# Register blueprints
app.register_blueprint(create_login_blueprint(), url_prefix='/login')
app.register_blueprint(create_resume_calibrator_blueprint(), url_prefix='/resume')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
