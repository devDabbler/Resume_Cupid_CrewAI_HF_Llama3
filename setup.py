from setuptools import setup, find_packages

setup(
    name='resume_cupid',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-session',
    ],
    entry_points={
        'console_scripts': [
            'resume_cupid = app:main',  # Define entry point for your app, change as needed
        ],
    },
)
