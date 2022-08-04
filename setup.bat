@echo off
echo Creating a virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo Installing all necessary libraries...
pip install -r requirements.txt
pause