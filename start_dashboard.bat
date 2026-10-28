@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
set "QF5210_LOCAL_MODE=1"
python -m streamlit run streamlit_app.py --server.headless false
pause
