@echo off
echo ============================================
echo   InterviewIQ — Setup and Run
echo ============================================
echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt
echo.
echo [2/3] Training ML model from dataset...
python ml/train_model.py
echo.
echo [3/3] Starting app...
echo Open: http://127.0.0.1:5000
echo.
python app.py
pause
