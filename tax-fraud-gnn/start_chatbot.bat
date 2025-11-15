@echo off
REM Start script for Tax Fraud Detection with Chatbot

echo ========================================
echo   Tax Fraud Detection System
echo   Starting Main Application with Chatbot
echo ========================================
echo.

REM Set Groq API Key from environment variable (do not hardcode in production)
REM For development, you can set the environment variable before running this script
REM Example: set GROQ_API_KEY=your_actual_key_here
if "%GROQ_API_KEY%"=="" (
    echo WARNING: GROQ_API_KEY environment variable not set!
    echo The chatbot will use fallback mode without AI capabilities.
    echo.
)

cd /d "%~dp0"

echo Starting Flask application with AI chatbot enabled...
echo.
echo Application will be available at:
echo   http://localhost:5000          - Main Dashboard
echo   http://localhost:5000/chatbot  - AI Chatbot
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

pause