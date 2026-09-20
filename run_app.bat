@echo off
title XM-CBM Clinical Decision Support System
cd /d "%~dp0"
echo ======================================================================
echo   Launching XM-CBM Clinical Decision Support System...
echo ======================================================================
echo.
python -m streamlit run app.py
pause
