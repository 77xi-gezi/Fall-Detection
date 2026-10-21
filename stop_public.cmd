@echo off
title Fall Detection - Stop External Access
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\public_access.py" stop
pause
