@echo off
title Fall Detection - Browser
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_web.ps1"
if errorlevel 1 pause
