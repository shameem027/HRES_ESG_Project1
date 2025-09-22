@echo off
cd /d "%~dp0"

REM Always launch PowerShell and keep it open after execution
powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0\project.ps1"
