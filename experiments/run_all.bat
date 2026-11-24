@echo off
REM Runs experiments\run_all.py with the research env's Python. All logic lives in run_all.py.
REM Usage:  experiments\run_all.bat            (extra args are passed through, e.g. --datasets ich)
if "%PY%"=="" set "PY=C:\Users\bahra\anaconda3\envs\researchenv\python.exe"
"%PY%" -u "%~dp0run_all.py" %*
