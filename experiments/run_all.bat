@echo off
REM Overnight run: tune -> scaling + 3/5/10-fold CV -> report, for SinoCT then ICH.
REM Safe to rerun after a crash or reboot: every step resumes where it stopped.
REM Usage (from anywhere):  experiments\run_all.bat
REM Override the Python path with:  set PY=C:\path\to\python.exe  before running.

setlocal
cd /d "%~dp0\.."
if "%PY%"=="" set PY=C:\Users\bahra\anaconda3\envs\researchenv\python.exe
if not exist logs mkdir logs

echo [%date% %time%] checking environment...
"%PY%" -c "import torch, sklearn, xgboost, optuna; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" || (
  echo Environment check failed. Install with:
  echo   "%PY%" -m pip install numpy pandas scikit-learn xgboost matplotlib optuna
  echo   "%PY%" -m pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  exit /b 1
)

for %%D in (sino ich) do (
  echo [%date% %time%] %%D: tuning   ^(log: logs\tune_%%D.log^)
  "%PY%" -u experiments\tune.py --dataset %%D --device cuda >> logs\tune_%%D.log 2>&1 || (echo %%D tuning FAILED, see logs\tune_%%D.log & exit /b 1)
  echo [%date% %time%] %%D: scaling + CV   ^(log: logs\scaling_%%D.log^)
  "%PY%" -u experiments\scaling_cv.py --dataset %%D --device cuda >> logs\scaling_%%D.log 2>&1 || (echo %%D scaling FAILED, see logs\scaling_%%D.log & exit /b 1)
  echo [%date% %time%] %%D: report
  "%PY%" -u experiments\make_report.py --dataset %%D >> logs\report_%%D.log 2>&1 || (echo %%D report FAILED, see logs\report_%%D.log & exit /b 1)
)
echo [%date% %time%] ALL DONE. Zip the results folder and send it over.
