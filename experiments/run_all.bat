@echo off
REM Overnight run: tune -> scaling + 3/5/10-fold CV -> report, for SinoCT then ICH.
REM Safe to rerun after a crash or reboot: every step resumes where it stopped.
REM Usage:  experiments\run_all.bat      (override Python with: set PY=C:\path\to\python.exe)

setlocal EnableExtensions
pushd "%~dp0.." || (echo Could not enter the repo folder & exit /b 1)
set "ROOT=%CD%"
set "LOGS=%ROOT%\logs"
if "%PY%"=="" set "PY=C:\Users\bahra\anaconda3\envs\researchenv\python.exe"
if not exist "%LOGS%\" mkdir "%LOGS%"

echo [%time%] checking environment...
"%PY%" -c "import numpy, scipy, scipy.stats, sklearn.model_selection, torch, xgboost, optuna; assert torch.cuda.is_available(), 'CUDA not available'; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| torch', torch.__version__, torch.cuda.get_device_name(0))"
if errorlevel 1 (
  echo.
  echo Environment check failed. Fix with:
  echo   "%PY%" -m pip install "numpy<2.5" scipy scikit-learn xgboost matplotlib optuna pandas pyarrow
  echo   "%PY%" -m pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  popd & exit /b 1
)

for %%D in (sino ich) do (
  call :step %%D tune "experiments\tune.py --dataset %%D --device cuda" || (popd & exit /b 1)
  call :step %%D scaling "experiments\scaling_cv.py --dataset %%D --device cuda" || (popd & exit /b 1)
  call :step %%D report "experiments\make_report.py --dataset %%D" || (popd & exit /b 1)
)
echo [%time%] ALL DONE. Zip the results folder and send it over.
popd
exit /b 0

:step
set "LOG=%LOGS%\%2_%1.log"
echo [%time%] %1: %2   (log: %LOG%)
"%PY%" -u %~3 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo.
  echo %1 %2 FAILED. Last lines of %LOG%:
  powershell -NoProfile -Command "Get-Content -Tail 25 '%LOG%'"
  exit /b 1
)
exit /b 0
