@echo off
rem easydetect lab, offline bundle - double-click to start it on Windows.
rem Nothing here needs the internet: Python, PyTorch (CPU), the lab and the
rem model weights it starts from are all inside this folder, and the lab is
rem told never to reach for the network. Unzip the whole zip first; started
rem from inside the zip, Windows copies only this file out and nothing works.
rem Pictures, labels and trained models are kept in the data folder beside it.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PYEXE=%ROOT%\python\python.exe"
if not exist "%PYEXE%" goto notunzipped
if not exist "%ROOT%\lab\run.py" goto notunzipped

rem where easydetect keeps the weights (filled when the bundle was built) and
rem where the lab keeps its datasets and runs: both inside this folder
set "EASYDETECT_HOME=%ROOT%\easydetect-cache"
if not defined EASYDETECT_LAB_HOME set "EASYDETECT_LAB_HOME=%ROOT%\data"
rem never the network: the weights mirror at an address nothing answers on,
rem and Hugging Face's own libraries (if anything loads them) offline
set "EASYDETECT_ASSETS_URL=http://127.0.0.1:9"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PIP_NO_INDEX=1"
rem this folder's Python only: not a Python installed on the PC, nor its packages
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
rem UTF-8 everywhere: folder names in Hangul (OneDrive\바탕 화면) print and open
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem OneDrive syncs every file as it is written: the database and each epoch's
rem weights, which slows training and can lock the database mid-run
echo "%ROOT%" | find /i "OneDrive" >nul && (
  echo [lab] 주의: 이 폴더는 OneDrive 안에 있어요. 동기화가 학습을 느리게 하고 파일을 잠글 수 있어요.
  echo [lab]       C:\easydetect-lab 처럼 OneDrive 밖으로 옮겨서 실행하는 걸 권해요.
)
echo [lab] 오프라인 묶음으로 시작해요. 인터넷이 없어도 돼요.
echo [lab] 처음 켤 때는 1~2분 걸릴 수 있어요. 이 창을 닫으면 lab이 꺼져요.
"%PYEXE%" "%ROOT%\lab\run.py" %*
if errorlevel 1 goto fail
exit /b 0

:notunzipped
echo [lab] python 폴더나 lab 폴더가 없어요.
echo [lab] zip 파일 안에서 바로 실행하지 말고, 먼저 "압축 풀기"로 전부 푼 다음
echo [lab] 풀린 폴더의 start-lab-offline.bat을 더블클릭하세요.
pause
exit /b 1

:fail
echo [lab] 문제가 생겼어요. 위의 메시지를 확인하세요.
echo [lab] 8080 포트를 다른 프로그램이 쓰고 있으면: start-lab-offline.bat --port 8090
pause
exit /b 1
