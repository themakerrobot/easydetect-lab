@echo off
rem easydetect lab - double-click to start it on Windows.
rem The first run makes a .venv folder here and installs what the lab needs
rem (a few minutes); later runs start the lab and open the browser.
rem To use a Python you already have (a conda env, say) instead of .venv:
rem   set LAB_PYTHON=C:\path\to\python.exe   before running this file.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if defined LAB_PYTHON (
  set "PYEXE=%LAB_PYTHON%"
  goto run
)

set "VENV=%~dp0.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" goto install
rem requirements.txt changed since the last install (a git pull): install again
fc /b requirements.txt "%VENV%\requirements.installed" >nul 2>nul || goto deps
goto run

:install
echo [lab] 처음 실행이라 필요한 것을 설치합니다. 몇 분 걸려요...
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY goto nopython
%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))" || goto oldpython
%PY% -m venv "%VENV%" || goto fail

:deps
"%PYEXE%" -m pip install --upgrade pip || goto fail
"%PYEXE%" -m pip install -r requirements.txt || goto fail
copy /y requirements.txt "%VENV%\requirements.installed" >nul

:run
"%PYEXE%" run.py %*
if errorlevel 1 goto fail
exit /b 0

:nopython
echo [lab] Python을 찾지 못했어요. https://www.python.org/downloads/ 에서 3.10 이상을 설치하세요.
echo [lab] 설치 화면에서 "Add python.exe to PATH"를 꼭 체크하세요.
pause
exit /b 1

:oldpython
echo [lab] Python 3.10 이상이 필요해요. https://www.python.org/downloads/
pause
exit /b 1

:fail
echo [lab] 문제가 생겼어요. 위의 메시지를 확인하세요.
pause
exit /b 1
