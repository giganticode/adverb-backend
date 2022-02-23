@ECHO OFF

:: check if Python is installed
ECHO Checking if python is installed

python --version
if errorlevel 1 goto errorNoPython
ECHO Python is installed...

:: virtual environment
ECHO Creating virtual environment...
python -m venv adverb-venv
ECHO Virtual environment created...
ECHO Activating virtual environment...
adverb-venv\Scripts\activate.bat
ECHO Virtual environment activated...

ECHO ************************************************************************************************************
ECHO **** PLEASE INSTALL RUST BEFORE CONTINUING AND RESTART THIS PROCEDURE (CLOSE ALSO THIS COMMAND PROMPT). ****
ECHO **** DOWNLOAD AND INSTALL FROM: https://www.rust-lang.org/tools/install                                 ****
ECHO **** SKIP THIS STEP AND CONTINUE, IF ALREADY INSTALLED                                                  ****
ECHO ************************************************************************************************************
start https://www.rust-lang.org/tools/install
pause

:: install dependencies
ECHO Installing Dependencies...
@pip install -r requirements.txt
ECHO Dependencies installed...

:: install pytorch
ECHO Installing PyTorch...
pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

:: start server
ECHO Starting API-Server...
run.bat

goto :eof

errorNoPython:
ECHO.
ECHO Error^: Python not installed
ECHO Install Python and add it to the PATH environment variable
start https://www.python.org/downloads/

PAUSE