@ECHO OFF

:: check if conda is installed
ECHO Checking if Anaconda is installed

conda --version
if errorlevel 1 goto errorNoConda
ECHO Anaconda is installed...

:: virtual environment
ECHO Creating virtual environment 'adverb'...
call conda env create -f environment.yml
ECHO Virtual environment 'adverb' created...

:: start server
ECHO Starting API-Server...
run.bat

goto :eof

errorNoConda:
ECHO.
ECHO Error^: Anaconda not installed
ECHO Install Anaconda and add it to the PATH environment variable

PAUSE