#!/bin/bash
# check if Anaconda is installed
echo Checking if Anaconda is installed...

if command -v conda &>/dev/null; then
    echo Anaconda is installed...
else
    echo Anaconda is not installed...
    echo Install Anaconda and add it to the PATH environment variable
    exit
fi

# setup virtual environment
echo Creating virtual environment 'adverb'...
conda env create -f environment.yml
echo Virtual environment 'adverb' created...

# start server
echo Starting API-Server...
./run.sh