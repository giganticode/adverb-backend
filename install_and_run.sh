#!/bin/bash
# check if python is installed
echo Checking if python is installed...

if command -v python3 &>/dev/null; then
    echo Python is installed...
else
    echo Python is not installed...
    echo Install Python and add it to the PATH environment variable
    xdg-open https://www.python.org/downloads/
    exit
fi

# setup virtual environment
echo Creating virtual environment...
python3 -m venv adverb-venv
echo Virtual environment created...

echo Activating virtual environment...
source adverb-venv/bin/activate
echo Virtual environment activated...

# install Rust for huggingface/transformers
echo Installing Rust...
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo Dependencies installed...

# install pytorch
echo Installing PyTorch...
pip3 install torch torchvision torchaudio

# start server
echo Starting API-Server at http://localhost:8080
./run.sh