# ADVERB (vscode-extension - backend)

> *This is the backend API service for the vscode-extension ADVERB: see https://github.com/giganticode/adverb*

## Quick start

### Prerequirements
Python version >= 3.6 required.
Download and install Python from the official website https://www.python.org/downloads/

Rust required.
Download and install Rust from the official website https://www.rust-lang.org/tools/install


### Installation

Clone the project
```
git clone https://github.com/giganticode/adverb-backend.git
```

#### Windows
Run the 'install_and_run.bat'-Script - it will install all dependencies and start the webservice.
```
cd adverb-backend
.\install_and_run.bat
```

#### Unix
Run the 'install_and_run.sh'-Script - it will install all dependencies and start the webservice.
```
cd adverb-backend
sh install_and_run.sh
```

### Manual installation
#### Windows only
Install PyTorch
```
pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

#### Install python dependencies
```
pip install -r requirements.txt
```

### Start the webservice
```
python webservice.py 
```

Optional parameters:

--port=8080 => Set the port of the webservice (default 8080)

--host=127.0.0.1 => Set the host of the webservice (default 127.0.0.1 )

--debug => Enable the debug mode (default false)


#### Start the webservice with the provided script
run the run.bat-script (Windows) or the run.sh-script (UNIX)

Running this script will start the webservice.

Note: on Windows you have to install PyTorch (See section 'Windows only')