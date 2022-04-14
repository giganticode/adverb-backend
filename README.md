# ADVERB (vscode-extension - backend)

> *This is the backend API service for the vscode-extension ADVERB: see https://github.com/giganticode/adverb*

## Quick start

### Prerequirements
Anaconda required.
Download and install Anaconda from the official website https://docs.anaconda.com/anaconda/install/index.html

### Installation

Clone the project
```
git clone https://github.com/giganticode/adverb-backend.git
```

#### Windows
Run the 'install_and_run.bat'-Script - it will create a virtual conda environment called 'adverb', install all dependencies and start the webservice.
```
cd adverb-backend
.\install_and_run.bat
```

#### Unix
Run the 'install_and_run.sh'-Script - it will create a virtual conda environment called 'adverb', install all dependencies and start the webservice.
```
cd adverb-backend
sh install_and_run.sh
```

### Manual installation

#### Create conda environment 'adverb'
```
conda env create -f environment.yml
```

#### Activate the conda environment
```
conda activate adverb
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