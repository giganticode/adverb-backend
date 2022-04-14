# ADVERB (vscode-extension - backend)

> *This is the backend API service for the vscode-extension ADVERB: see https://github.com/giganticode/adverb*

## Quick start

### Prerequirements
Anaconda required.
Download and install Anaconda from the official website https://docs.anaconda.com/anaconda/install/index.html

### Installation

#### Clone the project
```
git clone https://github.com/giganticode/adverb-backend.git
```

#### Create conda environment 'adverb'
For GPU-use:
```
conda env create -f conda_env.yml
```
For CPU-use:
```
conda env create -f conda_env_cpu.yml
```

#### Activate the conda environment
For GPU-use:
```
conda activate adverb
```
For CPU-use:
```
conda activate adverb-cpu
```

### Manually start the webservice
```
python webservice.py 
```

Optional parameters:

--port=8090 => Set the port of the webservice (default 8090)

--host=127.0.0.1 => Set the host of the webservice (default 0.0.0.0)

--debug => Enable the debug mode (default false)


### Start the webservice with the provided script
Running the following script will start the webservice.
Windows
```
run.bat
``` 
Unix
``` 
run.sh
``` 

## CPU execution

Install the conda environment from the `conda_env_cpu.yml` file. In addition, note that if you are testing CPU execution on a machine that includes GPUs you might need to specify CUDA_VISIBLE_DEVICES="" as part of your command.
