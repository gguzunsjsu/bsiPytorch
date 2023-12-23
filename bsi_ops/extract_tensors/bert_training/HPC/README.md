### Create a python virtual environment

python3 -m venv venv

### Activate virtual environment to install requirements

source venv/bin/activate 

### Install Requirements 

##### TORCH with CUDA 

`pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

##### requirements.txt file

pip3 install -r requirements.txt

### Install the dataset
wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt

### Create folder to store outputs
mkdir EsperBERTo

### Run Bash script
sbatch train_bash.sh

### To check status in queue
squeue

squeue -u <user_id>


