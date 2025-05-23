conda create -n adap python=3.10

conda activate adap

pip install -r ./requirement.txt 

pip3 install torch torchvision torchaudio # depends

mkdir ./results
mkdir ./external_lightning_logs

python setup.py develop

git clone https://github.com/bigscience-workshop/promptsource.git # check

cd promptsource/

pip install -e .

cd ../

# ---
pip install --upgrade pip
pip install -U -r requirements.txt

pip install -U optimum
pip install "fschat[model_worker,webui]"
pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
pip install pytorch-quantization==2.1.3 --extra-index-url https://pypi.ngc.nvidia.com

pip install lightning==2.2.5
pip install pytorch-lightning==2.2.5
git clone https://github.com/bigscience-workshop/promptsource.git
cd promptsource
pip install -e .