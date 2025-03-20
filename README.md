# Point Cloud VQ-VAE


## 1. Environment

```bash

conda env create -f environment.yml

conda activate UniMo

pip install git+https://github.com/openai/CLIP.git

cd pytorch3d_local
tar zxf 2.1.0.tar.gz
export CUB_HOME=$PWD\/cub-2.1.0
python setup.py install
pip install -e .

cd utils/emd
python setup.py install


git clone https://github.com/zshyang/ylib.git
cd ylib
pip install -e . 

```

### 1.1 Visualization Requirements

```bash
apt-get update
apt-get install xvfb
apt-get install mesa-utils
```

## 2. Train

```bash
xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name point --gpu_id 0 --vqvae_cfg default --max_epoch 60000 --eval_every_it 1000 --recons_loss emd
```