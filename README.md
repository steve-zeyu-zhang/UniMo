# Point Cloud VQ-VAE


## 1. Environment

```bash

conda env create -f environment.yml

cd pytorch3d
python setup.py install


cd utils/emd
python setup.py install

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