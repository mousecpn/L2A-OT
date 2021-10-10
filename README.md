# Learning to Generate Novel Domains for Domain Generalization

This is an unofficial PyTorch implementation of Learning to Generate Novel Domains for Domain Generalization (ECCV2020).

[[arxiv]](https://arxiv.org/abs/2007.03304)

Some of the code is borrowed from https://github.com/HAHA-DL/Episodic-DG and https://github.com/yunjey/stargan.

#### data

Please download the data from https://drive.google.com/drive/folders/1i23DCs4TJ8LQsmBiMxsxo6qZsbhiX0gw?usp=sharing and use the official train/val split.

```
Domain-Generalization-via-Image-Stylization
├── data
│   ├── Train val splits and h5py files pre-read
```

#### train

```
python main.py --train True
```

#### test

```
python main.py --train False
```

