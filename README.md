# Learning to Generate Novel Domains for Domain Generalization

This is an unofficial PyTorch implementation of Learning to Generate Novel Domains for Domain Generalization.

[[arxiv]](https://arxiv.org/abs/2007.03304)

Some of the code is borrowed from https://github.com/HAHA-DL/Episodic-DG and https://github.com/yunjey/stargan.

#### data

Please download the data from https://drive.google.com/open?id=0B6x7gtvErXgfUU1WcGY5SzdwZVk and use the official train/val split.

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

