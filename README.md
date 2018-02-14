*WIP, first version will be released by the end of Feburary 2018*

## Packages
* [pysc2](https://github.com/deepmind/pysc2) v1.1
* [sc2client-proto](https://github.com/Blizzard/s2client-proto) v3.18.0
* [pytorch](https://github.com/pytorch/pytorch) v0.3.0
* [StarCraft II (Linux)](https://github.com/Blizzard/s2client-proto#downloads) v.3.16.1
* [tensorboardX (Tensorboard for Pytorch)](https://github.com/lanpa/tensorboard-pytorch) v1.0

Detail installation [steps](#installation).

## <a id='installation'></a> Installation
#### pytorch
Follow instruction [here](http://pytorch.org) and chose OS, Package Manager, Python version and CUDA version accordingly.
- Linux
```bash
# check cuda version
nvcc --version
# use CUDA 9 as example
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
```
- OS X
```bash
# no GPU as example
pip install http://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl 
```
#### pysc2
```bash
pip install pysc2
```
#### tensorboardX
```bash
pip install git+https://github.com/lanpa/tensorboard-pytorch
# TensorFlow is required
pip install tensorflow
```
