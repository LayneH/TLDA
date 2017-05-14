# TLDA
TensorFlow implementation of the paper [Supervised Representation Learning: Transfer Learning with Deep Autoencoders][TLDA].

---
## Dependency

* [Numpy][np]
* [Tensorflow][tf] >= 1.0
* [CIFAR-100 Dataset][cifar]

---
## Setup
Run `bash setup.sh` in terminal to automatically download the CIFAR-100 dataset.

---

## Usage
Enter `python TLDA.py` in bash for fast with default setting.

Use `--cifar_path` to specify the path of pretrained VGG Net. By default, the model is located under `datasets/cifar-100-python` directory

Use `--outfile` to specify file to save the result. By default, the file is "result.csv".

Use `--help` to acquire more information.

---

[TLDA]:http://www.intsci.ac.cn/users/zhuangfuzhen/paper/IJCAI15-578.pdf
[np]:https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt
[tf]:http://tensorflow.org
[cifar]:https://www.cs.toronto.edu/~kriz/cifar.html
