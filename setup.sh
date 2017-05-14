# Get CIFAR100
mkdir -p datasets
wget -P datasets https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz

mkdir summary
