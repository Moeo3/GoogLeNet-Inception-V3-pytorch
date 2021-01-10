## About

This is my implementation of **GoogLeNet Inception V3** by **Pytorch**. 

The implementation can be adapted to **any size of picture**. 

I just removed the dropout layer and changed it to a reeling down-dimensional layer. Others is almost same as the original network.

I think my implementation is **clearer** than the one in [torchvision](https://pytorch.org/docs/stable/torchvision/index.html).

I've added **some comments** to my code that can help you better understand the network, as well as my code.

If my code can **help you**, then I'm really **honored**.



## Dependency

I didn't test it in other environments, but in my following environment the network can work.

Python 3.6.9

​	-- Pytorch 1.3.1

Cuda 10.1.105

​	-- CuDNN 7.6.5

Ubuntu 16.04.7 LTS



## Usage

Place the code in your project,  and just import to it.

```python
from googlenet_v3 import GoogLeNetV3
net = GoogLeNetV3(channels_in)
```



## License

APACHE LICENSE, VERSION 2.0
