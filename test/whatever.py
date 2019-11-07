#install gpu version of tensorflow
import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import torch.cuda
if torch.cuda.is_available():
    print('PyTorch found cuda')
else:
    print('PyTorch could not find cuda')