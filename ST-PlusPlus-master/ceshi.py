import math
import os.path

import h5py
import numpy as np
from PIL import Image
from PIL import Image
from torchvision import transforms

import random
import h5py
from torch import nn, optim
import torch
from torch import nn, optim
from data.dataset1 import RandomGenerator
id_to_reliability = []
id_to_reliability.append(('0000001', '0.0165', '6000'))
id_to_reliability.append(('0000002', '0.0125', '6032'))
id_to_reliability.append(('0000003', '0.0011', '5004'))
id_to_reliability.append(('0000004', '0.0123', '5001'))
id_to_reliability.append(('0000005', '0.0024', '4908'))
id_to_reliability.append(('0000006', '0.0016', '6040'))
id_to_reliability.append(('0000007', '0.0145', '6020'))
id_to_reliability.append(('0000008', '0.0001', '5092'))
id_to_reliability.append(('0000009', '0.0025', '4089'))
id_to_reliability.append(('0000010', '0.0044', '5000'))
length = len(id_to_reliability)
id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)
print("111111111111111111")
print(id_to_reliability)
new_reliability = id_to_reliability[round(length/3):round((length * 2)/3)]
new_reliability.sort(key=lambda elem: elem[2], reverse=False)
print("2222222222222222222")
print(new_reliability)
id_to_reliability[round(length/3):round((length * 2)/3)] = new_reliability
print("33333333333333333333")
print(id_to_reliability)
del id_to_reliability[7:-1]
print(id_to_reliability)

