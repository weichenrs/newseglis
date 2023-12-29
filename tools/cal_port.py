import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
import os
from utils import recursive_glob

# def calculate_weigths_labels(dataset, dataloader, num_classes):
#     # Create an instance from the data loader
#     z = np.zeros((num_classes,))
#     # Initialize tqdm
#     tqdm_batch = tqdm(dataloader)
#     print('Calculating classes weights')
#     for sample in tqdm_batch:
#         y = sample['label']
#         y = y.detach().cpu().numpy()
#         mask = (y >= 0) & (y < num_classes)
#         labels = y[mask].astype(np.uint8)
#         count_l = np.bincount(labels, minlength=num_classes)
#         z += count_l
#     tqdm_batch.close()
#     total_frequency = np.sum(z)
#     class_weights = []
#     for frequency in z:
#         class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
#         class_weights.append(class_weight)
#     ret = np.array(class_weights)
#     classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
#     np.save(classes_weights_path, ret)

#     return ret

num_classes = 16
z = np.zeros((num_classes,))
# filepath = '/media/dell/DATA/db/LIXINWEI/francelab/newdfctest/crop512testregclass13/lab'
# pathDir = recursive_glob(filepath, '.tif')
pathDir = open('').read().splitlines()

num = 0
for filename in pathDir:
    y = rasterio.open(filename).read(1)
    z += np.bincount(y[(y >= 0) & (y < num_classes)], minlength=num_classes)
    num += 1
    print(num)
total = np.sum(z)
frequency = z/total
print(z)
print(frequency)
# z=np.array([ 6.72801729e+08, 3.94770972e+08, 1.07451567e+08, 2.33195435e+08, 
#        4.71611986e+08, 6.02310238e+08, 4.81489793e+08, 9.09793057e+08, 
#        1.22039144e+09, 1.52616291e+08, 2.92624233e+08, 2.05592338e+08])

# fre = np.array([0.11711799, 0.06871977, 0.01870464, 0.0405935 , 0.08209587,
#        0.10484718, 0.08381535, 0.15837226, 0.21243968, 0.02656669,
#        0.05093857, 0.03578849])



# z13 = np.array([3.96382500e+06, 6.12750500e+06, 1.58219790e+07, 2.07583900e+06,
#        5.00322200e+06, 3.58373280e+07, 2.82092690e+07, 3.46948800e+07,
#        0.00000000e+00, 0.00000000e+00, 1.26006640e+07, 1.62057300e+07,
#        3.71485700e+06, 2.92624233e+08, 8.52344610e+07, 0.00000000e+00])
# 2100
# z12= np.array([9.96125000e+06, 1.84364570e+07, 9.30618200e+06, 1.15783400e+06,
#        6.28258200e+06, 7.01465200e+06, 2.22733200e+06, 3.95327800e+06,
#        0.00000000e+00, 0.00000000e+00, 5.66912490e+07, 1.45253533e+08,
#        1.48901434e+08, 0.00000000e+00, 4.64204890e+07, 0.00000000e+00])
# 1800
# # z11 = np.
# 3000
# z6 = np.array([2.39847660e+07, 5.51056846e+08, 3.80799736e+08, 0.00000000e+00,
#        4.18921230e+07, 7.17490880e+08, 4.07166077e+09, 1.38662986e+09,
#        0.00000000e+00, 0.00000000e+00, 1.08849092e+09, 5.47123477e+08,
#        0.00000000e+00, 0.00000000e+00, 7.36200630e+07, 0.00000000e+00])
# 3400
# z4 = np.array([3.32689000e+06, 8.84964912e+08, 4.24891820e+08, 0.00000000e+00,
#        5.06711928e+08, 2.48236016e+08, 0.00000000e+00, 2.61477921e+08,
#        0.00000000e+00, 0.00000000e+00, 2.40870426e+08, 5.90235460e+07,
#        0.00000000e+00, 0.00000000e+00, 6.29775650e+07, 0.00000000e+00])
# 3000
# z3 = np.array([2.32171600e+06, 6.18172576e+08, 3.73964804e+08, 3.14821690e+08,
#        6.00084380e+07, 2.58578621e+08, 9.62850750e+07, 2.80136444e+08,
#        0.00000000e+00, 0.00000000e+00, 3.37988918e+08, 8.03358960e+07,
#        0.00000000e+00, 0.00000000e+00, 6.38216620e+07, 0.00000000e+00])
# 2000


# zz = np.array([4.53874588e+08, 3.33888768e+08, 2.91407378e+07, 6.54856280e+07,
#        2.30988783e+09, 5.65259937e+07, 1.20481242e+09, 1.33163303e+09,
#        1.73622753e+08, 2.31479879e+07, 1.61308827e+07, 8.31993354e+07])

# port = zz/zz.sum()
# port = np.array(
#     [0.07463385, 0.05490373, 0.00479182, 0.01076827, 0.37983143,
#        0.00929497, 0.19811595, 0.21896997, 0.02855003, 0.00380639,
#        0.00265252, 0.01368106])
# print(z)
# print(frequency)

# frequency = np.array(
# [9.17023162e+08, 8.76319629e+08, 5.96030680e+07, 1.16494130e+08, 
#  3.14948117e+09, 1.13478167e+08, 3.28942564e+09, 1.84791268e+09, 
#  5.86803909e+08, 9.21462390e+07, 8.34964670e+07, 2.00915314e+08 ])
# total_frequency = frequency.sum()

# class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
# print(class_weight)
# class_weight = class_weight / np.median(class_weight)
# print(class_weight)