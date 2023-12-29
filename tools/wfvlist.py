import os, random, shutil

# srcDir ='D:/build/whu/sat/images'
srcDir = r'ori/Image__8bit_NirRGB/train'
pathDir = os.listdir(srcDir)
# np = []
# for name in pathDir:
#     if 'chicago' in name:
#         np.append(name)
# pathDir = np
filenumber = len(pathDir)

# sample1 = random.sample(pathDir, int(filenumber * 0.2))
sample1 = random.sample(pathDir, int(30))
target1 = open("myval.txt", 'w')
for name in sample1:
    target1.write("%s\n" % (name))
#
# # rate1 = 0.2
# # rate2 = 0.5
# picknumber1 = int(24)
# # if picknumber1 % 2 != 0:
# #     picknumber1 = picknumber1 + 1
# picknumber2 = int(12)
# #
# sample1 = random.sample(pathDir, picknumber1)
# sample2 = random.sample(sample1, picknumber2)
# print(filenumber,picknumber1,picknumber2)
# print(len(sample1),len(sample2))

# target1 = open("wfv_val.txt", 'w')
# for name in sample2:
#     target1.write("%s\n" % (name))

# target2 = open("wfv_test.txt", 'w')
# for name in sample1:
#     if name in sample2:
#         continue
#     target2.write("%s\n" % (name))

# target = open("wfv_train.txt", 'w')
# for name in pathDir:
#     if name in sample1:
#         continue
#     target.write("%s\n" % (name))
# #
# # srcDir = 'D:/build/inria/inria/images/'
# # pathDir = os.listdir(srcDir)
# # np = []
# # for name in pathDir:
# #     if 'austin' in name:
# #         np.append(name)
# # pathDir = np
# #
# # target2 = open("au_train.txt", 'w')
# # for name in pathDir:
# #     target2.write("%s\n" % (name))
# #
# # srcDir = r'D:\build\whu\satnewaug\images'
# # pathDir = os.listdir(srcDir)
# # target = open("arlsatnewaug.txt", 'w')
# # for name in pathDir:
# #     target.write("%s\n" % (name))
