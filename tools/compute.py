import numpy as np
import cv2
import argparse
import os

def recursive_glob(rootdir='.', suffix=''): # rootdir 是根目录的路径，suffix 是要搜索的文件后缀。
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)    # 这是一个列表推导式，它将匹配后缀的文件的完整路径组成的列表返回。列表推导式会遍历文件系统中的文件，然后检查文件名是否以指定的后缀结尾。
            for looproot, _, filenames in os.walk(rootdir)# 这是一个嵌套的循环，使用 os.walk 函数来遍历指定根目录 rootdir 及其子目录中的文件。os.walk 返回一个生成器，生成三元组 (当前目录, 子目录列表, 文件列表)。在这里，我们只关心当前目录和文件列表，因此使用 _ 来表示不关心的子目录列表。
            for filename in filenames if filename.endswith(suffix)]# 这是列表推导式的内部循环，遍历当前目录中的文件列表 filenames，然后检查每个文件名是否以指定的后缀 suffix 结尾。如果是，就将满足条件的文件名添加到最终的返回列表中。

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, default='/media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index')
    parser.add_argument("--suffix", type=str, default='.png')
    args = parser.parse_args()
   
    return args

def main():
    args = parse_args()

    namelist = recursive_glob(rootdir=args.label_path, suffix=args.suffix)
    # Load the images into a numpy array
    images = np.zeros((len(namelist), 6800, 7200), dtype=np.uint8)
    for i in range(len(namelist)):
        images[i] = cv2.imread(namelist[i], cv2.IMREAD_GRAYSCALE)

    # Flatten the array into a 1D array
    pixels = images.flatten()

    # Calculate the histogram
    hist, _ = np.histogram(pixels, bins=25, range=(0, 24))

    # Print the histogram
    print(hist)

    # 446988756
    hist = hist[1:]
    port = hist/np.sum(hist)*100
    np.set_printoptions(suppress=True, precision=2)
    print(hist)
    print(port)
    # a = [65531821, 24623590, 319572918,  47277894,  7861839, 72261747,
    #     3385076,  1335436, 14553348,  5493920, 86814843, 98021731, 131939228,
    #     13623062, 19222062,   120456,  8978163, 42302971,   483502,   337374,
    #     51838337,  3719600,  1247560,  1264766]
    
    # b = a/np.sum(a)*100
    # c = np.round(b, decimals=2)
    # np.set_printoptions(suppress=True, precision=2)
    # print(c)
#     [ 6.41  2.41 31.28  4.63  0.77  7.07  0.33  0.13  1.42  0.54  8.5   9.59
#       12.91  1.33  1.88  0.01  0.88  4.14  0.05  0.03  5.07  0.36  0.12  0.12]

# train
# [  92666848   88516412 1250244934  168245416   32640540  222613962
#    93384347    1007732   53538814    6833032  104040876  150997187
#   298702565   29723302   26001357    1040751  133215447  132651224
#      412816     553426   99008246    6190519    2381993    2700222]
# [ 3.09  2.95 41.71  5.61  1.09  7.43  3.12  0.03  1.79  0.23  3.47  5.04
#   9.97  0.99  0.87  0.03  4.44  4.43  0.01  0.02  3.3   0.21  0.08  0.09]

# val
# [ 21799817   7678022 306859506 119541107   5223728 110625242  94631760
#     114265  14964309   5967261  64887519  32983992  66536744   8566092
#   11381205    562019  67543465  46104001    118297     96750  29080709
#    1755277    412111    803724]
# [ 2.14  0.75 30.14 11.74  0.51 10.86  9.29  0.01  1.47  0.59  6.37  3.24
#   6.53  0.84  1.12  0.06  6.63  4.53  0.01  0.01  2.86  0.17  0.04  0.08]

# test
# [65531821, 24623590, 319572918,  47277894,  7861839, 72261747,  3385076,  
# 1335436, 14553348,  5493920, 86814843, 98021731, 131939228,  13623062, 
# 19222062,   120456,  8978163, 42302971,   483502,   337374, 51838337, 
# 3719600,  1247560,  1264766]
# [ 6.41  2.41 31.28  4.63  0.77  7.07  0.33  0.13  1.42  0.54  8.5   9.59
#   12.91  1.33  1.88  0.01  0.88  4.14  0.05  0.03  5.07  0.36  0.12  0.12]

# [ 179998486  120818024 1876677358  335064417   45726107  405500951
#   191401183    2457433   83056471   18294213  255743238  282002910
#   497178537   51912456   56604624    1723226  209737075  221058196
#     1014615     987550  179927292   11665396    4041664    4768712]
# [ 3.57  2.4  37.26  6.65  0.91  8.05  3.8   0.05  1.65  0.36  5.08  5.6
#   9.87  1.03  1.12  0.03  4.16  4.39  0.02  0.02  3.57  0.23  0.08  0.09]

if __name__ == '__main__':
    main()