# import rasterio
import os
# from rasterio.enums import Resampling
import numpy as np
from PIL import Image

alldatapath = [
    # r'ori/image/train/A1', r'ori/image/train/A2', r'ori/image/train/A3', r'ori/image/train/A4',
    # r'ori/image/test/A1', r'ori/image/test/A2', r'ori/image/test/A3', r'ori/image/test/A4',
    # r'ori/image/val/A1', r'ori/image/val/A2', r'ori/image/val/A3', r'ori/image/val/A4',
    # r'ori/mask/train/A1', r'ori/mask/train/A2', r'ori/mask/train/A3', r'ori/mask/train/A4',
    # r'ori/mask/test/A1', r'ori/mask/test/A2', r'ori/mask/test/A3', r'ori/mask/test/A4',
    # r'ori/mask/val/A1', r'ori/mask/val/A2', r'ori/mask/val/A3', r'ori/mask/val/A4',
    r'ori/RGB/train/A1', r'ori/RGB/train/A2', r'ori/RGB/train/A3', r'ori/RGB/train/A4',
    r'ori/RGB/test/A1', r'ori/RGB/test/A2', r'ori/RGB/test/A3', r'ori/RGB/test/A4',
    r'ori/RGB/val/A1', r'ori/RGB/val/A2', r'ori/RGB/val/A3', r'ori/RGB/val/A4',
    ]

allsavepath = [
    # r'crop_2048/image/train/A1', r'crop_2048/image/train/A2', r'crop_2048/image/train/A3', r'crop_2048/image/train/A4',
    # r'crop_2048/image/test/A1', r'crop_2048/image/test/A2', r'crop_2048/image/test/A3', r'crop_2048/image/test/A4',
    # r'crop_2048/image/val/A1', r'crop_2048/image/val/A2', r'crop_2048/image/val/A3', r'crop_2048/image/val/A4',
    # r'crop_2048/mask/train/A1', r'crop_2048/mask/train/A2', r'crop_2048/mask/train/A3', r'crop_2048/mask/train/A4',
    # r'crop_2048/mask/test/A1', r'crop_2048/mask/test/A2', r'crop_2048/mask/test/A3', r'crop_2048/mask/test/A4',
    # r'crop_2048/mask/val/A1', r'crop_2048/mask/val/A2', r'crop_2048/mask/val/A3', r'crop_2048/mask/val/A4',
    r'crop_2048/RGB/train/A1', r'crop_2048/RGB/train/A2', r'crop_2048/RGB/train/A3', r'crop_2048/RGB/train/A4',
    r'crop_2048/RGB/test/A1', r'crop_2048/RGB/test/A2', r'crop_2048/RGB/test/A3', r'crop_2048/RGB/test/A4',
    r'crop_2048/RGB/val/A1', r'crop_2048/RGB/val/A2', r'crop_2048/RGB/val/A3', r'crop_2048/RGB/val/A4',    
    ]

HH = 12500
WW = 12500

sizeImgH = 2048
sizeImgW = 2048

nnr = np.uint8(np.floor(HH/sizeImgH))
nnc = np.uint8(np.floor(WW/sizeImgW))

overlapH = np.int16(np.floor(sizeImgH - (HH-sizeImgH)/nnr))
overlapW = np.int16(np.floor(sizeImgW - (WW-sizeImgW)/nnc))

disH = sizeImgH - overlapH
disW = sizeImgW - overlapW

r = np.uint8(np.ceil(HH/disH))
c = np.uint8(np.ceil(WW/disW))

if (c-1)*disW+overlapW >= WW:
    c = c - 1   
if (r-1)*disH+overlapH >= HH:
    r = r - 1

# r = np.uint8(np.ceil(HH/disH)) - 1
# c = np.uint8(np.ceil(WW/disW)) - 1

for nn in range(len(alldatapath)):
    datapath = alldatapath[nn]
    savepath = allsavepath[nn]
    datadir = os.listdir(datapath)
    print(savepath)
    for ii in range(len(datadir)):
        path = os.path.join(datapath,datadir[ii])
        if 'image' not in savepath:
            img = np.expand_dims(np.array(Image.open(path)), 0)
        else:
            img = np.transpose(np.array(Image.open(path)), (2,0,1))
        [C,H,W] = img.shape

        for i in range(r):
            if i%10 == 0:
                print([ii, i*100/r])
            for j in range(c):
                temp = np.zeros([C,sizeImgH,sizeImgW],dtype=np.uint8)
                for cc in range(C):
                    if i==r-1:
                        if j==c-1:
                            temp[cc,:,:] = img[cc, H-sizeImgH:H, W-sizeImgW:W] 
                        else:
                            temp[cc,:,:] = img[cc, H-sizeImgH:H, j*disW:(j+1)*disW+overlapW] 
                    else:
                        if j==c-1:
                            temp[cc,:,:] = img[cc, i*disH:(i+1)*disH+overlapH, W-sizeImgW:W] 
                        else:
                            temp[cc,:,:] = img[cc, i*disH:(i+1)*disH+overlapH, j*disW:(j+1)*disW+overlapW]

                if 'image' not in savepath:
                    newpath = os.path.join(savepath,datadir[ii][:-4]+'_'+str(i+1)+'_'+str(j+1)+'.png')
                    temp = Image.fromarray(temp[0])
                else:
                    newpath = os.path.join(savepath,datadir[ii][:-4]+'_'+str(i+1)+'_'+str(j+1)+'.jpg')
                    temp = Image.fromarray(np.transpose(temp, (1,2,0)))

                os.makedirs(os.path.dirname(newpath), exist_ok=True)
                temp.save(newpath)

                
