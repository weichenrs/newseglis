import os, shutil
import os.path as osp

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

split = 'train'
# trainlst = open('dfc2022_labeled_val.txt')
# traindata = trainlst.readlines()
# traindata = [data.replace('\n','') for data in traindata]
# traindata = open('dfc2022_val66.txt').read().splitlines()
# traindata = [data.split('/')[-1][:-4] for data in traindata]
# oldimgpath = 'all/train/arf'
# newimgpath = 'all/val/arf'
# # ch
# oldlabpath = 'all/train/slope'
# newlabpath = 'all/val/slope'

# i = 0
# traindir = recursive_glob(oldimgpath, '.tif')

# for name in traindir:
#     if name.split('/')[-1][:-4].split('_')[0] in traindata:
#         data = name.split('/')[-2:]
#         oldrpath = name
#         oldspath = name.replace('arf','slope')
#         newrpath = oldrpath.replace('arf/train','arf/val').replace(oldimgpath, newimgpath)
#         newspath = oldspath.replace('slope/train','slope/val').replace(oldlabpath, newlabpath)
#         os.makedirs(os.path.dirname(newrpath), exist_ok=True)
#         os.makedirs(os.path.dirname(newspath), exist_ok=True)
        
#         shutil.move(oldrpath,newrpath)
#         shutil.move(oldspath,newspath)
#         # shutil.move(oldtpath,newtpath)
#         # shutil.move(oldqpath,newqpath)
#         i = i + 1
#         print(i)

traindata = open('dfc2022_unlabeled.txt').read().splitlines()
traindata = [data.split('/')[-1][:-4] for data in traindata]
oldimgpath = 'crop512/train/arf'
newimgpath = 'crop512_nodata/arf'
# ch
oldlabpath = 'crop512/train/slope'
newlabpath = 'crop512_nodata/slope'

i = 0
traindir = recursive_glob(oldimgpath, '.tif')

for name in traindir:
    # if name.split('/')[-1][:-4].split('_')[0] in traindata:
    if name.split('/')[-1][:-4].replace('_arf','').replace('_slope','') in traindata:
        data = name.split('/')[-2:]
        oldrpath = name
        oldspath = name.replace('arf','slope')
        newrpath = oldrpath.replace(oldimgpath, newimgpath)
        newspath = oldspath.replace(oldlabpath, newlabpath)
        os.makedirs(os.path.dirname(newrpath), exist_ok=True)
        os.makedirs(os.path.dirname(newspath), exist_ok=True)
        
        shutil.move(oldrpath,newrpath)
        shutil.move(oldspath,newspath)
        # shutil.move(oldtpath,newtpath)
        # shutil.move(oldqpath,newqpath)
        i = i + 1
        print(i)
