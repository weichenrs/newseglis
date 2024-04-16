import os
import torch
from configs import parse_args
from models import deeplab
from metrics.stream_metrics import StreamSegMetrics
from data.palette import get_palette
from torch.utils.data import Dataset
from PIL import Image
from data.transform import val_transform
from torch.utils.data import DataLoader
import numpy as np
from configs.config import FedFBP_names,FedInria_names,FedHBP_names
Image.MAX_IMAGE_PIXELS = 230000000000
#  Set GPU
#--------------------------------------------------
device_id = 2
os.environ["CUDA_VISIBLE_DEVICES"]=str(device_id)
torch.cuda.empty_cache()
args = parse_args('FedHBP')



class TestDataset(Dataset):
  def __init__(self, data_dir, client_id,transform=None):
    super(TestDataset, self).__init__()
    self.data_dir = data_dir
    self.client_id = client_id
    self.transform = transform
    client_dir = os.path.join(data_dir, 'A{}'.format(client_id))
    assert os.path.exists(client_dir), 'A{} does not exist.'.format(client_id)
    
    self.images_dir = os.path.join(client_dir, 'image')
    self.masks_dir = os.path.join(client_dir, 'mask')
    name_list_path = os.path.join(data_dir, 'A{}test.txt'.format(client_id))
    with open(name_list_path, 'r') as f:
      name_list = f.readlines()
    name_list =[name[:-5] for name in name_list]
  
    self.images_path = [os.path.join(self.images_dir, name.strip()+'.jpg') for name in name_list]
    self.masks_path = [os.path.join(self.masks_dir, name.strip()+'.png') for name in name_list]
    assert len(self.images_path) == len(self.masks_path), 'The number of images and masks does not match.'
    self.num_samples = len(self.images_path)
    print('Client {} has {} samples.'.format(client_id, self.num_samples))
    
  def __getitem__(self, index):
    image_path = self.images_path[index]
    mask_path = self.masks_path[index]
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
    if self.transform is not None:
      transformed = self.transform(image=image, mask=mask)
      image = transformed['image']
      mask = transformed['mask']
    return mask_path,image, mask
  
  def __len__(self):
    return self.num_samples













test_dataset = TestDataset('../../datasets/HBP/HBP', 1, val_transform)







palette = get_palette('./data/'+ args.dataset +'_palette.json')
if(args.model == 'deeplabv3plus_resnet50'):
      model = deeplab.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride)

checkpoint_path = './checkpoints/2024-XXXXXX/XXXXXX.pth'
model.load_state_dict(torch.load(checkpoint_path,map_location='cpu')['model_state'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


metrics = StreamSegMetrics(args.num_classes)
metrics.reset()
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
row = 12500
col = 12500
window_size = 1024
central_size = 512

row_patch = int(row/central_size)+1
col_patch = int(col/central_size)+1

merge_row = row_patch*central_size + window_size - central_size
merge_col = col_patch*central_size + window_size - central_size


edge =int((window_size-central_size)/2)

with torch.no_grad():
    for k, (path, images, masks) in enumerate(val_loader):
        # slide window predict and merge
        merge_predict = np.zeros((1,merge_row,merge_col))
        name = path[0].split('/')[-1]
        images_padded = torch.zeros(1,images.shape[1],merge_row,merge_col)
        
        images_padded[:,:,edge:edge+images.shape[2],edge:edge+images.shape[3]] = images
        masks_padded = torch.zeros((1,merge_row,merge_col))
        masks_padded[:,edge:edge+masks.shape[1],edge:edge+masks.shape[2]] = masks
        index= 0
        for i in range(0,images.shape[3],int(window_size/2)):
          for j in range(0,images.shape[2],int(window_size/2)):
            index += 1
            print(index)
            image_patch = images_padded[:,:,j:j+window_size,i:i+window_size]
            mask_patch = masks_padded[:,j:j+window_size,i:i+window_size]
            image_patch = image_patch.to(device, dtype=torch.float32)
            mask_patch = mask_patch.to(device, dtype=torch.long)
            outputs = model(image_patch)
            outputs = outputs[:,1:,:,:]
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            preds = preds + 1
            targets = mask_patch.cpu().numpy()
            preds[targets == 0] = 0
            preds = preds[:,edge:edge+central_size,edge:edge+central_size]
            targets = targets[:,edge:edge+central_size,edge:edge+central_size]
            metrics.update(targets, preds)

            merge_predict[:,j+edge:j+edge+central_size,i+edge:i+edge+central_size] = preds
            
        merge_predict = merge_predict[:,edge:edge+row,edge:edge+col]
        # save predict
        merge_predict = merge_predict.squeeze()
        merge_predict = merge_predict.astype(np.uint8)
        merge_predict = Image.fromarray(merge_predict)
        merge_predict.putpalette(palette)

        save_path = os.path.join('./predict/HBP1', name)
        merge_predict.save(save_path)
        masks = masks.squeeze()
        masks = masks.detach().cpu().numpy().astype(np.uint8) 
        masks = Image.fromarray(masks)
        masks.putpalette(palette)
        mask_path = os.path.join('./predict/HBP1', name[:-4]+'_mask.png')
        masks.save(mask_path)
        print('predict saved to {}'.format(save_path))



    if args.dataset == 'FedFBP':
      score = metrics.get_results(FedFBP_names)
    if args.dataset == 'FedHBP':
      score = metrics.get_results(FedHBP_names)
      
    
    
    confusion_matrix = score['confusion_matrix']
    iou = score['Class IoU']
    
    miou = score['Mean IoU']
    with open('./predict/HBP1/score1.txt', 'w') as f:
      f.write('Mean IoU: ' + str(score['Mean IoU']) + '\n')
      for k in iou.keys():
        f.write(k + ': ' + str(iou[k]) + '\n')
      for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
          f.write(str(confusion_matrix[i][j]) + ' ')
        f.write('\n')


