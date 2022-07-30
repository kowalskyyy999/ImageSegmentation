#!/usr/bin/python3
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class KittiDataset(Dataset):
    def __init__(self, root_dir, images, masks=None, mapping=None, transform=None, resize_mask=None):
        self.root_dir = root_dir
        self.images = images
        self.masks = masks
        self.mapping = mapping
        self.transform = transform
        self.resize_mask = resize_mask
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, 'image_2', self.images[index])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        if self.masks is not None: 
            mask_path = os.path.join(self.root_dir, 'semantic_rgb', self.masks[index])
            mask = Image.open(mask_path)
            if self.resize_mask is not None:
                mask = mask.resize((self.resize_mask[1], self.resize_mask[0]), resample=Image.Resampling.NEAREST)
            mask = np.array(mask)
            mask = self.mask_to_class(mask)
            
            return {'image':image, 'mask':mask}
        
        return {'image':image, 'mask':None}
    
    def mask_to_class(self, mask):
        target = torch.from_numpy(mask)
        h, w = target.shape[0], target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        target = target.permute(2, 0, 1).contiguous() # 3, h, w
        for k in self.mapping:
            idx = (target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            valid_idx = (idx.sum(0) == 3)
            masks[valid_idx] = torch.tensor(self.mapping[k], dtype=torch.long)
        return masks