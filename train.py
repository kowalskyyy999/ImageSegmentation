#!/usr/bin/python3
import sys 
import importlib.util

from scripts.dataset import *
from scripts.engine import *
from scripts.lossFn import *
from scripts.model import *
from scripts.utils import *

from sklearn.model_selection import train_test_split

import torchvision.transform as T
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs='?', type=str, const="model.pth", help="The Name of Model Saving")
parser.add_argument("--batch-size", nargs='?', type=int, const=16, help="Batch Size for Train Data")
parser.add_argument("--batch-size-val", nargs="?", type=int, const=16, help="Batch Size for Validation/Testing Data")
parser.add_argument("--epochs", type=int, action="store", help="Number of Epochs for Training")
parser.add_argument("--lr", type=float, action="store", help="Number of Learning Rate for RMSprop optimizers")
args = parser.parse_args()

TRAIN_BS = 16
VAL_BS = 16
EPOCHS = 100

def main():
    rootDir = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training running on {device} Device")

    spec = importlib.util.spec_from_file_location('labels', "data/devkit/helpers/labels.py")
    labels = importlib.util.module_from_spec(spec)
    sys.modules['labels'] = labels
    spec.loader.exec_module(labels)
    
    encoder, decoder = mapping_colors(labels)
    
    trainTransform = T.Compose([
        T.Resize((375, 1241)),
        T.ColorJitter(brightness=0.6, contrast=0.8, saturation=0.2),
        T.ToTensor()])
    
    valTransform = T.Compose([
        T.Resize((375, 1241)),
        T.ToTensor()
    ])

    images = list(sorted(os.listdir(os.path.join("data/training", 'image_2'))))
    masks = list(sorted(os.listdir(os.path.join("data/training", 'semantic_rgb'))))

    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.3, random_state=42)

    train_dataset = KittiDataset(rootDir, images=train_images, masks=train_masks, mapping=encoder, transform=trainTransform, resize_mask=(366, 1234))
    val_dataset = KittiDataset(rootDir, images=val_images, masks=val_masks, mapping=encoder, transform=valTransform, resize_mask=(366, 1234))
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)
   
    model = DeepLabV3Plus(3, 35, 32, (375, 1241))
   
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    
    engine = Engine(model, optimizer, criterion=getLoss, epochs=args.epochs, device=device)
    
    engine.fit(trainLoader, valLoader)
    
    payload = {'state_dict': engine.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'encoder':encoder,
            'decoder':decoder}
    torch.save(payload, args.model)
    print('Training Successfully!!!')

if __name__ == "__main__":
    main()
