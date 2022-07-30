#!/usr/bin/python3
import cv2
import argparse
import matplotlib.pyplot as plt
from scripts.utils import *
from scripts.model import *
from scripts.engine import *
import torch
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, action='store', help="Image path")
parser.add_argument("--model", type=str, action='store', help="Name of Model saved")
parser.add_argument("--save", action='store_true', help="Save image prediction")
parser.add_argument("--image-prediction", type=str, action='store', help="Name of Image prediction")
parser.add_argument("--video-prediction", type=str, action="store", help="Name of Video Prediction")
parser.add_argument("--video", nargs='?', const=False, type=str, help="Video FileName")
args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.Resize((375, 1241)),
        T.ToTensor()
        ])
    model = DeepLabV3Plus(3, 35, 32, (375, 1241))
    
    if device.type == 'cuda':
        print("Load Model in GPU Device")
        payload = torch.load(f'{args.model}')
    else:
        print("Load Model in CPU Device")
        payload = torch.load(f'{args.model}', map_location=device)

    model.load_state_dict(payload['state_dict'])
    encoder = payload['encoder']
    decoder = payload['decoder']
    
    engine = Engine(model, optimizer=None, criterion=None, epochs=None, device=device)

    if args.video:
        video_prediction(args.video, args.video_prediction, engine, transform, encoder, decoder, device)
    else:
        mask = image_prediction(args.image, transform, engine, encoder, decoder, device)
        plt.imshow(mask)
        plt.title("Prediction")
        plt.axis(False)
        if args.save:
            plt.savefig("predict/{args.image_prediction}")
        plt.show()

if __name__ == "__main__":
    main()
