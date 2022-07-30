import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import random_split
import glob
import cv2
import copy
import matplotlib.pyplot as plt

def mapping_colors(labels):
    colors = []
    for i in range(len(labels.labels)):
        colors.append(list(labels.labels[i].color))

    colors = np.array(colors)
    encoder = {tuple(c) : t for c, t in sorted(zip(colors.tolist(), range(len(colors))))}
    encoder.update({(190, 153, 153):13})
    encoder.update({(152, 251, 152):22})
    encoder.update({(250, 170, 160): 9})
    decoder = {encoder[k]: k for k in encoder}
    return encoder, decoder

def splitDataset(dataset, valSize=0.2):
    lenDataset = int(len(dataset))
    train = int(lenDataset - (lenDataset * valSize))
    val = int(lenDataset - train)
    lengths = [train, val]
    trainDataset, valDataset = random_split(dataset, lengths=lengths)
    return trainDataset, valDataset

def crop_image(layer, target_size):
    _, _, h, w = layer.size()
    y = (h - target_size[0])//2
    x = (w - target_size[1])//2
    return layer[:,:, y:(y+target_size[0]), x:(x+target_size[1])] 

def image_prediction(image_path, transform, engine, mapping, rev_mapping):
    image = Image.open(image_path).convert('RGB')
    image_processed = transform(image)
    image = image_processed.unsqueeze(0)
    pred = engine.predict(image)
    pred_mask = torch.zeros(3, pred.size(0), pred.size(1), dtype=torch.uint8)
    for k in rev_mapping:
        pred_mask[:, pred==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
    return pred_mask.permute(1, 2, 0).numpy()

def video_prediction(video, namePrediction,engine, transform, mapping, rev_mapping):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video = cv2.VideoWriter(namePrediction, cv2.VideoWriter_fourcc(*'XVID'), fps, (1234, 366))
    
    if (cap.isOpened() == False):
        print("Error Opening Video")
    
    success, image = cap.read()

    while success:
        image_copy = copy.deepcopy(image)
        image = Image.fromarray(image_copy).convert('RGB')
        imageTensor = transform(image)
        imageTensor = frameTensor.unsqueeze(0)
        prediction = engine.predict(imageTensor)
        pred_mask = torch.zeros(3, prediction.size(0), prediction.size(1), dtype=torch.uint8)
        for k in rev_mapping:
            pred_mask[:, prediction==k] = torch.zeros(rev_mapping[k]).byte().view(3, 1)
        pred = pred_mask.permute(1, 2, 0).numpy()

        video.write(pred)

        success, image = cap.read()

    video.release()
    cap.release()
    cv2.destroyAllWindows()

def meanImageSize(images):
    w = h = 0
    for image in images:
        img = Image.open(image).convert('RGB')
        w += img.size[0]
        h += img.size[1]
    return w//len(images), h//len(images)

def predictionOnTestImages(engine, transform, MAPPING, REV_MAPPING):
    imagesTesting = list(sorted(glob.glob(os.path.join("../ImageSegmentation/data/testing/image_2", "*.png"))))
    index = np.random.randint(0, len(imagesTesting))
    print("Test Index : ",index)
    image = imagesTesting[index]
    mask = image_prediction(image, transform, engine, MAPPING, REV_MAPPING)

    image = cv2.imread(imagesTesting[index])
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].imshow(image)
    ax[0].set_title("Original Image", size=18)
    ax[1].imshow(mask)
    ax[1].set_title("Prediction", size=18)
    plt.show()

