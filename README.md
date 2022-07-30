# Re-Implementation DeepLabV3Plus Image Segmentation

## Getting Started
A Re-Implementation project Image Segmentation Using Architecture [DeepLabV3Plus](https://arxiv.org/abs/1802.02611). The Dataset is [CityScapes Dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)

### Training
First of all you must download and extract the [dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) in *data* folder. Then download the labels IDs, names and instance classes of the Cityscapes dataset are used and can be found [here](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)

```python
# Training 
>>> python3 train.py --model ${Name of Model Save} --batch-size ${Number of Batch Size training data} --batch-size-val ${Number of Batch Size validation data} --epochs ${Number of Epochs} --lr ${Number of Learning Rate}
>>> python3 train.py --model model.pth --batch-size 16 --batch-size-val 16 --epochs 100 --lr 1e-4
```

### Inference
For Inference have two mode, image prediction and video prediction
1. Image Prediction
```python
>>> python3 inference.py --model ${Name of Model Saved} --image ${Path of Single Image file} --save ${Bool to save the image} --name-prediction ${Name of Image Prediction}
>>> python3 inference.py --model model.pth --image images.png --save --name-prediction images_prediction.png  # Image Prediction with Save the Prediction to a file
>>> python3 inference.py --model model.pth --image image.png    # Image Prediciton just show the Result
```

2. Video Prediction
```python
>>> python3 inference.py --model ${Name of Model Saved} --video ${Path of Video file} --video-prediction ${Name of Video Prediction}
>>> python3 inference.py --model model.pth --video videos.mp4 --video-prediction videos_prediction.mp4
```
