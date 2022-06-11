# MS2Net
This is a PyTorch implementation of the model in this paper:

[Multi-Stage Fusion and Multi-Source Attention Network for Multi-Modal Remote Sensing Image Segmentation](https://doi.org/10.1145/3484440)

## Dependencies
* PyTorch 1.6.0
* torchvision 0.7.0
* torchsummary 
* numpy
* imageio
* tqdm
* opencv-python
* Pillow
* tensorboard
* tifffile

## Tips
Please modify the ```dataloader/dataset.py``` according to the name of the images in the dataset

Train:
```
python main.py --savedir=' Model save path ' --lr=1e-4 --step_loss=50
```

## Citation
Please cite this paper if you use this code in your own work:
```
@inproceedings{ZhaoMS2,
  title={Multi-Stage Fusion and Multi-Source Attention Network for Multi-Modal Remote Sensing Image Segmentation},
  author={Jiaqi Zhao, Yong Zhou, Boyu Shi, Jingsong Yang, Di Zhang, Rui Yao},
  booktitle={ACM Transactions on Intelligent Systems and Technology, Association for Computing Machinery (ACM)},
  year={2021}
}
```