# Estimation of Rotation Angle

The code is the homework of the multimedia content security course, the reproduction of the rotation angle estimation algorithm of the paper "Estimation of Image Rotation Angle Using Interpolation-Related Spectral Signatures With Application to Blind Detection of Image Forgery" 

## Requirements

```
cv2               4.4.0
numpy             1.19.2
matplotlib        3.3.4
PIL               8.1.2
```

## Dataset

USC-SIPI database can be downloaded from [here](http://sipi.usc.edu/database/).

Use script files `tiff2bmp.py` to convert tiff format to bmp format 

These images were rotated using the watermark attacking tool StirMark 4.0, which can be downloaded from [here](http://www.petitcolas.net/fabien/watermarking/stirmark/).

Put rotated image in `/BMPdataset` directory.

## Usage

```sh
bash run.sh
```

## Citation

```
@inproceedings{
  title={Estimation of Rotation Angle},
  author={Chengyuan Mai},
  year={2021}
}
```