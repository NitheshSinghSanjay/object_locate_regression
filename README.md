# Problem Description

This project goal is to find location of a phone dropped on the floor from a single RGB
camera image. This is a regression problem which can be solved using a simple convolution nural network. Example of the image is shown below:
<br /> <br />
<img src="find_phone/0.jpg">
<br /> <br />
Left-top corner of the image
is defined as (x, y) = (0, 0), left-bottom as (x, y) = (0, 1), right-top as (x, y) = (1, 0), and finally
right-bottom corner as (x, y) = (1, 1). Goal is to find normalized coordinates
of the center of the phone. In the example above, the coordinates of the phone are
approximately (x, y) = (0.83, 0.13).

# Dataset
Dataset contains 129 rgb images of size 490 x 326 x 3 and the labels are in the labels.txt file. Each line of the labels.txt is composed of img_path , x , y separated by spaces: <br />
img_path , x (coordinate of the phone), y (coordinate of the phone)

# Model Architecture
Below picture shows the architectural details of the CNN used to solve this problem. The architecture is quite big considering the simple problem. But smaller networks might also work fine when having a big dataset.
<br />
<img src="model.jpg" height=1000 width=500>
# Training

## Data Augmentation
Gaussian blur is applied on the 129 images using scipy python module. After data augmentation there are 258 images for training. Additional data augmentation such as horizontal and vertical flips can be applied.
To apply data augmentation run the below code. "./find_phone" is the directory where the images are stored.<br>
` $ python augment_data.py ./find_phone ` <br>

## Train
