CSC6621 Spring 2024
Dr. Barista

Amini, Arsalon
Bernitt, Jessie
Holcombe, Aidan

Instructions to Build / Run the Docker Image:

- Build the Docker Image from the DockerFile Using below commands

  1. docker build -t csc6621-final-project-image .

- Run the Docker Container using below commands
  1. docker run -p 8888:8888 csc6621-final-project-image

- Access the jupyternotebook by going to local host and entering the access token
  http://localhost:8888

Background on Dataset
CelebA is a large-scale face attributes dataset consisting of 200,000 celebrity images, each with 40 attribute annotations
Due to its large size and diversity, CelebA is suitable for pretraining facial recognition models, even with a limited number of examples for fine-tuning

Data Sets

1. Wild images (202,599 web face images)

- data_wild_img_celeba.7z.zip (img_celeba.7z.001....img_celeb1.7z.014)

2. aligned & cropped

- Images are first roughly aligned using similarity transformation according to the two eye locations;
- Images are then resized to 218\*178;

* data_img_crop_align_celeba_jpg.zip
* data_img_crop_align_celeba_png.zip (img_align_celeba_png.7z.001...img_align_celeba_png.7z.016)

Source:
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. "CelebA: A Large-Scale Celebrities Attributes Dataset." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
