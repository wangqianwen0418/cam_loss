# cam_bbox
this repo contains three parts:

1. a keras implementation of (Class Activation Mapping) CAM    
a CVPR16 paper, https://arxiv.org/pdf/1512.04150.pdf  
you can find the code at model/cam_keras.py

2. an algorithm that adds extract heatmap loss to tell a classifier model where it should look   
the code is at model/model.py  
we fine tune a vgg19 model that has been pretrained in imageNet   
by default, it uses Microsoft COCO dataset and tries to find out whether an image has a cat or not   
(you should create a folder named data and download coco dataset in it)  
to train the model  
```$ python model.py --exp your_experiment_name --year the_year_of_the_cocodataset --bbox whether_to_use_the_extract_loss --per percentage_of_the_training_data_to_use```

3. a website to show and compare the results generated by models with and without the heatmap loss  
run ```python damp.py``` to save the results as npy  
run ```python show.py``` generate heatmap images and t-sne positions for each images  
follow the readme file in frontEnd folder to run the website 


all the used python packages are in requirements.txt, 
you can install all of them by running  
```$ pip install -r requirements.txt```