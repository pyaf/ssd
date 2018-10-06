# Code documentation.

I trained SSD (Single shot multibox detector) framework to detect faces in the provided images. The SSD code is adapted form of original open source ssd.pytorch implementation which can be found [here](https://github.com/amdegroot/ssd.pytorch/)

I converted the given .csv format annotations to xml annotations (like the ones used for VOC dataset). The code to do so can be found in `prepare_data.ipynb`. Now, I modified default configurations like num of classes, learning rate.

The base model is VGG16, the model uses pretrained weights trained on ImageNet dataset at the start of the training process.
The model was trained for 18000 iterations. The model performance can be seen in the `model_analyse.ipynb` file, it contains code to predict bounding boxes for faces in images using the trained model.



## train.py contains the training code.
## test_submission.py uses the trained model to predict on test cases and create submission.csv file

## weights folder contains:

* `vgg16_reducedfc.pth` weights of VGG model trained on imagenet used as base model in SSD
* `ssd300_FACE_18000.pth` trained weights of SSD on training set of the problem.

trained_model_path = 'weights/ssd300_FACE_33000.pth' is the best model to predict on. :) enjoy. fuck the world.