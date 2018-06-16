# Transfer Learning for Food Classification

This project is a food classification transfer learning project based on pretrained VGG16, VGG19 models. 
## Getting Started
* **please download the data first**

### Discription

### Requirements
* **Package**
* **Python 3.6.4**
* **Pytorch 0.4.0**
* **OpenCV**
```shell
pip install Python
pip install pytorch
sudo apt install opencv-python
```


* **Dataset**
* The original Food 101 dataset can be download from [http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)


* **Models**
* Due to the limitation of the volume, we put trained models as wellas extracted features in the [Google Drive](https://drive.google.com/open?id=1y1U0QvATyppVD1IvNcizq6X6QyjuN89g)
* Pleas put those data files in the same directory of the notebooks.

Example files:
```
./test_data.pkl
./test_data_vgg16vgg19.pkl
./test_data_vgg19.pkl
./test_vgg16vgg19_data.pkl
./training_data.pkl
./training_data_vgg16vgg19.pkl
./training_data_vgg16vgg19_1.pkl
./training_data_vgg16vgg19_2.pkl
./training_data_vgg16vgg19_3.pkl
./training_data_vgg16vgg19_all.pkl
./training_data_vgg19.pkl
./vgg16+vgg19_data.pkl
./vgg16_data.pkl
./vgg19_data.pkl

./show_most_confusion.ipynb
./error_analysis.ipynb
./vgg19_acc_topn.ipynb
./show_topn_vgg19.ipynb
./plots
./data_preprocess_vgg16+vgg19.ipynb
./vgg16_acc_topn.ipynb
./error_heatmap.ipynb
./guided_backprop.ipynb
./gradcam.ipynb
./vanilla_backprop.ipynb
./show_topn_vgg16.ipynb
./data_preprocess_alexnet.ipynb
./data_preprocess_densenet.ipynb
./README.md
./data_preprocess_vgg19.ipynb
./data_preprocess_vgg16.ipynb
./vgg16+vgg19_acc_topn.ipynb
./misc_functions.py



```



### Code organization 

* ** Feature Extraction
* **data_preprocess_vgg16.ipynb** : Extract features from food images using the VGG16 pretrained model.
* **data_preprocess_vgg19.ipynb** : Extract features from food images using the VGG19 pretrained model.
* **data_preprocess_vgg16+vgg19.ipynb** : Extract features from food images using the VGG16 and VGG19 pretrained model.

* **Training**
* **vgg16_acc.ipynb** : Train the FoodNet with features extracted by VGG16.
* **vgg19_acc.ipynb** : Train the FoodNet with features extracted by VGG19.
* **vgg16+vgg19_acc.ipynb** : Train the FoodNet with features both from VGG16 and VGG19.

* **Result**
* **show_topn_vgg16.ipynb** : Plot the topN results.
* **show_topn_vgg19.ipynb** : Plot the topN results.

* **Analysis**
* **error_analysis.ipynb**: Error analysis.
* **error_heatmap.ipynb**  : Error analysis with heatmap.
* **show_most_confusion.ipynb** : Show the pictures which are the most confusing pictures. 

* **CNN Visualization**
* **Reference**: [https://github.com/utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* **misc_function** : misc function
* **gradcam.ipynb** : Gradient-weighted Class Activation Map
* **guided_backprop.ipynb** : Guided Backpropagation
* **vanilla_backprop.ipynb** : Vanilla Backpropagation
















