# Image Captioning with RNN

### Dataset
An LSTM network is trained on the [VizWiz-Captions dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). It consists of 23,431 training images with 117,155 training captions; 7,750 validation images with 38,750 validation captions; 8,000 test images with 40,000 test captions (unavailable to developers; reserved for the image captioning competition). Here, a subset of the train images (8000 images in the ‘train1’ dataset) is used to train the RNN; and the validation images are used to validate the performance of image captioning.

### Approach

<img src="https://user-images.githubusercontent.com/51696913/169579429-90ccab9f-5cbb-46f9-8bc0-ad3f29763d91.png" width="600">

Certain data pre-processing steps such as converting all letters to lowercase, eliminating special characters, removing numbers from captions are done, and a vocabulary consisting of all the unique words across the train captions is created. Special tokens – ‘startseq’ and ‘endseq’ are added to each caption to indicate the beginning and ending of captions. A pre-trained GLOVE model is used for creating the word embedding matrix is taken as the input.

A pre-trained InceptionV3 model on the ImageNet dataset (transfer learning) is used to extract a 2048 length vector for each image (this is shown as Input 2 in the image above). A generator model is used to train images, in a sequential manner using partial captions as input; when the model is given ‘startseq’ as input it tries to predict the next word in the caption as its immediate output, and so on.

The outputs from the embedding layer are given to the LSTM layer which is then merged into a single tensor with the outputs from the dense layer of the image input layer. Dropout layers are added to prevent the model from overfitting. The final dense layer is of size 2514 since the vocabulary that we have created for the training set is of the same shape, and the word with the maximum probability is selected from the softmax layer. All models are trained for a total of 20 epochs.

GLOVE Word Embedding model - https://www.kaggle.com/datasets/incorpes/glove6b200d 

### Directory
```
Dataset/ 
train/
├─ data/
├─ descriptions.txt
├─ encoded_train_images.pkl
├─ img_ids.txt
├─ annotations_train1.txt
val/
├─ data/
├─ descriptions.txt
├─ encoded_test_images.pkl
├─ img_ids.txt
├─ annotations_val.txt
glove.6B.200d.txt
model_weights/
image_captioning.ipynb
```

### Results

<img src = "https://user-images.githubusercontent.com/51696913/169579946-a0c30bd9-f12e-42b9-9982-4ddb06e65512.png" width = "300">
<img src = "https://user-images.githubusercontent.com/51696913/169579968-e4ddf32b-cd66-4040-ac13-baaff4ce57ce.png" width = "300">
<img src = "https://user-images.githubusercontent.com/51696913/169579976-d55c0ebc-25bf-466a-ba54-21eb95401a1e.png" width = "300">

*Training Performance - Optimizing the RNNs*

<img src = "https://user-images.githubusercontent.com/51696913/169579983-7f864626-bf42-4b97-aeea-f993ff4aa27c.png" width = "500">

*Validation Performance - Average of BLEU scores over the validation set*

<img src = "https://user-images.githubusercontent.com/51696913/169579989-e0b96582-5e0c-42d4-a949-fe214662c0bf.png" width = "800">


