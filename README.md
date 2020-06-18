This project is to learn how to build a simple CNN network from scratch for classification. Also,image augmentation and image classification using VGG16 architecture is performed . 

The images used for training and validation is saved in the data.zip folder.It consists of 4 classes - Dogs,Cats,Horses,Humans; each with 202 images.
The images generated for train and test are saved in Trn_Augmented_Images and Tst_Augmented_Images folders respectively.

The python file(Codefile.py) consists of following steps:
1) Build CNN from scratch
     - Custom CNN model is created for classifying images.
2) Image Augmentation using 'flow_from_directory':
      - It takes the path to a directory & generates batches of augmented data.
3) Transfer Learning using VGG16 Architecture:
      - 2 types of transfer learning is performed:
         - Only the final layer is trained for 4 classes and the rest of the other layers are kept frozen.
         - 3 new layers(Flatten,Dense,Output) are added and training is performed.
