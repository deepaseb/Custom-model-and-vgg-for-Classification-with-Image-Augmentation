The python file consists of following codes:
1) Build CNN from scratch
     - Custom CNN model is created for classifying images.
2) Image Augmentation using 'flow_from_directory':
      - It takes the path to a directory & generates batches of augmented data.
3) Transfer Learning using VGG16 Architecture:
       - 2 types of transfer learning is performed:
           - Only the final layer is trained for 4 classes and the rest of the other layers are kept frozen.
           - 3 new layers(Flatten,Dense,Output) are added and training is performed.
