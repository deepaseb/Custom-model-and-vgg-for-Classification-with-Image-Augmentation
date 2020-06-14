import os
import numpy as np

from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense
from keras.applications.vgg16 import VGG16
from keras.models import Model


# Get the current working directory
PATH = os.getcwd()

#Define the data path
DATA_PATH = os.path.join(PATH, 'data')
data_dir_list = os.listdir(DATA_PATH)


# Required variables declaration and initialization
img_rows=224
img_cols=224
num_channel=3

num_epoch=10
batch_size=32

img_data_list=[]
classes_names_list=[]


# Read the images and store them in the list

import cv2

for dataset in data_dir_list:
    classes_names_list.append(dataset) 
    img_list=os.listdir(DATA_PATH+'/'+ dataset)
    for img in img_list:
        input_img=cv2.imread(DATA_PATH + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows, img_cols))
        img_data_list.append(input_img_resize)


# Get the number of classes
num_classes = len(classes_names_list)

# Image preprocessiong
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape

classes = np.ones((num_of_samples,), dtype='int64')

classes[0:202]=0
classes[202:404]=1
classes[404:606]=2
classes[606:]=3

# Convert class labels to numberic using on-hot encoding
classes = to_categorical(classes, num_classes)


# Shuffle the dataset
X, Y = shuffle(img_data, classes, random_state=2)

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# Defining the model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])


# Training/fit the model 
hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))


# Predict and compute the confusion matrix
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
confusion_matrix(np.argmax(y_test, axis=1), y_pred)


#Saving and loading model and weights
model.save_weights("model.h5")
loaded_model.load_weights("model.h5")
model.save('model.hdf5')
loaded_model = load_model('model.hdf5')


# Image Augmentation using ImageDataGenerator class

# Path to save Augmented Images
TRN_AUGMENTED = os.path.join(PATH , 'Trn_Augmented_Images')
TST_AUGMENTED = os.path.join(PATH , 'Tst_Augmented_Images')

train_data_gen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.5, 
    zoom_range=0.4, 
    vertical_flip=True,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True) 

test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        DATA_PATH,
        target_size=(img_rows, img_cols), 
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb', 
        shuffle=True,  
        save_to_dir=TRN_AUGMENTED, 
        save_prefix='TrainAugmented', 
        save_format='png')

test_generator = test_data_gen.flow_from_directory(
        DATA_PATH,
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb', 
        shuffle=True, 
        seed=None, 
        save_to_dir=TST_AUGMENTED, 
        save_prefix='TestAugmented', 
        save_format='png')


# Fit and predict 
model.fit_generator(train_generator, epochs=num_epoch, validation_data=test_generator,validation_steps=25,steps_per_epoch=X_train.shape[0]/batch_size)
fd_model_predict = model.predict_generator(test_generator, verbose=1,steps=X_test.shape[0]/batch_size)
fd_model_predict.argmax(axis=-1)


# Transfer Learning 
# Custom_vgg_model_1
#Training the classifier alone
image_input = Input(shape=(img_rows, img_cols, num_channel))
model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False

custom_vgg_model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

custom_vgg_model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

Y_train_pred = custom_vgg_model.predict(X_test)

y_train_pred = np.argmax(Y_train_pred, axis=1)


# Transfer Learning - 2

# Training the feature extraction also
model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')

last_layer = model.get_layer('block5_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
    layer.trainable = False

custom_vgg_model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


hist = custom_vgg_model2.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

Y_train_pred = custom_vgg_model2.predict(X_test)

y_train_pred = np.argmax(Y_train_pred, axis=1)

