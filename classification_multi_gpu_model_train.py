from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.vgg19 import VGG19

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf

import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

import pickle

seed = 7
np.random.seed(seed)
 
batch_size = 24  

train_dir = './datasets/bird_pics/train'
valid_dir = './datasets/bird_pics/validate'

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True) # set as training data

validation_generator = valid_datagen.flow_from_directory(
    valid_dir, 
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True) # set as validation data

callbacks_list = [
        EarlyStopping(monitor='val_acc', patience=5),
        ReduceLROnPlateau(monitor='val_acc', factor=0.03, patience=3)
    ]

# models = ( ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3)),
#  InceptionResNetV2(weights=None, include_top=False, input_shape=(224, 224, 3)),
#  InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3)))

# model_generator = (model for model in models)
# del models
# model_names = ['ResNet50', 'InceptionResNetV2', 'InceptionV3']

name = "InceptionResNetV2"
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['acc'],
              optimizer=Adam(lr=1e-3))
    
hist = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    callbacks=callbacks_list,
    epochs = 100)

data = [hist.history['acc'], hist.history['val_acc']]
with open('./result/hist_model({0})_none.txt'.format(name), 'wb') as f:
    pickle.dump(data, f)

model.save_weights("./result/model({0})_weights_imagenet.h5".format(name))
model.save("./result/model({0})_weights_imagenet.h5".format(name))
model_json = model.to_json()
with open("./result/model({0})_imagenet.json".format(name), "w") as json_file:
    json_file.write(model_json)

fig, axes = plt.subplots(2)

twin_axes_ax = np.array((axes[0].twinx(), axes[1].twinx()))
axes[0].plot(hist.history['loss'], 'y', label='train loss')
axes[1].plot( hist.history['acc'], 'b', label='train acc')
twin_axes_ax[0].plot(hist.history['val_loss'], 'r', label='val loss')
twin_axes_ax[1].plot(hist.history['val_acc'], 'g', label='val acc')
axes[1].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].set_ylabel('accuray')
axes[0].legend(loc='upper right')
twin_axes_ax[0].legend(loc='center right')
axes[1].legend(loc='center right')
twin_axes_ax[1].legend(loc='lower right')
plt.show()
print("-------------\n\n")
print("{0}.\n\n".format(name))


test_dir = './datasets/bird_pics/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224), 
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical') 

outputs = np.zeros((test_generator.samples, test_generator[0][1].shape[1]), dtype='float32')
for i, (inputs, output) in enumerate(test_generator):
    outputs[i] = output
    if i >= test_generator.samples-1:
        break

yhat = model.predict(test_generator, steps=test_generator.samples // batch_size)
yhat_indexes = np.argmax(yhat, axis=1)
test_indexes = np.argmax(outputs, axis=1)
print("Confusion matrix : ")
print(confusion_matrix(test_indexes, yhat_indexes)) 
print("\n\n")
print("Classification report : ")
print(classification_report(test_indexes, yhat_indexes)) 
print("\n\n\n")