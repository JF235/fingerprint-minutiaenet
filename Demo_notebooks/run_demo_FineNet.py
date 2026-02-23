import matplotlib
matplotlib.use('Agg')
import sys,os
sys.path.append(os.path.realpath('../FineNet'))
import FineNet_model

from FineNet_model import FineNetmodel, plot_confusion_matrix

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['KERAS_BACKEND'] = 'tensorflow'



# ============= Hyperparameters ===============
batch_size = 32
num_classes = 2
path_to_model = '../Models/FineNet.h5'
input_shape = (224, 224, 3)
# ============= end Hyperparameters ===============


# =============== DATA loading ========================
test_path = '../Dataset/test/'

# Feed data from directory into batches
test_gen = ImageDataGenerator()
test_batches = test_gen.flow_from_directory(test_path, target_size=(input_shape[0], input_shape[1]), classes=['minu', 'non_minu'], batch_size=batch_size, shuffle=False)
# =============== end DATA loading ========================


#============== Define model ==================
model = FineNetmodel(num_classes = num_classes,
                     pretrained_path = path_to_model,
                     input_shape = input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0),
              metrics=['accuracy'])
#============== End define model ==============

score = model.evaluate_generator(test_batches)
print 'Test accuracy:', score[1]

test_labels = test_batches.classes[test_batches.index_array]
# ============= Plot confusion matrix ==================

predictions = model.predict_generator(test_batches)

cm = confusion_matrix(test_labels, np.argmax(predictions,axis=1))
cm_plot_labels = ['minu','non_minu']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# # Can use this
# from keras.preprocessing.image import load_img
# image = load_img('../Dataset/samples/m2.jpg',target_size=(224,224))

# or this
import cv2

image = cv2.imread('../Dataset/samples/m2.jpg')
image = cv2.resize(image, dsize=(224, 224),interpolation=cv2.INTER_NEAREST)
image = np.expand_dims(image, axis=0)

[class_idx] = np.argmax(model.predict(image),axis=1)
print class_idx
print test_batches.class_indices

