#!/usr/bin/env python
# coding: utf-8

# In[1]:


#frequently used or necessary libraries
import random
import numpy as np
import tensorflow as tf
import re
from tensorflow.python.client import device_lib
import torch
from tqdm import tqdm
from exe import exe


# From here, starts the relevant part.

# In[4]:


#The code works with tensorflow 1 (1.15), but I have tensorflow 2 that I will use as tf1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[5]:


#necessary libraries, functions, layers, etc.
from sklearn.metrics import classification_report, precision_recall_fscore_support
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from IPython.display import Image
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from os import listdir


# We will use the Inception v3 model with transfer learning, due to the small size of the dataset. The point is to use a huge, already trained model as a feature extractor, and build a classifier over it. 
# 
# I used 10 relatively different spices here. For ease of use, I batch renamed the pictures so the filenames always start with their classes, for example: anisedownload (14).jpeg instead of download (14).jpeg
# 
# You should look up and understand every words you don't understand here. You don't necessarily have to understand fully the mathematics behind everything, but at least get the picture. Definitely learn about the Inception V3 model, and also transfer learning.
# 
# 1. Preprocess dataset
#     1. Create integer labels for the classes 
#     2. Resize every picture to 299x299 pixels (standard size for Inception v3)
#     3. Put the picture to an array (vector space)
#     4. Use the preprocessor for this particular model 
#     5. Store the arrays and labels in two separate lists
#     6. Split dataset into train 70% and test 30% sets (due to the limited sample size, use stratified splitting to avoid bias)
#     7. As our loss function will be categorical cross entropy (for multiclass), convert y labels to a one-hot encoded format
# 2. Define our model
#     1. Load the already pretrained Inception v3 model, but without the classifier part (as we will just use this model as a feature extractor)
#     2. To avoid contradicting dimensions, flatten the output of the Inception v3 model (again, without the classifier part)
#     3. Add a simple dense layer to learn more (we can experiment with other type of layers, or add more, etc.)
#     4. Add a classification layer with softmax activation (multiclass sigmoid, basically)
# 3. Compile and fit the model
#     - We can try other optimizers, but looks like Adam with default parameters works well in this case
#     - We can try other batch sizes, but due to the size of the dataset, I wouldn't go over 64
#     - It's already somewhat overfitting around 100 epochs, can try dropouts or reguralizers to overcome this
# 4. Evaluate the model
#     - For 10 classes, the achieved 76.5% accuracy is actually pretty good. The class-wise precision, recall, and F1 values are also good.
#     - You can try more classes, or more similar classes, or both, because one of the reasons it's that good is that even simple the colors of these spices/herbs are quite different mostly. Good example for this is the chili powder, look at the precision, recall, and f1 values for it...

# In[9]:


folder = 'classes/trial/combined/'
photos, labels = list(), list()
for file in listdir(folder): 
    if file.startswith('anise'):
        output = 0.0
    elif file.startswith('basil'):
        output = 1.0
    elif file.startswith('chili_powder'):
        output = 2.0
    elif file.startswith('cloves'):
        output = 3.0
    elif file.startswith('coriander'):
        output = 4.0
    elif file.startswith('mustard'):
        output = 5.0
    elif file.startswith('peppermint'):
        output = 6.0
    elif file.startswith('poppy_seeds'):
        output = 7.0
    elif file.startswith('vanilla'):
        output = 8.0
    elif file.startswith('wasabi'):
        output = 9.0

    photo = load_img(folder + file, target_size=(299, 299))
    photo = img_to_array(photo)
    #photo = photo.reshape((1, photo.shape[0], photo.shape[1], photo.shape[2]))
    photo = preprocess_input(photo)
    photos.append(photo)
    labels.append(output)

photos = np.asarray(photos)
labels = np.asarray(labels)

np.save('spices_photos.npy', photos)
np.save('spices_labels.npy', labels)


# In[10]:


print(photos.shape, labels.shape)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(photos, labels, test_size=0.3, stratify=labels)


# In[12]:


y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=10), tf.keras.utils.to_categorical(y_test, num_classes=10)


# In[13]:


model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)
model = Model(inputs=model.inputs, outputs=output)
#let's see our full model
model.summary()


# In[15]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32)


# In[17]:


score = model.evaluate(X_test,
                       y_test,
                       batch_size=32,
                       verbose=0)


# In[18]:


print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


# In[30]:


y_pred = model.predict(X_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)


# In[37]:


labels=['anise','basil','chili_powder','cloves','coriander','mustard','peppermint','poppy_seeds','vanilla','wasabi']


# In[40]:


print(classification_report(np.argmax(y_test, axis=1), y_pred_bool, target_names=labels))


# In[46]:


#save model
model.save('spice_model')


# In[47]:


#this is how the model looks (I used Netron, to visualize the architecture from the saved model)
Image(filename='spice_model.png') 


# In[ ]:




