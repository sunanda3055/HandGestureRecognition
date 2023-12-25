# Hand Gesture Recognition

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import gc
import pywt
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Image dimension details
IMG_SIZE = 128
img_w = 128
img_h = 128
NUM_OF_CLASS = 16


def change_img_scale(img, scale):
    if scale=='rgb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif scale=='gray':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img
    

def crop_hand(img):
    # Convert image to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle of the hand region
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Crop the image to the hand region
    cropped = img[y:y+h, x:x+w]
    
    return cropped


def processFrame(img):
    # Check number of channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Convert to 8-bit unsigned integer image
    img = np.uint8(img)

    # Apply bilateral filter to remove noise while preserving edges
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # Apply sharpening and enhancement using unsharp masking
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Apply cropping of hand region for better recognition
    img = crop_hand(img)
    
    # Resize the image to the input shape
    img = cv2.resize(img, (128, 128))
    
    # Apply Otsu's thresholding to create a binary image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area (presumably the hand region)
    max_contour = max(contours, key=cv2.contourArea)
    
    # Create a Binary mask of the hand region
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    # Apply bitwise_and operation to retain information only in the hand region
    masked_img = cv2.bitwise_and(img, mask)

    # Haar Wavelet Transform
    LL, (LH, HL, HH) = pywt.dwt2(masked_img, 'haar')
    img = np.concatenate((LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()))
    
    # Local Binary Pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(masked_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Fusion
    img = np.concatenate((img, hist))

    # Normalize pixel values
    img = masked_img / 255
    
    return img


x=[]
y=[]

TRAIN_PATH = '../handData'

count = 0
for number in os.listdir(TRAIN_PATH):
    print('***',number,'***')
    NUMBER_PATH = os.path.join(TRAIN_PATH, number)
    count = count + 1
    for pose in os.listdir(NUMBER_PATH):
        POSE_PATH = os.path.join(NUMBER_PATH, pose)
#         print(POSE_PATH)

        for img in os.listdir(POSE_PATH):
            IMG_PATH = os.path.join(POSE_PATH, img)
#             print(IMG_PATH) 
            
            frame = cv2.imread(IMG_PATH, cv2.COLOR_BGR2RGB)
            frameScale = change_img_scale(frame, scale='rgb')
            frameProcess = processFrame(frameScale)
            x.append(frameProcess)

            y.append(int(pose.split('_')[0]))            

#         break
#     break
            
            
x_data = np.array(x)
y = np.array(y)
y = y.reshape(len(x),1)


def one_hot_encoded(y):
    p = list(np.unique(y))
    dictionary = dict()
    final_result = []
    
    for i in range(len(p)):
        dictionary[p[i]] = i
        
    for i in y:
        actual = [0 for j in range(len(p))]
        actual[dictionary[i[0]]] = 1
        final_result.append(actual)
        
    return np.array(final_result)
    
y_data = one_hot_encoded(y)


x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

#Modified Alexnet-11 Layer Model
model = Sequential()

# First convolutional layer
model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', input_shape=(img_w, img_h, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Second convolutional layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Third convolutional layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fourth convolutional layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fifth convolutional layer with 256 filters
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Sixth convolutional layer with 256 filters
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Seventh convolutional layer with 256 filters
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Eighth convolutional layer with 512 filters
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Ninth dense layer with 4096 units
model.add(Flatten())
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Tenth dense layer with 4096 units
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Eleventh Output layer with 16 units
model.add(Dense(NUM_OF_CLASS, activation='softmax'))

model.summary()


def get_f1(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val


#Compiling model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall', get_f1])

EPOCHS = 10
BATCH_SIZE = 16

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(x_validate, y_validate),callbacks=[early_stop])

# Loss Curves

plt.plot(history.history['loss'], marker='o', markerfacecolor='blue')
plt.plot(history.history['val_loss'], marker='o', markerfacecolor='orange')
plt.legend(['train', 'test'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')

# Accuracy Curves

plt.plot(history.history['accuracy'], marker='o', markerfacecolor='blue')
plt.plot(history.history['val_accuracy'], marker='o', markerfacecolor='orange')
plt.legend(['train', 'test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')

acc_train = model.evaluate(x_train, y_train)
print('training accuracy:', str(round(acc_train[1]*100, 2))+'%')

acc_test = model.evaluate(x_test, y_test)
print('testing accuracy:', str(round(acc_test[1]*100, 2))+'%')

# Make predictions on test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred_classes))

class_name = ['palm', 'I', 'fist_moved', 'down', 'index', 'ok', 'palm_m', 'c', 'heavy', 'hang', 'two', 'three', 'four', 'five', 'palm_u', 'up']
class_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Compute normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print confusion matrix
print(cm)


cmap = sns.color_palette("Blues", as_cmap=True)

# Plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, cmap=cmap, fmt='g', xticklabels=range(16), yticklabels=range(16))

# Set plot labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_name, yticklabels=class_name, cmap="YlGnBu")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
