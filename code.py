!pip install kaggle

import kaggle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from tensorflow.keras.models import Model

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
! unzip chest-xray-pneumonia.zip

TRAINING_DATA_PATH = "./chest_xray/train/"
TESTING_DATA_PATH = "./chest_xray/test/"
VALIDATION_DATA_PATH = "./chest_xray/val/"
NUM_FT_EPOCHS = 6
NUM_T_EPOCHS = 15
BATCH_SIZE = 32
IMG_HEIGHT =299
IMG_WIDTH = 299

gen = ImageDataGenerator(rescale = 1./255,
                         zoom_range = 0.05,
                         width_shift_range = 0.05,
                         height_shift_range = 0.05,
                         brightness_range = [0.95,1.05])

train_data_gen= gen.flow_from_directory(
    directory = TRAINING_DATA_PATH ,#"/content/gdrive/My Drive/chest_xray_images/train/",
    color_mode = "rgb",
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 42)

test_gen = ImageDataGenerator(rescale = 1./255.)
valid_data_gen = test_gen.flow_from_directory(
    directory   = VALIDATION_DATA_PATH, #"/content/gdrive/My Drive/chest_xray_images/val/",
    color_mode  = "rgb",
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode  = "categorical",
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    seed        = 42)

test_data_gen= test_gen.flow_from_directory(
    directory   = TESTING_DATA_PATH, #"/content/gdrive/My Drive/chest_xray_images/test/",
    color_mode  = "rgb",
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode  = "categorical",
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    seed        = 42)

var = train_data_gen.class_indices
class_names = list(var.keys())
freq = np.unique(train_data_gen.classes, return_counts=True)

#plt.title("Trainning dataset")
plt.figure(figsize=(10, 6))
plt.bar(class_names, freq[1])
plt.title("Class Distribution in Training Dataset")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# clearly an umbalanced data set, so some class weights
cls_wg = compute_class_weight(class_weight = "balanced", classes= np.unique(train_data_gen.classes), y= train_data_gen.classes)
cls_wg = dict(zip(np.unique(train_data_gen.classes), cls_wg))

def draw_img(img, true_labels, predictions = None):
    plt.figure(figsize=[12, 18])
    for i in range(24):
        plt.subplot(6, 4, i+1)
        plt.imshow(img[i])
        plt.axis('off')
        if (predictions is not None):
            plt.title("{}\n {} {:.1f}%".format(class_names[np.argmax(true_labels[i])], class_names[np.argmax(predictions[i])], 100 * np.max(predictions[i])))
        else:
            plt.title(class_names[np.argmax(true_labels[i])])

x,y = next(train_data_gen)
draw_img(x,y)

def spot_hm(y_true, y_pred, class_names, ax, title):
    c_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = c_matrix[1,1]/(c_matrix[1,0]+c_matrix[1,1])
    specificity = c_matrix[0,0]/(c_matrix[0,0]+c_matrix[0,1])
    new_title = f'{title}\n Sensitivity = {sensitivity:.2f} Specificity = {specificity:.2f}'
    sns.heatmap(c_matrix, annot=True, square=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap=plt.cm.Blues, cbar=False, ax=ax)
    ax.set_title(new_title, fontsize = 12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "right")
    ax.set_ylabel('True Label', fontsize= 10)
    ax.set_xlabel('Predicted Label', fontsize = 10)

def grad_hm(image, model, last_conv_layer_name):
    img_array = tf.expand_dims(image, axis=0)
    last_layer_activation = model.layers[-1].activation
    model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)
    j_et = cm.get_cmap("jet")
    j_et_colors = j_et(np.arange(256))[:, :3]
    j_et_heatmap = j_et_colors[heatmap]
    j_et_heatmap = tf.keras.utils.array_to_img(j_et_heatmap)
    j_et_heatmap = j_et_heatmap.resize((img_array[0].shape[1], img_array[0].shape[0]))
    j_et_heatmap = tf.keras.utils.img_to_array(j_et_heatmap)
    s_img = j_et_heatmap * 0.4 + img_array[0] * 255
    s_img = tf.keras.utils.array_to_img(s_img)
    model.layers[-1].activation = last_layer_activation
    return s_img

# function to plote training history
def plot_history(history):
    # store results
    acuracy = history.history['accuracy']
    v_acuracy = history.history['val_accuracy']
    da_loss = history.history['loss']
    val_da_loss = history.history['val_loss']
    # loss
    plt.subplot(2, 1, 2)
    plt.plot(da_loss, label='Training Loss')
    plt.plot(val_da_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss. \nTrain Loss: {str(round(da_loss[-1],3))}\nValidation Loss: {str(round(val_da_loss[-1],3))}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)
    plt.show()
    # accuracy
    plt.figure(figsize=(5, 8))
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acuracy, label='Training Accuracy')
    plt.plot(v_acuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(f'\nTraining and Validation Accuracy. \nTrain Accuracy: {str(round(acuracy[-1],3))}\nValidation Accuracy: {str(round(v_acuracy[-1],3))}')

def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    model = Model(base_model.inputs, outputs)
    return model

def fit_model(model, base_model, epochs, fine_tune = 0):
    early = tf.keras.callbacks.EarlyStopping( patience = 10,
                                              min_delta = 0.001,
                                              restore_best_weights = True)

    print("Unfreezing layers in the base model = ", fine_tune)
    if fine_tune > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        base_model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(train_data_gen,
                        validation_data = valid_data_gen,
                        epochs = epochs,
                        callbacks = [early],
                        class_weight=cls_wg)

    return history

base_inceeption =tf.keras.applications.InceptionResNetV2(
                     include_top = False,
                     weights = 'imagenet',
                     input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
                     )
m_inception =create_model(base_inceeption)
history = fit_model(m_inception, base_inceeption, epochs = NUM_T_EPOCHS)

plot_history(history)

# Unfreeze
layers=len(base_inceeption.layers)
print("Inception base layers = ", layers)
history = fit_model(m_inception, base_inceeption, epochs = NUM_FT_EPOCHS, fine_tune = int(layers/4))

plot_history(history)

auc_score=m_inception.evaluate(test_data_gen)

print(auc_score)
print("Accuracy: {:.2f}%".format(auc_score[1] * 100))
print("Loss: {:.3f}".format(auc_score[0]))

test_data_gen.reset()
inception_predictions = m_inception.predict(test_data_gen)
incept_prediction_classes = np.argmax(inception_predictions, axis=1)

fpr_inception, tpr_inception, thresholds_inception = roc_curve(test_data_gen.classes, inception_predictions[:,1])
auc_inception = auc(fpr_inception, tpr_inception)

test_data_gen.reset()
x, y = next(test_data_gen)
draw_img(x, y, inception_predictions)

last_conv_layer_name = "conv_7b_ac"
heatmap_list =[]
for img in x:
    heatmap = grad_hm(img, m_inception, last_conv_layer_name)
    heatmap_list.append(heatmap)

draw_img(heatmap_list, y, inception_predictions)

# load the VGG16 architecture with imagenet weights as base
base_vgg16 = tf.keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
m_vgg16=create_model(base_vgg16)
history = fit_model(m_vgg16, base_vgg16, epochs = NUM_T_EPOCHS)

plot_history(history)

# Unfreeze
layers = len(base_vgg16.layers)
print("VGG16 base layers = ", layers)

history = fit_model(m_vgg16, base_vgg16, epochs = NUM_FT_EPOCHS, fine_tune = int(layers/4))

plot_history(history)

auc_score = m_vgg16.evaluate(test_data_gen)
print(auc_score)
print("Accuracy: {:.2f}%".format(auc_score[1] * 100))
print("Loss: {:.3f}".format(auc_score[0]))

# computation for confusion matrix
test_data_gen.reset()

vgg16_test_preds = m_vgg16.predict(test_data_gen)
vgg16_test_pred_classes = np.argmax(vgg16_test_preds, axis=1)

fpr_vgg16, tpr_vgg16, thresholds_vgg16 = roc_curve(test_data_gen.classes, vgg16_test_preds[:,1])
auc_vgg16 = auc(fpr_vgg16, tpr_vgg16)

test_data_gen.reset()
x,y = next(test_data_gen)
draw_img(x, y, vgg16_test_preds)

last_conv_layer_name = "block5_conv3"

heatmap_list =[]

for img in x:
    heatmap = grad_hm(img, m_vgg16, last_conv_layer_name)
    heatmap_list.append(heatmap)

draw_img(heatmap_list, y, vgg16_test_preds)

#xception
xception_base_model = tf.keras.applications.xception.Xception(
                                 include_top = False,
                                 weights = 'imagenet',
                                 input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
                                 )

xception_model = create_model(xception_base_model)

history = fit_model(xception_model, xception_base_model, epochs = NUM_T_EPOCHS)

plot_history(history)

# Unfreeze
layers = len(xception_base_model.layers)
print("xception base layers = ", layers)

history = fit_model(xception_model, xception_base_model, epochs = NUM_FT_EPOCHS, fine_tune = int(layers/4))

plot_history(history)

auc_score = xception_model.evaluate(test_data_gen)

print(auc_score)
print("Accuracy: {:.2f}%".format(auc_score[1] * 100))
print("Loss: {:.3f}".format(auc_score[0]))

# computation for the confusion matrix
test_data_gen.reset()

xception_test_preds = xception_model.predict(test_data_gen)
xception_test_pred_classes = np.argmax(xception_test_preds, axis = 1)

fpr_xception, tpr_xception, thresholds_xception = roc_curve(test_data_gen.classes, xception_test_preds[:,1])
auc_xception = auc(fpr_xception, tpr_xception)

test_data_gen.reset()
x, y = next(test_data_gen)
draw_img(x, y, xception_test_preds)

last_conv_layer_name = "block14_sepconv2_act"

heatmap_list = []

for img in x:
    heatmap = grad_hm(img, xception_model, last_conv_layer_name)
    heatmap_list.append(heatmap)

draw_img(heatmap_list, y, xception_test_preds)

#densenet

from tensorflow.keras.applications import DenseNet121

densenet_base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

densenet_model = create_model(densenet_base_model)
history = fit_model(densenet_model, densenet_base_model, epochs=NUM_T_EPOCHS)

layers = len(densenet_base_model.layers)
history = fit_model(
    densenet_model,
    densenet_base_model,
    epochs=NUM_FT_EPOCHS,
    fine_tune=int(layers / 4)
)

auc_score = densenet_model.evaluate(test_data_gen)
print(auc_score)
print("Accuracy: {:.2f}%".format(auc_score[1] * 100))
print("Loss: {:.3f}".format(auc_score[0]))

test_data_gen.reset()

densenet_test_preds = densenet_model.predict(test_data_gen)
densenet_test_pred_classes = np.argmax(densenet_test_preds, axis = 1)

fpr_densenet, tpr_densenet, thresholds_densenet = roc_curve(test_data_gen.classes, densenet_test_preds[:,1])
auc_densenet = auc(fpr_densenet, tpr_densenet)

test_data_gen.reset()
x, y = next(test_data_gen)
draw_img(x, y, densenet_test_preds)

last_conv_layer_name = "pool2_conv"

heatmap_list = []

for img in x:
    heatmap = grad_hm(img, densenet_model, last_conv_layer_name)
    heatmap_list.append(heatmap)

draw_img(heatmap_list, y, densenet_test_preds)

# Display confusion matrix
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 8))

true_classes = test_data_gen.classes

spot_hm(true_classes, incept_prediction_classes, class_names, ax1, title = "Inception")
spot_hm(true_classes, vgg16_test_pred_classes, class_names, ax2, title = "VGG16")
spot_hm(true_classes, xception_test_pred_classes, class_names, ax3, title = "Xception")
spot_hm(true_classes, densenet_test_pred_classes, class_names, ax4, title = "Densenet")

fig.suptitle("Confusion Matrix Model Comparison", fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top=1.25)
plt.show()

#ensemble
ensemble_preds = np.add(np.add(inception_predictions, vgg16_test_preds),densenet_test_preds, xception_test_preds)/4.0

ensemble_test_pred_classes = np.argmax(ensemble_preds, axis = 1)

test_data_gen.reset()
x, y = next(test_data_gen)
draw_img(x, y, ensemble_preds)

ensemble_accuracy = np.mean(ensemble_test_pred_classes == test_data_gen.classes)
print("Ensemble Accuracy: {:.2f}%".format(ensemble_accuracy * 100))

#  confusion matrix
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 8))

true_classes = test_data_gen.classes

spot_hm(true_classes, ensemble_test_pred_classes, class_names, ax1, title = "Ensemble")

fig.suptitle("Confusion Matrix Model", fontsize = 15)
fig.tight_layout()
plt.show()

fpr_ensemble, tpr_ensemble, thresholds_ensemble = roc_curve(test_data_gen.classes, ensemble_preds[:,1])
auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_inception, tpr_inception, label='ResNet (area = {:.3f})'.format(auc_inception))
plt.plot(fpr_vgg16, tpr_vgg16, label='VGG16 (area = {:.3f})'.format(auc_vgg16))
plt.plot(fpr_xception, tpr_xception, label='Xception (area = {:.3f})'.format(auc_xception))
plt.plot(fpr_densenet, tpr_densenet, label='Densenet (area = {:.3f})'.format(auc_densenet))
plt.plot(fpr_ensemble, tpr_ensemble, label='Ensemble (area = {:.3f})'.format(auc_ensemble))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

