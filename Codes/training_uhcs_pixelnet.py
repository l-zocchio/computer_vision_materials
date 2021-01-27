import numpy as np
import tensorflow as tf
import tifffile
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import os
import random
import pickle
from DataLoad.data_utils import *
from Model.Hypercolumn import *
from Model.MultilayerPerceptron import *


# Selecionando a semente para gerar números pseudo-aleatórios
random.seed(10)
np.random.seed(10)

"""
Carregando imagens de UltraHigh Carbon Steel Database
"""
images = []
labels = []
images_dir = "uhcs/images"
labels_dir = "uhcs/labels"
for filename in os.listdir(images_dir):
    img = tifffile.imread(images_dir+"/"+filename)
    img = img[:484]
    img = np.stack((img,)*3, axis=-1)
    images.append(img)
X_uhcs_data = np.array(images)
images_tensor = tf.constant(images)
for filename in os.listdir(labels_dir):
    lbl = tifffile.imread(labels_dir+"/"+filename)
    lbl_zeros = np.zeros(lbl.shape)
    lbl = np.fmax(lbl_zeros,lbl)
    lbl = lbl[:484]
    lbl = one_hot_encode(lbl, 4)
    labels.append(lbl)
y_uhcs_data = np.array(labels)
labels_tensor = tf.constant(labels)
uhcs_data = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))

# Dividindo os dados em teste e treino
X_uhcs_train, X_uhcs_test, y_uhcs_train, y_uhcs_test = train_test_split(X_uhcs_data, y_uhcs_data, test_size=0.2)

#carregando imagens de Particles Database
# images = []
# labels = []
# images_dir = "particles/images"
# labels_dir = "particles/labels"
# for filename in os.listdir(images_dir):
#     img = tifffile.imread(images_dir+"/"+filename)
#     img = img[:484]
#     img = np.stack((img,)*3, axis=-1)
#     images.append(img)
# X_particle_data = np.array(images)
# images_tensor = tf.constant(images)
# for filename in os.listdir(labels_dir):
#     lbl = tifffile.imread(labels_dir+"/"+filename)
#     lbl_zeros = np.zeros(lbl.shape)
#     lbl = np.fmax(lbl_zeros,lbl)
#     lbl = lbl[:484]
#     lbl = one_hot_encode(lbl, 4)
#     labels.append(lbl)
# y_particle_data = np.array(labels)
# labels_tensor = tf.constant(labels)
# particle_data = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))

"""
Creating the Model
"""
base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(484, 645, 3))
base_model.trainable = False
inputlayers = [
               'block1_conv2',
               'block2_conv2',
               'block3_conv3',
               'block4_conv3',
               'block5_conv3'
]
hc_model = build_hc_model(base_model, inputlayers)
pixelnet_model = build_model(hc_model, mc_dropout=True)

"""
Training the Model
"""
batchsize=1
opt = Adam()
loss_fn = CategoricalCrossentropy()
my_metrics = [
           CategoricalAccuracy()
]
pixelnet_model.compile(optimizer = opt, loss = loss_fn, metrics = my_metrics)

path = ".\\saved_model"
if not os.path.isdir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("\nCreation of the directory %s failed \n" % path)
    else:
        print ("\nSuccessfully created the directory %s \n" % path)
else:
    print("\nDirectory %s already exists" % path)

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

learning_rate_scheduler = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint('.\\tf_ckpts', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

uchs_history = pixelnet_model.fit(
    x = X_uhcs_train,
    y = y_uhcs_train,
    batch_size = batchsize,
    validation_split = 0.1,
    epochs = 100,
    verbose = 1,
    callbacks = [checkpoint, early_stopping, learning_rate_scheduler]
)

pixelnet_model.save(path, save_format='tf')
with open("X_uhcs_train.pickle","wb") as f:
    pickle.dump(X_uhcs_train, f)
with open("y_uhcs_train.pickle","wb") as f:
    pickle.dump(y_uhcs_train, f)
with open("X_uhcs_test.pickle","wb") as f:
    pickle.dump(X_uhcs_test, f)
with open("X_uhcs_test.pickle","wb") as f:
    pickle.dump(X_uhcs_test, f)

pixelnet_model.evaluate(X_uhcs_test, y_uhcs_test)