import tifffile
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import os
import random
import pickle
from multilayer_perceptron import *
from hypercolumn import *

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def one_hot_encode(labels, nclasses):
    l_shape = list(labels.shape)
    newshape = np.concatenate((l_shape, [nclasses]))
    onehot = np.zeros(newshape).astype(np.int8)
    for coordinates in cartesian([np.array(range(shape)) for shape in l_shape]):
        one_val = np.concatenate((coordinates, [int(labels[tuple(coordinates)])])) 
        onehot[tuple(one_val)] = 1
    return onehot

def one_hot_decode(label):
    l_shape = list(label.shape[:-1])
    decoded = np.zeros(l_shape).astype(np.int8)
    for coordinates in cartesian([np.array(range(dim)) for dim in l_shape]):
        decoded[tuple(coordinates)] = np.argmax(label[tuple(coordinates)])
    return np.squeeze(decoded)

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

"""
Criando o Modelo
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
    x = X_uhcs_data,
    y = y_uhcs_data,
    batch_size = batchsize,
    validation_split = 0.1,
    epochs = 100,
    verbose = 1,
    callbacks = [checkpoint, early_stopping, learning_rate_scheduler]
)

pixelnet_model.save(path, save_format='tf')
with open("X_uhcs_data.pickle","wb") as f:
    pickle.dump(X_uhcs_data, f)
with open("y_uhcs_data.pickle","wb") as f:
    pickle.dump(y_uhcs_data, f)

pixelnet_model.evaluate(X_uhcs_data, y_uhcs_data)