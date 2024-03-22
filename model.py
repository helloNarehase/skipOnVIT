import os
import random
import sys

import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import keras.ops as ops
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

image_size = 256

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:

        tf.config.experimental.set_memory_growth(gpus[0], True)

        # tf.config.experimental.set_virtual_device_configuration(gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*11.2)])

    except RuntimeError as e:
        print(e)
# num_patches = (image_size // patch_size) ** 2

import dpmodel
model = dpmodel.end2end_vit()
model.summary()
def calculate_loss(target, pred):
    
    ssim_loss_weight = 0.85
    l1_loss_weight = 0.1
    edge_loss_weight = 0.9

    # Edges 
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))
    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1
        - tf.image.ssim(
            target, pred, max_val=image_size, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )
    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    loss = (
        (ssim_loss_weight * ssim_loss)
        + (l1_loss_weight * l1_loss)
        + (edge_loss_weight * depth_smoothness_loss)
    )

    return loss

import cv2
from pathlib import Path
import multiprocessing

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

master = "nyu_data/data/nyu2_train/"

tL = "nyu_data/data/nyu2_train.csv"
trainList = pd.read_csv(tL, names=["img", "depth"])
imgs = trainList["img"].tolist()
depths = trainList["depth"].tolist()

img_depth = list(zip(imgs, depths))

paths = [str(x) for x in Path(master).glob("*/*.jpg")]
print(len(paths))
shape = (image_size, image_size)

def hef(path:str):
    img = cv2.imread(path)
    img = img[10:, 10:, :]
    img = img[:-10, :-10, :]
    img = cv2.resize(img, shape)

    depth = cv2.imread(path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
    depth = depth[10:, 10:]
    depth = depth[:-10, :-10]
    depth = cv2.resize(depth, shape)
    # depth = cv2.resize(depth, (shape[0]//2, shape[0]//2))
    # print(depth.max(), depth.min())
    # print(depth.shape)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    img = (img - img.min()) / (img.max() - img.min())

    return {"image" : img, "depth" : depth}

def loaders(fileNames = paths):

    pol = multiprocessing.Pool(14)
    a = pol.map(hef, fileNames)
    # print(a)
    return a


random.shuffle(paths)
# dataset = loaders(paths)
from sklearn.model_selection import train_test_split
tr, va = train_test_split(paths[:20], test_size=0.2)

dataset = loaders(tr)
print(len(dataset))
X_Train = [i["image"] for i in dataset]
Y_Train = [np.expand_dims(i["depth"], -1) for i in dataset]
dataset = None


dataset = loaders(va)
print(len(dataset))
X_v = [i["image"] for i in dataset]
Y_v = [np.expand_dims(i["depth"], -1) for i in dataset]
dataset = None

xs = X_Train[0]
# X_Train = tf.convert_to_tensor(X_Train)
# Y_Train = tf.convert_to_tensor(Y_Train)
ys = Y_Train[0]
path = "nyu_data/data/nyu2_train/nyu_office_1_out/1.jpg"

img = cv2.imread(path)
img = img[10:, 10:, :]
img = img[:-10, :-10, :]
img = cv2.resize(img, shape)
img = np.expand_dims(img, 0)
img = (img - img.min()) / (img.max() - img.min())

depthImg = cv2.imread(path.replace("jpg", "png"))
depthImg = depthImg[10:, 10:, :]
depthImg = depthImg[:-10, :-10, :]
depthImg = cv2.resize(depthImg, shape)
depthImg = (depthImg - depthImg.min()) / (depthImg.max() - depthImg.min())

cv2.imwrite("depth_live.png",  ops.convert_to_numpy(model(img)[0])*225.)

# exit()
learning_rate = 0.0001
weight_decay = 0.0001

optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
)
lossFn = keras.losses.BinaryCrossentropy()

save_model_path = "dpmodel.weights.h5"

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(save_model_path)
        return super().on_epoch_end(epoch, logs)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6) 
rLR = keras.callbacks.ReduceLROnPlateau (
    monitor='val_loss',
    factor=0.5,
    patience=24,
    min_lr=1e-6,
    verbose=1,
)
callbacks_list = [
    es,
    rLR,
    CustomCallback()
]

def binary_iou_loss(y_true, y_pred):
    bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
    intersection = ops.sum(y_true * y_pred)
    union = ops.sum(y_true) + ops.sum(y_pred) - intersection                        
    iou = (intersection + 1e-6) / (union + 1e-6)
                                
    loss = bce_loss - ops.log(iou)
                                    
    return loss      

def iou(y_true, y_pred):
    intersection = ops.sum(y_true * y_pred)
    union = ops.sum(y_true) + ops.sum(y_pred) - intersection                        
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou                     

import math


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(),
        # layers.RandomZoom(height_factor=(0.1, 0.2)),
        # layers.RandomRotation(factor=0.02),
    ],
    name="data_augmentation",
)

class CustomPyDataset(keras.utils.PyDataset):

    def __init__(self, x_set, y_set, batch_size, opens = True, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.opens = opens
        # Return number of batches.

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def aug(self, x, y):
        last = x.shape[-1] 
        data = np.concatenate([x, y], -1)
        data = data_augmentation(data)
        # print(data[:last].shape, data[last:].shape)
        return data[:, :, :, :last], data[:, :, :, last:]

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        if self.opens:
            return None
        # np.array([opentif(file_name) for file_name in batch_x]), np.array([opentif_mask(file_name) for file_name in batch_y])
        else:
            prox, proy = self.aug(np.array(batch_x), np.array(batch_y))
            
            return prox, proy
ds = CustomPyDataset(x_set=X_Train, y_set=Y_Train, batch_size=6, opens = False)
vs = CustomPyDataset(x_set=X_v, y_set=Y_v, batch_size=6, opens = False)
# vds = CustomPyDataset(x_set=vxs, y_set=vys, batch_size=6, opens = False)
model.load_weights(save_model_path)

# model.compile(optimizer = keras.optimizers.AdamW(0.001), loss = calculate_loss, metrics = [iou, keras.losses.binary_crossentropy])
# history = model.fit(ds, epochs = 100, callbacks = callbacks_list, validation_data = vs)



p = model.predict(ops.expand_dims(xs, 0))[0]
plt.imshow(p)
plt.savefig("pre.png")

plt.imshow(ys)
plt.savefig("y.png")

xs = cv2.imread("tt.jpg")
xs = cv2.resize(xs, (256,256))
xs = np.array(xs, np.float32)/255.


p = model.predict(ops.expand_dims(xs, 0))[0]



# fig = plt.figure(figsize=(15, 10))
# ax = plt.axes(projection="3d")

# STEP = 3
# for x in range(0, p.shape[0], STEP):
#     for y in range(0, p.shape[1], STEP):
#         ax.scatter(
#             [p[x, y]] * 3,
#             [y] * 3,
#             [x] * 3,
#             c=tuple(xs[x, y, :3] / 255),
#             s=3,
#         )
#     ax.view_init(45, 135)
# plt.show()

plt.imshow(p)
plt.savefig("tt.png")
