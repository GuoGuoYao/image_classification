# -*- coding: utf-8 -*-

from dataGenerator import DataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications import VGG16
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 多块显卡的话,指定使用第几块显卡
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Activation,GlobalAveragePooling2D
import argparse


classnames = ['OCEAN', 'MOUNTAIN', 'LAKE', 'FARMLAND', 'DESERT', 'CITY']
NUM_CLASSES = len(classnames)
EPOCHES = 40
BATCH_SIZE = 32
PATCH_SIZE = (256, 256)
LR = 3e-4  # learning rate，希望它在模型前期大，以快速找到收敛域，后期小得以找到最优解
decay = 0.1
ratio = 0.6

def Network():  # VGG16
    inputs = Input(shape=(PATCH_SIZE[0], PATCH_SIZE[1], 3))  # 大小是256*256

    # 一般这个叫做一个block 因为参数都是32
    conv1_1 = Conv2D(64, (3, 3), padding="same", strides=(1, 1), use_bias=False)(inputs)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv1_1)
    conv1_2 = Activation('relu')(conv1_2)

    maxpool_1 = MaxPool2D(pool_size=(2, 2))(conv1_2)  # 第一次pool 大小变成128*128

    conv2_1 = Conv2D(128, (3, 3), padding="same", strides=(1, 1), use_bias=False)(maxpool_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv2_1)
    conv2_2 = Activation('relu')(conv2_2)

    maxpool_2 = MaxPool2D(pool_size=(2, 2))(conv2_2)  # 第二次pool 大小变成64*64

    conv3_1 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), use_bias=False)(maxpool_2)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv3_1)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv3_2)
    conv3_3 = Activation('relu')(conv3_3)

    maxpool_3 = MaxPool2D(pool_size=(2, 2))(conv3_3)  # 第三次pool 大小变成32*32

    conv4_1 = Conv2D(512, (3, 3), padding="same", strides=(1, 1), use_bias=False)(maxpool_3)
    conv4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv4_1)
    conv4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), padding="same", strides=(1, 1), use_bias=False)(conv4_2)
    conv4_3 = Activation('relu')(conv4_3)

    maxpool_4 = MaxPool2D(pool_size=(2, 2))(conv4_3)  # 第四次pool 大小变成16*16

    flatten = Flatten()(maxpool_4)

    dense_1 = Dense(200, activation='relu')(flatten)  # 也可以把激活函数拆出来写成上面的那种 Activation('relu') 的格式
    dense_2 = Dense(20, activation='relu')(dense_1)
    outputs = Dense(NUM_CLASSES, activation='softmax')(dense_2)  #　最后一定是和类别数相同的

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def fineturn_vgg():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    dense_1 = Dense(200, activation='softmax')(x)
    dense_2 = Dense(NUM_CLASSES, activation='softmax')(dense_1)
    model = Model(inputs=base_model.input, outputs=dense_2)
    model.summary()
    return model


def model_train(args):
    print("enter train process")
    path_workplace = os.getcwd()  # 获得当前工作目录
    train_dir = os.path.join(path_workplace,'train')
    gen = DataGenerator(PATCH_SIZE, BATCH_SIZE, classnames, train_dir,ratio)  # 每一批数据打包成一个yield放到显存中，

    def schedule(EPOCHES, decay=0.9):
        return LR * decay ** (EPOCHES)

    model_path = "./model_weight/weight-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpointer = keras.callbacks.ModelCheckpoint(model_path, verbose=1, monitor='val_acc', save_best_only=True)
    LRScheduler= keras.callbacks.LearningRateScheduler(schedule)
    tensor = keras.callbacks.TensorBoard(log_dir='./log',
                                   histogram_freq=0, write_graph=True, write_images=False,
                                   embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    if args.model == 'Network':
        model = Network()
    elif args.model == 'fineturn_vgg':
        model = fineturn_vgg()

    optim = keras.optimizers.Adam(lr=LR)
    # optim = keras.optimizers.RMSprop(lr=LR)
    # optim = keras.optimizers.SGD(lr=LR, momentum=0.9, decay= LR / EPOCHES, nesterov=False)


    # model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])        #### 二分类
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.fit_generator(gen.generator(True),
                        steps_per_epoch=gen.train_batches,
                        epochs=EPOCHES, verbose=1,
                        validation_data=gen.generator(False),
                        validation_steps=gen.valid_batches,
                        callbacks=[checkpointer, LRScheduler, tensor])
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default = 'Network',help="select a model")
    args = parser.parse_args()
    return args


# 主函数 只会在 python 文件名.py 时被调用 被import时不会执行
if __name__ == '__main__':
    import time
    starttime = time.time()
    args = parse_args()
    model_train(args)
    endtime = time.time()
    print("runtime:  ", (endtime - starttime))