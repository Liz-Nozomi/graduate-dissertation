import tensorflow as tf

from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def cnn():
#定义CNN。直接用就行
    ## INPUT LAYER
    input_layer = Input((20, 20, 20, 3))
    
    ## CONVOLUTIONAL LAYERS
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)
    
    ## MAXPOOLING LAYER
    pooling_layer1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)
    
    ## CONVOLUTIONAL LAYERS
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    
    ## MAXPOOLING LAYER
    pooling_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer4)
    
    ## BATCH NORMALIZATION ON THE CONVOLUTIONAL OUTPUTS BEFORE FULLY CONNECTED LAYERS
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)
    
    ## FULLY CONNECTED LAYERS/ DROPOUT TO PREVENT OVERFITTING
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(1)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(1)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    
    ## DEFINE MODEL WITH INPUT AND OUTPUT LAYERS
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

