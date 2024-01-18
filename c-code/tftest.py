import tensorflow as tf

from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import numpy as np
from sklearn.model_selection import train_test_split

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

x_data = []
y_label = []



if __name__ == "__main__":
    np.random.seed(0)
    ## PART I: DATA COLLECTION
    ## reactant1

            

                

    ## PART II: PREPROCESS FEATURES AND LABELS
    ## CHANGE TO ARRAY FORMAT FOR X
    x = np.asarray(x_data)
    
    ## ONE-HOT ENCODING Y
    y = to_categorical(y_label)
    
    ## SPLIT DATA
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
    #%%
    ## CALL 3D-CNN
    model = cnn()
    
    ## COMPILE CNN AND TRAIN
    '''
    Loss function is categorical_crossentropy
    Optimizer is adam with lr = 0.00001
    Metrics is accuracy
    A check point is set as finding the weights with max validation accuracy and store it
    A check point can also be min validation loss
    '''
    trained = 'no' # or 'yes'
    ## CHANGE TO YOUR DIRECTORY TO STORE WEIGHTS
    
    if trained == 'no':
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.00001), metrics=['acc'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(x=x_train, y=y_train, batch_size=18, epochs=500, validation_split=0.2, callbacks=callbacks_list)
    
    ## TEST CNN
    model.load_weights(filepath)
    
    # COMPILE MODEL AGAIN WITH STORED WEIGHTS
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.00001), metrics=['acc'])
    model.summary()
    print("Created model and loaded weights from file")
    
    
    ## ACCURACY METRICS
    pred = model.predict(x_test)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
        
    accuracy_score(y_true, y_pred)