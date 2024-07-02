import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D

def build_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential()
    model.add(Conv2D(16,input_shape=input_shape,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(GlobalAveragePooling2D())
    # output class probabilities
    model.add(Dense(num_classes,activation='softmax'))

    # configure the training
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    return model