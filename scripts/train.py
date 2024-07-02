from keras.datasets import cifar10
from models.model import build_model

# load the cifar10 dataset
(X_train,Y_train),(X_test,Y_test)  = cifar10.load_data()

# This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.
X_train = X_train.reshape(50000,32,32,3)
X_test = X_test.reshape(10000,32,32,3)

# Normalize the pixel values from 0 to 1
X_train = X_train/255
X_test  = X_test/255

# Cast to float
X_train = X_train.astype('float')
X_test  = X_test.astype('float')

# Build and train model
model = build_model()
model.fit(X_train,Y_train,batch_size=32, epochs=10, validation_split=0.1, shuffle=True)

# Save model
model.save('models/cifar10_model.h5')