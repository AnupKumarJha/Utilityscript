import keras
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split
batch_size=32
train_generator = train_datagen.flow_from_directory(
    "/content/Train", #training directory
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') 
# set as training data

validation_generator = train_datagen.flow_from_directory(
    "/content/Train", # same directory as training data
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') 
# set as validation data
nb_epochs=15
model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs)
