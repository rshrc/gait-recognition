from keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
import simplejson as sj

def create_model():
    model = Sequential()
    model.add(Convolution2D(4, (3, 3), input_shape=(240, 320, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=1, activation='sigmoid'))
    return model

def train(model, training_set, test_set):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(
            training_set,
            steps_per_epoch=250,
            epochs=25,
            verbose=1,
            validation_data=test_set,
            validation_steps=62.5
    )

def save_model(model):
    print("Saving...")
    model.save_weights("model.h5")
    print(" [*] Weights")
    open("model.json", "w").write(
            sj.dumps(sj.loads(model.to_json()), indent=4)
    )
    print(" [*] Model")

def load_model():
    print("Loading...")
    json_file = open("model.json", "r")
    model = model_from_json(json_file.read())
    print(" [*] Model")
    model.load_weights("model.h5")
    print(" [*] Weights")
    json_file.close()
    return model

def dataset_provider(datagen):
    return datagen.flow_from_directory(
        'imagesrc',
        target_size=(240, 320),
        batch_size=32,
        class_mode='binary'
    )

# Primary datagen
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# Validation datagen
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Primary Set for training
training_set = dataset_provider(train_datagen)
# Secondary / Test set for validation
test_set = dataset_provider(test_datagen)

model = create_model()
train(model, training_set, test_set)
save_model(model)