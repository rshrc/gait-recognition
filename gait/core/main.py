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

import shutil
import os
import numpy as np


# Copy-pasta function to split dataset for training and validation
def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")

def create_model():
    model = Sequential()
    model.add(Convolution2D(4, (3, 3), input_shape=(320,240, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=560, activation='relu'))
    model.add(Dense(output_dim=1, activation='softmax'))
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

def dataset_provider(datagen,dirpath):
    return datagen.flow_from_directory(
        dirpath,
        target_size=(320,240),
        batch_size=32,
        class_mode='binary'
    )

# Primary datagen
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2)
# Validation datagen
test_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2)

# Primary Set for training
training_set = dataset_provider(train_datagen, 'imagesrc-tmp/train')
# Secondary / Test set for validation
test_set = dataset_provider(test_datagen,'imagesrc-tmp/test')

model = create_model()
train(model, training_set, test_set)
save_model(model)