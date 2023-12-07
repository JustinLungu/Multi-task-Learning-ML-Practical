import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os


# Function to rename files in a directory
def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg.chip.jpg"):
            new_filename = filename.replace('.jpg.chip.jpg', '.jpg')
            current_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(current_filepath, new_filepath)


# Function to extract information from image file names
def extract_info_from_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])
    return age, gender, race


def can_extract_info(filename):
    parts = filename.split('_')
    return len(parts) >= 4


# Function to get image file names in a folder
def get_image_info(folder_path):
    image_info = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") and can_extract_info(filename):
            age, gender, race = extract_info_from_filename(filename)
            image_id = os.path.splitext(filename)[0]
            image_info.append((image_id, age, gender, race))

    return image_info


# Function to split data into training and validation sets
def split_data(data, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data


# Function to create ImageDataGenerator with specified augmentations
def create_data_generator():
    return ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


# Function to preprocess data
def preprocess_data(data_path, output_path, image_size=(128, 128), batch_size=32):
    rename_files(data_path)
    image_info = get_image_info(data_path)
    df = pd.DataFrame(image_info, columns=['Image_ID', 'Age', 'Gender', 'Race'])

    train_data, val_data = split_data(df)

    datagen = create_data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=data_path,
        x_col='Image_ID',
        y_col=['Age', 'Gender', 'Race'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode='other',
        save_to_dir=output_path,
        save_prefix='aug',
        save_format='jpg'
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_data,
        directory=data_path,
        x_col='Image_ID',
        y_col=['Age', 'Gender', 'Race'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode='other',
        save_to_dir=output_path,
        save_prefix='aug',
        save_format='jpg'
    )

    return train_gen, val_gen


# Example usage
data_path = "../data/UTKFace"
output_path = "../data/augmented_data"
image_size = (128, 128)
batch_size = 32

train_generator, val_generator = preprocess_data(data_path, output_path, image_size, batch_size)
