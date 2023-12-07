import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
# Function to rename files in a directory
def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg.chip.jpg"):
            # Create the new filename by replacing '.jpg.chip.jpg' with '.jpg'
            new_filename = filename.replace('.jpg.chip.jpg', '.jpg')

            # Construct the full file paths
            current_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(current_filepath, new_filepath)
            #print(f"Renamed {filename} to {new_filename}")

# Function to extract information from image file names
def extract_info_from_filename(filename):
    # Split the filename by underscores
    parts = filename.split('_')
    # Extract relevant information
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])
    return age, gender, race

def can_extract_info(filename):
    parts = filename.split('_')
    if (len(parts)<4):
        print(filename + " has missing values.")
        return False
    else:
        return True

# Function to get image file names in a folder
def get_image_info(folder_path):
    image_info = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Consider only JPG files
            if can_extract_info(filename):
                age, gender, race = extract_info_from_filename(filename)
                image_id = filename  # Remove file extension to get image ID
                image_info.append((image_id, age, gender, race))

    return image_info

# Create a DataFrame from image information
def create_dataframe(folder_path):
    image_info = get_image_info(folder_path)
    df = pd.DataFrame(image_info, columns=['Image_ID', 'Age', 'Gender', 'Race'])
    return df

# Function to split data into training and validation sets
def split_data(data, test_size=0.1, random_state=42):
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=random_state)
    #train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data


# Function to create ImageDataGenerator with specified augmentations
def create_data_generator():
    return (ImageDataGenerator(
        rescale=1. / 255,),
            ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ))


# Function to preprocess data
def preprocess_data(data_path, output_path, image_size=(128, 128), batch_size=32):
    df = pd.read_csv('../data/UTKFace_labels_for_trying.csv')
    train_data, val_data = split_data(df)

    datagen, augmented_datagen = create_data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=data_path,
        x_col='Image_ID',
        y_col=['Age', 'Gender', 'Race'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode='raw',
    )

    train_gen_augmented = augmented_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=data_path,
        x_col='Image_ID',
        y_col=['Age', 'Gender', 'Race'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode='raw',
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
        class_mode='raw',
        save_to_dir=output_path,
        save_prefix='aug',
        save_format='jpg'
    )

    return train_gen, train_gen_augmented


folder_path = '../data/for_trying'

rename_files(folder_path)

# Define the directory to save the DataFrame
output_directory = '../data/'

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the output file path
output_file_path = os.path.join(output_directory, 'UTKFace_labels_for_trying.csv')

# Save the DataFrame to a CSV file in the new directory
image_dataframe = create_dataframe(folder_path)
image_dataframe.to_csv(output_file_path, index=False)

# Example usage
data_path = "../data/for_trying"
output_path = "../data/augmented_data"

# Create the directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

image_size = (128, 128)
batch_size = 32

train_generator, train_aug = preprocess_data(data_path, output_path, image_size, batch_size)
