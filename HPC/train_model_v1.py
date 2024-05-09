# System Imports
import os
import shutil
import logging

# Data pre-processing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential



# Set up logging
tensorboard_log_dir = os.path.expanduser('~/Desktop/tensorboard_logs')
tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, write_graph=True)


class LoggingCallback(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary with training/validation loss and metrics (e.g., accuracy)
        train_acc = logs.get('accuracy', 'N/A')
        val_acc = logs.get('val_accuracy', 'N/A')
        self.logger.info(f"Epoch {epoch + 1}: Training Accuracy = {train_acc}, Validation Accuracy = {val_acc}")

log_file = os.path.expanduser('~/Desktop/celeba_image_classification_model_v1.log')
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

logging_callback = LoggingCallback(logger)

##### LOAD AND PREPARE DATA #####

# Load the identity file into a DataFrame
df = pd.read_csv('~/CELEBA_CSC6621_FINAL/data/identity_CelebA.txt', delim_whitespace=True, header=None, names=['filename', 'label'])
logger.info("Loaded identity data")

# Count the occurrences of each label
label_counts = df['label'].value_counts()

# Remove rows where labels appear less than 25x
df_filtered = df[df['label'].map(label_counts) > 25]

# Split into training and testing sets
train_df, test_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['label'])
logger.info(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

# Convert label column to string - required for downstream data generators
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)

logger.info(f"Unique classes in training set: {train_df['label'].nunique()}")
logger.info(f"Unique classes in testing set: {test_df['label'].nunique()}")

# Copy the actual images to the respective folders based on the train-test split of the labels
def copy_images(df, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists
    for filename in df['filename']:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

# Set directory paths
source_directory = '~/img_align_celeba'
train_directory = '~/train'
test_directory = '~/test'

# Move train and test images
copy_images(train_df, source_directory, train_directory)
copy_images(test_df, source_directory, test_directory)
logger.info("Completed copying training and testing images")

# Load data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'path_to_test_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Define the model
num_classes = 10153
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=test_generator
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test accuracy:", test_accuracy)
