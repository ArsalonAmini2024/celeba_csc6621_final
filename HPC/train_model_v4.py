#### Vision Transformer ######

# System Imports
import os
import shutil
import logging
from datetime import datetime

# Data Pre-processing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.applications import vit_b16  # Vision Transformer

# Configure logging with the unique log file
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.expanduser(f"~/Desktop/tensorboard_logs/vision_transformer/{current_time}")
log_file = os.path.expanduser(f"~/Desktop/vision_transformer_{current_time}.log")  # Updated logger file name

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LoggingCallback(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy', 'N/A')
        val_acc = logs.get('val_accuracy', 'N/A')
        self.logger.info(f"Epoch {epoch + 1}: Training Accuracy = {train_acc}, Validation Accuracy = {val_acc}")

# Tensorboard logs
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

logger = logging.getLogger()
logging_callback = LoggingCallback(logger)

##### LOAD AND PREPARE DATA #####

df = pd.read_csv('~/Desktop/identity_CelebA.txt', delim_whitespace=True, header=None, names=['filename', 'label'])
logger.info("Loaded identity data")

label_counts = df['label'].value_counts()
df_filtered = df[df['label'].map(label_counts) > 25]
train_df, test_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['label'])
logger.info(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
num_classes = train_df['label'].nunique()
logger.info(f"Unique classes in training set: {num_classes}")
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
logger.info(f"Unique classes in training set: {train_df['label'].nunique()}")
logger.info(f"Unique classes in testing set: {test_df['label'].nunique()}")

# Copy the images to the respective folders based on the train-test split of the labels
def copy_images(df, source_dir, target_dir):
    # Expand source and target directories properly
    source_dir = os.path.expanduser(source_dir)
    target_dir = os.path.expanduser(target_dir)

    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    for label in df['label'].unique():
        class_dir = os.path.join(target_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # Filter images for this label
        images = df[df['label'] == label]['filename']
        for filename in images:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(class_dir, filename)

            # Check if the file already exists in the target directory
            if not os.path.exists(target_path):
                shutil.copy(source_path, target_path)

# Move Images to respective directories
source_directory = os.path.expanduser('~/Desktop/img_align_celeba')
train_directory = os.path.expanduser('~/Desktop/train')
test_directory = os.path.expanduser('~/Desktop/test')

# Move train and test images
copy_images(train_df, source_directory, train_directory)
copy_images(test_df, source_directory, test_directory)
logger.info("Completed copying training and testing images")

# Log the number of files in training and testing directories
train_files = sum([len(files) for r, d, files in os.walk(train_directory)])
test_files = sum([len(files) for r, d, files in os.walk(test_directory)])
logger.info(f"Number of train images: {train_files}")
logger.info(f"Number of test images: {test_files}")

# Load data using ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

logger.info("Image data generators initialized")

# Define the Vision Transformer model
base_model = vit_b16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
logger.info("Model architecture summarized")

history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[logging_callback, tensorboard_callback])
logger.info("Model training completed")

test_loss, test_accuracy = model.evaluate(test_generator)
logger.info(f"Test accuracy: {test_accuracy}")
print("Test accuracy:", test_accuracy)
