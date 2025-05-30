import os

DATA_PATH = 'data'
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')

BATCH_SIZE = 64
EPOCHS = 350
WARMUP_EPOCHS = 20
LEARNING_RATE = 1E-4

EPSILON = 1E-6
IMAGE_SIZE = (448, 448)

S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 2      # Number of classes in the dataset
