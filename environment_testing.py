import tensorflow as tf
import numpy as np
import torch

print(np.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
print("CUDA available:", torch.cuda.is_available())