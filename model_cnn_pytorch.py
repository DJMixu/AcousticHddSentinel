"""
Alias modułu architektury modelu 2D CNN w PyTorch (AcousticHddSentinelCNN).
Główna definicja znajduje się w model_cnn.py.
"""

from model_cnn import (
    INPUT_SHAPE,
    INPUT_SHAPE_PT,
    DEFAULT_NUM_CLASSES,
    AcousticHddSentinelCNN,
    build_hdd_cnn_model,
    build_transfer_learning_model,
)

# Alias zgodnościowy dla wcześniejszych wywołań
build_hdd_cnn_model_pt = build_hdd_cnn_model
build_transfer_learning_model_pt = build_transfer_learning_model
