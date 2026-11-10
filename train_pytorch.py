"""
Alias potoku treningowego 2D CNN w PyTorch.
Główny potok treningowy znajduje się w train_cnn.py.
"""

from train_cnn import (
    run_training,
    train_one_epoch,
    evaluate_epoch,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves,
    main,
)

if __name__ == "__main__":
    main()
