# Fruit Classification Using Transfer Learning

This project demonstrates how to classify fruits using a deep learning model based on transfer learning. The model leverages the pre-trained VGG16 architecture and fine-tunes it for the task of fruit classification.

## Features

- **Dataset Handling**: Automatically downloads and extracts the Fruits-360 dataset.
- **Data Augmentation**: Applies various augmentation techniques to improve model generalization.
- **Transfer Learning**: Uses the VGG16 model pre-trained on ImageNet as the base model.
- **Fine-Tuning**: Unfreezes specific layers of the base model for fine-tuning.
- **Visualization**: Includes accuracy and loss curves, as well as sample predictions with actual labels.

## Directory Structure

Ensure your dataset is organized as follows:

```
dataset/
├── train/
│   ├── Class1/
│   ├── Class2/
│   ├── Class3/
│   └── (other classes...)
├── val/
│   ├── Class1/
│   ├── Class2/
│   ├── Class3/
│   └── (other classes...)
└── test/
    ├── Class1/
    ├── Class2/
    ├── Class3/
    └── (other classes...)
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Google Colab (optional for running the notebook)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fruit-classification-transfer-learning.git
   cd fruit-classification-transfer-learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Fruit_Classification_Using_Transfer_Learning.ipynb
   ```

4. Follow the instructions in the notebook to train and evaluate the model.

## Key Steps in the Notebook

1. **Import Libraries**: Load all necessary libraries for data processing, model building, and visualization.
2. **Download Dataset**: Automatically download and extract the Fruits-360 dataset.
3. **Data Generators**: Set up training, validation, and testing data generators with augmentation.
4. **Model Building**: Define the VGG16-based model architecture with custom layers.
5. **Training**: Train the model with early stopping and learning rate scheduling.
6. **Fine-Tuning**: Unfreeze specific layers of the base model for fine-tuning.
7. **Evaluation**: Evaluate the model on the test set and visualize predictions.

## Results

- **Accuracy**: Achieved high accuracy on the test set.
- **Visualization**: Plotted training and validation accuracy/loss curves.
- **Predictions**: Displayed sample predictions with actual labels.

## Colab Integration

If running on Google Colab, the notebook includes a script to download the dataset and save the trained model or results to your local machine.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits)
- TensorFlow and Keras for providing the tools for deep learning.
