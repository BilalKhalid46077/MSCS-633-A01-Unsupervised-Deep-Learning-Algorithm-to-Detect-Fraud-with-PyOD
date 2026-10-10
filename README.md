# Fraud Detection Using AutoEncoder

## Project Overview

This project uses an AutoEncoder neural network to identify potentially fraudulent credit card transactions. The model learns patterns from legitimate transactions and uses those patterns to identify transactions that look unusual. This is an example of anomaly detection using Artificial Intelligence and Machine Learning.

The project uses the anonymized credit card transaction dataset from Kaggle.

**Dataset:** https://www.kaggle.com/datasets/whenamancodes/fraud-detection

## Objectives

- Build an AutoEncoder-based fraud detection model using Python.
- Prepare and scale the transaction data.
- Train the model using legitimate transactions.
- Identify unusual transactions that may be fraudulent.
- Evaluate the model using standard classification metrics and graphs.

## Tools and Technologies

- **Programming language:** Python
- **Development environment:** Google Colab
- **Machine Learning library:** PyOD
- **Deep Learning library:** PyTorch
- **Data processing:** NumPy and pandas
- **Model evaluation:** scikit-learn
- **Visualization:** Matplotlib

## Dataset Information

The dataset contains 284,807 credit card transactions and 31 columns. It includes 30 input features and one target column named `Class`. In this dataset, `Class = 0` represents a legitimate transaction, while `Class = 1` represents a fraudulent transaction.

The dataset contains 492 fraudulent transactions, making fraud a very small part of the total data. This imbalance makes fraud detection challenging because a model can appear accurate while still missing fraudulent transactions.

## Methodology

The project follows these main steps:

1. Load and inspect the credit card transaction dataset.
2. Check for missing values and separate the input features from the target column.
3. Split the data into training and testing sets using stratified sampling.
4. Fit the feature scaler using legitimate training transactions only.
5. Train the AutoEncoder on legitimate training transactions.
6. Calculate anomaly scores for the test transactions.
7. Identify unusual transactions using the model's decision threshold.
8. Evaluate the results using a confusion matrix, precision, recall, F1-score, ROC-AUC, and average precision.
9. Save the evaluation results, predictions, and graphs.

The model uses a neural network architecture with the following layer sizes: `30, 16, 8, 16, 30`. The network compresses the input data and then attempts to reconstruct it. Transactions that are reconstructed with larger errors may be considered unusual.


## Results

The model was evaluated on 56,962 test transactions, including 98 fraudulent transactions.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.85% |
| Fraud Precision | 9.58% |
| Fraud Recall | 67.35% |
| Fraud F1-score | 16.77% |
| ROC-AUC | 95.83% |
| Average Precision | 19.00% |

The model identified 66 of the 98 fraudulent transactions and missed 32. It also flagged 623 legitimate transactions as potentially fraudulent. The results show that the model can identify many fraudulent transactions, but it also produces a high number of false alarms. Therefore, the results need improvement before a system like this could be used in a real financial environment.

## How to Run the Project

1. Download or clone this repository.
2. Download the dataset from Kaggle.
3. Place `creditcard.csv` in the same directory as `main.py`.
4. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Python program:

   ```bash
   python main.py
   ```

6. Check the `fraud_detection_outputs` folder for the results and graphs.

The dataset is not included in this repository. Users must download it separately from Kaggle.
