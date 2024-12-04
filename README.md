# Sentiment Analysis and Classification Using Apache Spark

## Introduction
This project leverages **Apache Spark** to perform sentiment analysis on the Amazon Reviews dataset. Using **natural language processing (NLP)** and **machine learning techniques**, it classifies reviews as positive or negative and compares the performance of different classification models.

---

## Project Structure
The repository includes the following components:
- **`main.py`**: Main script for data loading, preprocessing, and model training.
- **`README.md`**: Project documentation.
- **`data/`**: Directory containing training and testing datasets.
- **`output/`**: Directory for saving results and model outputs.

---

## Environment Requirements
To run the project, ensure the following environment setup:
- **Operating System**: Windows/Linux/MacOS
- **Python Version**: 3.8+
- **Spark Version**: 3.3.0+
- **Dependencies**: Listed in `requirements.txt`

## Dataset

This project uses the **Amazon Reviews dataset**, which includes the following features:
- **polarity**: Sentiment label (1 = Negative, 2 = Positive)
- **title**: Review title
- **text**: Review content

### Dataset Paths
- **Training data path**: `gs://your-bucket-name/train.csv`
- **Testing data path**: `gs://your-bucket-name/test.csv`

You can download the dataset from [Kaggle - Amazon Reviews Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews).

---

## Setup and Execution

### Steps to Run the Code

1. **Upload Dataset**  
   Upload the training and testing datasets to your cloud storage bucket (e.g., Google Cloud Storage).

2. **Modify the Code**  
   Open the script `main.py` and update the dataset paths to your bucket paths. Example:
   ```python
   train_path = "gs://your-bucket-name/train.csv"
   test_path = "gs://your-bucket-name/test.csv"

3. **Upload the Code**  
   Upload your modified script (main.py) into your bucket.

4.	**Create a Cluster**
   Set up a Spark cluster in your cloud environment (e.g., Google Cloud Dataproc, AWS EMR, or Alibaba E-MapReduce).

5.	**Submit the Job**
   Use the following command to submit the job to your cluster:spark-submit gs://your-bucket-name/main.py  

