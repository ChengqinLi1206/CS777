from pyspark.sql import SparkSession
from pyspark.sql import Row
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, lit
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("AmazonReviewML").getOrCreate()

# Load dataset
data_train_path = "gs://metcs777_test/train.csv"
data_test_path = "gs://metcs777_test/test.csv"

data_train = spark.read.csv(data_train_path, header=False, inferSchema=True) \
                       .toDF("polarity", "title", "text")

data_test = spark.read.csv(data_test_path, header=False, inferSchema=True) \
                      .toDF("polarity", "title", "text")


# Append both DataFrames
combined_df = data_train.union(data_test)

# Drop duplicates
combined_df = combined_df.dropDuplicates()

# Show the structure of the DataFrame
combined_df.printSchema()

# Count different values in the polarity column
polarity_counts = combined_df.groupBy("polarity").count().orderBy("polarity")

# Convert to Pandas DataFrame for visualization
polarity_counts_pd = polarity_counts.toPandas()

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(polarity_counts_pd['polarity'], polarity_counts_pd['count'], color=['red' if x < 1.5 else 'blue' for x in polarity_counts_pd['polarity']], edgecolor='black')

# Add labels and title
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Breakdown of Polarity in Amazon Reviews')

# Add count on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# Show the plot
plt.ylim(0, max(polarity_counts_pd['count']) * 1.1)  # Set y-axis limit slightly above the max count
plt.show()

# Map polarity to binary labels
combined_df = combined_df.withColumn("label", when(col("polarity") == 1, 1).otherwise(0))

# Handle null values in the text column
combined_df = combined_df.fillna({"text": ""})

# Text feature extraction
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Split the data into training and testing sets
train_df, test_df = combined_df.randomSplit([0.8, 0.2], seed=42)

# Define models
logistic_regression = LogisticRegression(maxIter=10, regParam=0.01)
svm = LinearSVC(maxIter=10, regParam=0.01)
random_forest = RandomForestClassifier(numTrees=100, maxDepth=5, seed=42)

# Create pipelines
lr_pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, logistic_regression])
svm_pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, svm])
rf_pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, random_forest])

# Train models
lr_model = lr_pipeline.fit(train_df)
svm_model = svm_pipeline.fit(train_df)
rf_model = rf_pipeline.fit(train_df)

# Evaluate models
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Logistic Regression
lr_predictions = lr_model.transform(test_df)
lr_accuracy = evaluator.evaluate(lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

# SVM
svm_predictions = svm_model.transform(test_df)
svm_accuracy = evaluator.evaluate(svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Random Forest
rf_predictions = rf_model.transform(test_df)
rf_accuracy = evaluator.evaluate(rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Compare results
results = {
    "Model": ["Logistic Regression", "SVM", "Random Forest"],
    "Accuracy": [lr_accuracy, svm_accuracy, rf_accuracy]
}

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Define a function to compute TPR and TNR
def calculate_metrics(predictions):
    # True Positives
    TP = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    # True Negatives
    TN = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    # False Positives
    FP = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    # False Negatives
    FN = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

    # Calculate TPR and TNR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity, Recall
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity

    return {"TPR": TPR, "TNR": TNR, "TP": TP, "TN": TN, "FP": FP, "FN": FN}

# Logistic Regression Metrics
lr_metrics = calculate_metrics(lr_predictions)
print(f"Logistic Regression - TPR: {lr_metrics['TPR']:.2f}, TNR: {lr_metrics['TNR']:.2f}")

# SVM Metrics
svm_metrics = calculate_metrics(svm_predictions)
print(f"SVM - TPR: {svm_metrics['TPR']:.2f}, TNR: {svm_metrics['TNR']:.2f}")

# Random Forest Metrics
rf_metrics = calculate_metrics(rf_predictions)
print(f"Random Forest - TPR: {rf_metrics['TPR']:.2f}, TNR: {rf_metrics['TNR']:.2f}")

# Combine all results
metrics_results = {
    "Model": ["Logistic Regression", "SVM", "Random Forest"],
    "Accuracy": [lr_accuracy, svm_accuracy, rf_accuracy],
    "TPR": [lr_metrics["TPR"], svm_metrics["TPR"], rf_metrics["TPR"]],
    "TNR": [lr_metrics["TNR"], svm_metrics["TNR"], rf_metrics["TNR"]],
    "TP": [lr_metrics["TP"], svm_metrics["TP"], rf_metrics["TP"]],
    "TN": [lr_metrics["TN"], svm_metrics["TN"], rf_metrics["TN"]],
    "FP": [lr_metrics["FP"], svm_metrics["FP"], rf_metrics["FP"]],
    "FN": [lr_metrics["FN"], svm_metrics["FN"], rf_metrics["FN"]],
}

# Convert to a Pandas DataFrame for display
metrics_df = pd.DataFrame(metrics_results)
print(metrics_df)

# Combine the two results (results_df and metrics_df) into a single DataFrame
results_combined = {
    "Model": ["Logistic Regression", "SVM", "Random Forest"],
    "Accuracy": [lr_accuracy, svm_accuracy, rf_accuracy],
    "TPR": [lr_metrics["TPR"], svm_metrics["TPR"], rf_metrics["TPR"]],
    "TNR": [lr_metrics["TNR"], svm_metrics["TNR"], rf_metrics["TNR"]],
    "TP": [lr_metrics["TP"], svm_metrics["TP"], rf_metrics["TP"]],
    "TN": [lr_metrics["TN"], svm_metrics["TN"], rf_metrics["TN"]],
    "FP": [lr_metrics["FP"], svm_metrics["FP"], rf_metrics["FP"]],
    "FN": [lr_metrics["FN"], svm_metrics["FN"], rf_metrics["FN"]],
}

# Convert the combined dictionary to a Pandas DataFrame
combined_df = pd.DataFrame(results_combined)

# Convert the Pandas DataFrame to a Spark DataFrame
spark_combined_df = spark.createDataFrame(combined_df)

# Define the output path in your Google Cloud bucket
output_path = "gs://metcs777_test/output_all_results"

# Save the Spark DataFrame to Google Cloud Storage as a CSV file
spark_combined_df.write.mode("overwrite").csv(output_path, header=True)

print("All results saved to Google Cloud Storage successfully!")