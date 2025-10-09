---
marp: true
theme: default
paginate: true
footer: 'Eng. Josué Huamán'
style: |
  section {
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/5/51/Google_Cloud_logo.svg');
    background-size: 250px;
    background-position: 95% 90%; /* esquina inferior derecha */
    background-repeat: no-repeat;
    opacity: 1;
  }

---

![bg](https://upload.wikimedia.org/wikipedia/commons/5/51/Google_Cloud_logo.svg)

---
<!-- _class: lead -->
<!-- _paginate: false -->

# Book Summary Presentation

### AI-Generated Book Summary

---<!-- _class: lead -->

## 📚 Table of Contents

1. **Table of Contents** (4 sections)
2. **Official Google Cloud CertifiedProfessional Machine Learning EngineerStudy Guide** (0 sections)
3. **Acknowledgments** (0 sections)
4. **About the Author** (0 sections)
5. **About the Technical Editors** (2 sections)
6. **Introduction** (9 sections)
7. **Chapter 1Framing ML Problems** (7 sections)
8. **Chapter 2Exploring Data and Building Data Pipelines** (11 sections)
9. **Chapter 3Feature Engineering** (9 sections)
10. **Chapter 4Choosing the Right ML Infrastructure** (8 sections)
11. **Chapter 5Architecting ML Solutions** (8 sections)
12. **Chapter 6Building Secure ML Pipelines** (6 sections)
13. **Chapter 7Model Building** (9 sections)
14. **Chapter 8Model Training and Hyperparameter Tuning** (10 sections)
15. **Chapter 9Model Explainability on Vertex AI** (4 sections)
16. **Chapter 10Scaling Models in Production** (9 sections)
17. **Chapter 11Designing ML Training Pipelines** (7 sections)
18. **Chapter 12Model Monitoring, Tracking, and Auditing Metadata** (9 sections)
19. **Chapter 13Maintaining ML Solutions** (8 sections)
20. **Chapter 14BigQuery ML** (9 sections)
21. **AppendixAnswers to Review Questions** (14 sections)
22. **Index** (24 sections)
23. **Online Test Bank** (1 sections)
24. **WILEY END USER LICENSE AGREEMENT** (0 sections)# 2. Official Google Cloud CertifiedProfessional Machine Learning EngineerStudy Guide

---

# 3. Acknowledgments

---

# 4. About the Author

---

# 5. About the Technical Editors

---

## 5.1. About the Technical Proofreader

* **About the Author**: Educator and Google Cloud Certified Multi-Tester
    * Has created multiple courses on machine learning
    * Plays tabletop games, reads sci-fi/fantasy novels, and hikes in his free time

---

## 5.2. Google Technical Reviewer

* **Acknowledgments**
    * _Special thanks to_ Emma Freeman, Google Technical Reviewer
        * _Reviewed book proofs_

---

# 6. Introduction

---

## 6.1. Google Cloud Professional Machine Learning Engineer Certification

**Professional Machine Learning Engineer**
* Designs, builds, and productionizes ML models using Google Cloud technologies
* Develops responsible AI practices throughout the development process
* Proficient in model architecture, data pipeline interaction, and metrics interpretation

---

### 6.1.1. Why Become Professional ML Engineer (PMLE) Certified?

**Gaining PMLE Certification**
* Enhances marketability and job prospects
* Recognizes expertise in AI/ML on GCP
* Raises customer confidence in secure, cost-effective, and scalable ML solutions

---

### 6.1.2. How to Become Certified

* **Google Cloud Exam Requirements**: 
  * Industry experience: 3+ years, with Google Cloud design and management
  * Exam format: Online or in-person, proctored
* _Exam Details_:
  * Time: 2 hours
  * Questions: 50-60 multiple-choice
* _Registration Options_: 
  * Online proctored exam (www.webassessor.com)
  * In-person proctored exam at a testing center (www.kryterion.com)

---

## 6.2. Who Should Buy This Book

**Overview**
* Book covers ML technology on Google Cloud Platform
* Designed for students, developers, and IT professionals seeking expertise in ML engineering

* **Key Topics:**

  * Machine learning process (data to deployment)
  * Best practices for custom vs. AutoML/pretrained models
  * Designing secure ML cloud environments

---

## 6.3. How This Book Is Organized

**Machine Learning Fundamentals**
* **Overview**: The book consists of 14 chapters covering various aspects of machine learning (ML) from problem framing to deployment in production.
* _Key Topics_: ML infrastructure, feature engineering, model building, training and hyperparameter tuning, explainability, scalability, security, and maintenance.
* **Learning Path**: Readers will learn how to design reliable, scalable, and secure ML solutions, from data exploration to model monitoring and auditing.

---

### 6.3.1. Chapter Features

### Exam Structure and Study Tips

* **Exam Essentials**: Summary of chapter objectives
* Review questions at the end of each chapter (8+)
* Focus on learning underlying topic, not memorizing specific answers
* Read chapters from start to finish and review regularly for best results

---

## 6.4. Bonus Digital Contents

* **Companion Resources**: 
    * Online learning environment with practice tests, flash cards, and glossary
    * 30-question assessment test, 100+ review questions per chapter, and two 50-question bonus exams
    * Digital companion files include 50+ questions in flash card format

---

### 6.4.1. Interactive Online Learning Environment and Test Bank

You can access all these resources at www.wiley.com/go/sybextestprep .

---

## 6.5. Conventions Used in This Book

* _Typographic Styles_ are used throughout the book to:
  * Highlight **key terms** in italicized text
  * Indicate configuration files and other technical details in monospaced font

The book also employs various convention-based formatting to:
  * Distinguish between main content and supplementary notes (*_Notes_*), which provide interesting but peripheral information
  * Offer helpful *_Tips_* that can save time or frustration

---

## 6.6. Google Cloud Professional ML Engineer Objective Map

**Summary** * **Low-Code ML Solutions**: Architecting, building, training, and serving machine learning models using BigQuery ML, ML APIs, AutoML, Vertex AI, and other Google Cloud services. * **Collaboration and Model Prototyping**: Collaborating with teams to manage data and models, prototyping models in Jupyter notebooks, tracking and running ML experiments, and testing model performance. * **Scaling and Serving Models**: Scaling prototypes into production-ready models, serving models through batch and online inference, scaling online model serving, and tuning ML models for training and serving. * _Key Considerations:_ 1. **Hardware Evaluation**: Choosing appropriate hardware (CPU, GPU, TPU, edge) for training and serving models. 2. **Model Explainability**: Monitoring model

---

performance, tracking bias, fairness, and explainability on Vertex AI Prediction. 3. **Continuous Evaluation Metrics**: Establishing continuous evaluation metrics (Vertex AI Model Monitoring, Explainable AI) to monitor ML solution readiness.

---
## 6.7. How to Contact the Publisher

* **Reporting Errors**: If you find a mistake in this book, report it to John Wiley & Sons via email (_wileysupport@wiley.com_) using the subject line "Possible Book Errata Submission".

---

# 7. Chapter 1Framing ML Problems

---

## 7.1. Translating Business Use Cases

**Step 1: Identify Use Case**

* Define the problem or opportunity
* Align with business goals and stakeholders' expectations
* Determine key metrics for success

**Step 2: Match Use Case to Machine Learning Problem**

* Choose a suitable algorithm and metric
* Consider impact, data availability, and feasibility of solution

**Step 3: Validate Solution Feasibility**

* Assess existing technology and available data
* Evaluate budget constraints and potential risks
* Identify potential advancements in related fields (e.g. natural language processing)

---

## 7.2. Machine Learning Approaches

* **Understanding ML Approaches**: Machine learning has many algorithms and methods, each solving specific types of problems.
* **Classification Categories**: ML approaches are categorized into groups based on data type, prediction type, etc.
* *_Knowledge Required_*: Understanding these categories is crucial to solve use cases correctly and choose the right approach.

---

### 7.2.1. Supervised, Unsupervised, and Semi‐supervised Learning

* **Classification by Type**: Machine learning approaches can be classified into two main types: supervised and unsupervised.
    * **Supervised Learning**: Uses labeled data to train models, e.g., image classification, sentiment analysis
    * _Unsupervised Learning_: Uses unlabeled data to group or classify data, e.g., clustering algorithms, topic modeling

---

### 7.2.2. Classification, Regression, Forecasting, and Clustering

**Classification**

* Predicting labels or classes
* Binary classification (2 labels): e.g., dogs vs cats
* Multiclass classification (more than 2 labels): e.g., classifying millions of objects in a picture
* Different algorithms used for binary, multiclass, and multi-thousand-class problems

**Regression**

* Predicting a number: e.g., house price, rainfall amount
* Predicted value's range depends on the use case
* Uses different ML algorithms than classification

**Forecasting**

* Predicting future values using time-series data
* Models predict future values based on past data

**Clustering**

* Algorithm groups data points into clusters based on similarities and differences
* Example: grouping cities by distances between house coordinates

---

## 7.3. ML Success Metrics

### **Choosing the Right Machine Learning Metric**

* To determine if a trained model is accurate enough, an ML metric or suite of metrics is used.
* Metrics align with business success criteria and are chosen based on the class of problems.

### **Classification Metrics**
* **Precision**: Measures the percentage of positive predictions that were actually correct. Lower false positives.
	+ Formula: `TP / (TP + FP)`
* **Recall**: Measures the percentage of true positives correctly predicted. Lower false negatives.
	+ Formula: `TP / (TP + FN)`
* **F1 Score**: Harmonic mean of precision and recall, reduces both false positives and false negatives.
	+ Formula: `(2 x Precision x Recall) / (Precision + Recall)`

---

### 7.3.1. Area Under the Curve Receiver Operating Characteristic (AUC ROC)

**ROC Curve**
* Graphical plot summarizing binary classification model performance
* x-axis: False positive rate, y-axis: True positive rate
* Ideal point: Top-left corner (100% TP, 0% FP)
* _Key goal:_ Maximize curve distance from diagonal line

---

### 7.3.2. The Area Under the Precision‐Recall (AUC PR) Curve

* **AUC-PR Curve**: A graphical plot illustrating the relationship between recall and precision.
    * Measures how well a model separates positive instances from negative ones.
    * The best AUC-PR is a horizontal line at the top of the curve, indicating 100% precision and recall.

---

### 7.3.3. Regression

### Regression Metrics Summary * **Mean Absolute Error (MAE)**: Average absolute difference between actual and predicted values. * *Average of absolute differences*: RMSE: Square root of average squared differences + _Penalizes large errors_: Used to penalize incorrect predictions + _Range: 0 - ∞_ * **Root Mean Squared Logarithmic Error (RMSLE)**: Asymmetric metric, penalizes under-prediction + _Uses natural logarithm +1_: For asymmetry and under-prediction penalty + _Range: 0 - ∞_ * **Mean Absolute Percentage Error (MAPE)**: Average absolute percentage difference between actual and predicted values + _Used for proportional differences_ + _Range: 0 - 100%_ * **R-squared (R²)**: Square of Pearson correlation coefficient, measures fit + _Range: 0 - 1_, generally higher indicates better fit

---
## 7.4. Responsible AI Practices

**Machine Learning Best Practices**
=====================================

* _Fairness_: Consider fairness in ML solutions, as models can reflect and reinforce unfair biases.
	+ Use statistical methods to measure bias in datasets and test ML models for bias.
* _Interpretability_: Improve interpretability using model explanations that quantify input feature contributions.
	+ Some algorithms support model explanations; others do not.
* _Security_: Identify potential threats, keep learning to stay ahead of the curve, and develop approaches to combat these threats.

---

## 7.5. Summary

* *_Understanding Business Use Cases_*: Learning to frame a machine learning problem statement from a business use case
    * Identifying key dimensions of a request
    * Defining a clear problem statement
    * _Setting the foundation for data analysis_

---

## 7.6. Exam Essentials

* **Machine Learning Fundamentals**
	+ Understand business use cases and problem types (regression, classification, forecasting)
	+ Familiarize with data types and popular algorithms
	+ Know ML metrics and match them to use cases (_precision, recall, F1, AUC ROC, RMSE, MAPE_)
* **Responsible AI Practices**
	+ Understand Google's Responsible AI principles
	+ Apply recommended practices for fairness, interpretability, privacy, and security

---

# 8. Chapter 2Exploring Data and Building Data Pipelines

---

## 8.1. Visualization

**Data Visualization**
* Technique to explore trends and outliers in data
* Helps with data cleaning and feature engineering

---

### 8.1.1. Box Plot

* **Box Plot**: Visual representation of data, showing _25th, 50th (Median), and 75th percentiles_.
* **Components**:
	+ Body: Interquartile Range (IQR) where most data points are present
	+ Whiskers: Maximum and Minimum values

---

### 8.1.2. Line Plot

* **What is a Line Plot?**: A graph that displays the relationship between two variables over time.
* **Trend Analysis**: Used to identify patterns and trends in data changes over time.
* *_Visual Representation_*: Shows data points connected by straight line segments, forming a visual representation of the relationship.

---

### 8.1.3. Bar Plot

*_Bar Plot_*


* A graphical representation used for analyzing trends in data and comparing categorical values
* Used to visualize sales figures, website traffic, or revenue over time
* _Helps identify patterns and correlations in data_

---

### 8.1.4. Scatterplot

* **What is a Scatterplot?**: A scatterplot is a type of plot used to visualize data and show relationships between two variables.
* *It displays clusters and patterns in data.*
* *Commonly used to analyze two-variable relationships.*

---

## 8.2. Statistics Fundamentals

**Measures of Central Tendency**
* Mean: average value
* Median: middle value (arranged in order)
* Mode: most frequently occurring value

---

### 8.2.1. Mean

Mean is the accurate measure to describe the data when we do not have any outliers present.

---

### 8.2.2. Median

### **Calculating Median**

* Find the median when there are outliers in a dataset.
* Arrange data values from lowest to highest value.
* If even numbers, take the average of middle two values.
* If odd numbers, take the middle value.
* Example: 
    * 1, 1, 2, 4, 6, 6, 9 -> Median is 4
    * 1, 1, 4, 6, 6, 9 -> Median is 5

---

### 8.2.3. Mode

### What is Mode?
* _The value(s) that appear most frequently in a dataset_
* Used when most data points are equal but one point stands out as an outlier
* **Example**: Dataset 1, 1, 2, 5, 5, 5, 9 has mode 5

---

### 8.2.4. Outlier Detection

**Outliers' Impact on Statistics**

* **Mean**: sensitive to outliers, which can significantly change its value
* Variance measures the average of squared differences from the mean
* _Comparison between mean, median, and mode_:
    * Mean: 12.72 (with outlier), 29.16 (without outlier)
    * Median: 13 (with outlier), 14 (without outlier)
    * Mode: 15

---

### 8.2.5. Standard Deviation

### **Key Statistics Concepts**

* **Standard Deviation**: Square root of variance
* **Identifying Outliers**: Data points > 1 standard deviation from the mean are considered unusual.
* **Covariance**: Measure of how two random variables vary from each other

---

### 8.2.6. Correlation

### Correlation and Its Types
* **Definition**: Correlation is a normalized form of covariance, ranging from -1 to +1.
* **Types**:
	+ **Positive Correlation**: Values increase together.
	+ **Negative Correlation**: Values decrease together.
	+ **Zero Correlation**: No substantial impact on each other.

---

## 8.3. Data Quality and Reliability

* **Data Quality is Crucial**
	+ Unreliable data can lead to poor model performance
	+ Data should be clean, reliable, and well-defined before training
	+ Common issues: label errors, noise in features, outliers, and skew

---

### 8.3.1. Data Skew

**Data Skew**
* _Definition:_ Asymmetry in a normal distribution curve due to outliers or uneven data distribution
* **Types:** Right-skewed and left-skewed data
* **Effect on models:** Extreme outliers affect prediction capabilities

---

### 8.3.2. Data Cleaning

* Normalization transforms features to similar scales, improving model performance and training stability.
* **Goal**: Reduce variability in feature values.
* *_Key benefit_*: Improved model accuracy and reliability.

---

### 8.3.3. Scaling

### **Scaling in Deep Neural Networks**

* **Benefits**: 
  * Improves gradient descent convergence
  * Removes "NaN traps"
  * Prevents features with large ranges from dominating the model
* **Use cases**: Uniformly distributed or nearly normal data with few outliers, such as age or income

---

### 8.3.4. Log Scaling

* **Log Scaling**: Used when data samples have different scales, e.g., large numbers and small values.
    * _Scales large values to a comparable range_
    * Example: `log(100,000) = 5`, `log(100) = 2`

---

### 8.3.5. Z‐score

### Z-Score Scaling Method
* **Calculating Scaled Value**: `(value − mean) / stddev`
* _Mean_: 100
* _Standard Deviation_: 20
* Example: Original value = 130 (Scaled value: 1.5)
* A z-score between -3 and +3 indicates a valid value

---

### 8.3.6. Clipping

* _Feature Clipping_: Limits feature values to a fixed range (above/below) to prevent extreme outliers from affecting models.
* Can be performed **before** or **after** other normalization techniques.
* Aims to reduce the impact of outliers on model performance.

---

### 8.3.7. Handling Outliers

* An **outlier** is a value that significantly differs from other data points due to human error, skew, or other factors.
* Techniques for detecting outliers include:
	+ Box plots
	+ Z-score
	+ Interquartile range (IQR)
	+ Clipping
* Once detected, outliers can be **removed**, **imputed** with a specific value (mean, median, mode, or boundary values), or **replaced**.

---

## 8.4. Establishing Data Constraints

# Schema for Data Constraint
* Establishes a consistent and reproducible check for data quality issues
    * Defines metadata to describe data properties (data type, range, format, distribution)
* Enables metadata-driven preprocessing and validation of new data
* Catches anomalies such as skews and outliers during training and prediction

---

### 8.4.1. Exploration and Validation at Big‐Data Scale

### **Data Validation for Large Datasets**

* **What is TFDV?**: TensorFlow Data Validation (TFDV) is a part of the TensorFlow Extended (TFX) platform, providing libraries for data validation and schema validation.
* **Key Features**:
	+ Detects data anomalies and schema anomalies
	+ Provides a defined contract between ML pipeline and data
	+ Enables baseline definition to detect skew or drift in deployed models

---

## 8.5. Running TFDV on Google Cloud Platform

**TFDV Core APIs**
* Built on Apache Beam SDK
* **Data Processing Pipelines**: Run in managed Dataflow service at scale
* Integrated with:
  * _BigQuery_ for serverless data warehousing
  * Google Cloud Storage (data lakes)

---

## 8.6. Organizing and Optimizing Training Datasets

**Dataset Splitting**
* Training Dataset: actual dataset for model training
* Validation Dataset: subset for hyperparameter tuning and model evaluation
* Test Dataset: sample for testing model performance after training and validation

---

### 8.6.1. Imbalanced Data

**Handling Imbalanced Data**
==========================

* When dealing with imbalanced data, where one class has significantly fewer instances than others, traditional machine learning algorithms can be biased towards the majority class.
* Downsampling and upweighting the majority class can improve model performance by increasing the representation of the minority class.

**Downsampling and Upweighting**
-----------------------------

* Downsampling reduces the number of samples from the majority class to balance the dataset.
* Upweighting increases the importance of downsampled examples during training, allowing models to converge faster.

---

### 8.6.2. Data Splitting

**Data Splitting for Classification**
* Random splitting can cause skew due to clustered examples (e.g., same topic, written on same timeline)
* Fix: split data based on publication time or date
* Examples:
	+ Train on stories from one month, test on next month
	+ Use separate dates for training and testing sets

---

### 8.6.3. Data Splitting Strategy for Online Systems

### Data Splitting for Online Systems
* **Split data by time** to ensure validation set mirrors training-prediction lag.
* **Use time-based splits** for large datasets and _ensure model can access all relevant data at prediction time_.
* Use **domain knowledge** to determine when a random split is suitable versus a time-based approach.

---

## 8.7. Handling Missing Data

**Handling Missing Data**
* Deleting rows/columns with high null/NaN values can lead to poor model performance.
* Replacing missing values with mean, median, mode, or imputing most frequent category can prevent data loss.

**Alternative Methods**

* Using machine learning algorithms like k-NN, Naive Bayes, or Random Forest that ignore missing values
* Interpolating missing values using the last valid observation or time-series interpolation

---

## 8.8. Data Leakage

**Data Leakage in Machine Learning**
=====================================

* **Definition**: Exposing a machine learning model to test data during training, leading to overfitting.
* _Causes_:
	+ Adding target variable as feature
	+ Including test data with training data during splitting
	+ Applying preprocessing techniques to entire dataset
* _Examples_:
	+ Time-series data: using future data for current features or predictions
	+ Randomly splitting data without considering the issue of leakage

---

## 8.9. Summary

* **Data Visualization**: Techniques like box plots, line plots, and scatterplots are used to visualize data.
* **Data Cleaning and Normalization**:
	+ Log scaling
	+ Scaling
	+ Clipping
	+ Z-score normalization
* **Data Validation and Schema Definition**:
	+ Establishing data constraints
	+ Defining a data schema in an ML pipeline
	+ Validating data using TFDV (for large-scale deep learning systems)

---

## 8.10. Exam Essentials

**Data Preprocessing Fundamentals**

* _Visualizing Data_: Box plots, line plots, scatterplots for understanding data distribution and outliers.
* _Statistical Terms_: Mean, median, mode, standard deviation, and correlation analysis using line plots.
* _Data Quality & Reliability_: Outliers, data skew, data cleaning techniques (log scaling, scaling, clipping, z-score), data schema validation, TFDV.

---

# 9. Chapter 3Feature Engineering

---

## 9.1. Consistent Data Preprocessing

**Data Preprocessing Approaches**

* **Pretraining Data Transformation**
  * Advantages:
    + Computation performed only once
    + Can analyze entire dataset for transformations
  * Disadvantages:
    + Same transformation must be reproduced at prediction time
    + Updating transformation can lead to slow iterations and increased compute time
* **Inside Model Data Transformation**
  * Advantages:
    + Easy to decouple data and transformation
    + Same transformation used during training and serving
  * Disadvantages:
    + Can increase model latency with large or computation-heavy transformations

---

## 9.2. Encoding Structured Data Types

### Types of Data in Feature Engineering
* **Categorical Data**: Defines a category with limited values (e.g., yes/no, male/female)
* **Numeric Data**: Represented by scalar values or continuous data (e.g., observations, recordings, measurements)

---

### 9.2.1. Why Transform Categorical Data?

### Conversion of Categorical Data in Machine Learning
* **Categorical data cannot directly feed into ML algorithms** 
    * Most ML algorithms require **numeric input and output variables**
    * Converting categorical data to numeric data is necessary for algorithm operation

---

### 9.2.2. Mapping Numeric Values

* **Data Transformation**: Numeric data often requires _normalization_ (scaling) and/or _bucketing_ (grouping into categories). 
* *Normalization*: scales values to common range.
* *Bucketing*: groups values into discrete categories.

---

### 9.2.3. Mapping Categorical Values

* **Converting Categorical Data**
* _Transforms categorical data into numerical data for machine learning models_ 
* **Methods include: one-hot encoding, label encoding, and target encoding**

---

### 9.2.4. Feature Selection

### Feature Selection Techniques for Dimensionality Reduction

* **Methods to reduce dimensionality**:
	+ Remove most important features
	+ Combine new features using PCA and t-SNE
* **Benefits**: reduces noise, overfitting, and increases model performance
* **Advantages**: less training time, computational resources

---

## 9.3. Class Imbalance

* **Class Imbalance in Classification Models**: 
  * Classification models predict positive or negative outcomes.
  * False negatives (sick patients mislabeled as healthy) are a problem, as they can lead to delayed treatment.
  * Minimizing false negatives is key to improving classification model accuracy.

---

### 9.3.1. Classification Threshold with Precision and Recall

**Binary Classification**
* _Classification Threshold_: a value chosen by humans to map logistic regression output to a binary category (e.g., 0.5)
* _Precision Recall Curve_: indicates how well a model predicts the positive class
* **Trade-offs between Precision and Recall**: 
    * _High Precision_ means fewer false positives, but may reduce recall (more false negatives)
    * _High Recall_ means more correct positives, but may increase false positives

---

### 9.3.2. Area under the Curve (AUC)

**Classification Problem Types**

* _AUC ROC_: Balanced dataset with equal examples of both classes.
* _AUC PR_: Imbalanced dataset, where one class has significantly more examples than the other.

---

## 9.4. Feature Crosses

**Feature Crosses**
* _Created by multiplying two or more features together (e.g. A * B)_
* Used to represent nonlinearity in linear models or to combine features with less predictive ability
* Example: `location × time of day` to improve prediction accuracy for street crowd density

---

## 9.5. TensorFlow Transform

* **Efficient Input Pipeline**: Increasing model performance with TensorFlow
* *_Key Components:_* 
  * TF Data API
  * TensorFlow Transform

---

### 9.5.1. TensorFlow Data API (tf.data)

### Efficient Data Input Pipeline Best Practices
#### Improve Model Execution Speed
* **Prefetch Transformation**: Overlap preprocessing and model execution using `tf.data.Dataset.prefetch`
* **Interleave Transformation**: Parallelize data reading to mitigate extraction overhead with `tf.data.Dataset.interleave`
* **Cache Transformation**: Cache data in memory during the first epoch with `tf.data.Dataset.cache`

---

### 9.5.2. TensorFlow Transform

**TensorFlow Transform Library**
_____________________________

* **Overview**: Allows transformations prior to training and emits a TensorFlow graph for reproducing these transformations during training, avoiding training-serving skew.
* **Key Steps**: 
    * _Data Analysis & Transformation_: Analyze and transform data using `tf.Transform` in Cloud Dataflow.
    * _Model Training & Serving_: Train and serve models using TensorFlow and Vertex AI services.

---

## 9.6. GCP Data and ETL Tools

* **Google Cloud Tools for Data Transformation**
	+ *_Cloud Data Fusion_*: A managed service for building and managing ETL/ELT pipelines, supporting Cloud Dataproc and allowing MapReduce and Spark streaming pipeline execution.
	+ *_Dataprep by Trifacta_*: A serverless, UI-based tool for visually exploring, cleaning, and preparing data for analysis and machine learning at any scale.

---

## 9.7. Summary

### **Feature Engineering**

* Transforming numerical and categorical features for model training and serving is crucial.
	+ Techniques: bucketing, normalization, linear encoding, one-hot encoding, hashing, embedding
* Selecting features and dimensionality reduction techniques like PCA are also important.
* AUC PR is more effective than AUC ROC for imbalanced classes.

### **Data Processing**

* Representing data for TensorFlow using tf.data
* Using tf.Transform for pipeline processing on Google Cloud
	+ Tools: Cloud Data Fusion, Cloud Dataprep

---

## 9.8. Exam Essentials

**Data Processing Best Practices**
* _Understand when and how to transform data before training_
* **Key techniques:** bucketing, normalization, hashing, one-hot encoding
* **Feature selection**: understand why it's needed and techniques like dimensionality reduction

---

# 10. Chapter 4Choosing the Right ML Infrastructure

---

## 10.1. Pretrained vs. AutoML vs. Custom Models

### _Pretrained Models_
* **_Ease of Use and Speed_**: Pretrained models are easily deployable via APIs, requiring minimal algorithm knowledge.
* **_Advantages_**: Fast incorporation of ML into applications, no need to provision cloud resources or manage scalability.
* **_Limitations_**: Least customizable, may not meet specific use case requirements.

---

## 10.2. Pretrained Models

* **Pretrained Models**: Machine learning models trained on large datasets, perform well in benchmark tests, and are supported by large teams.
* _Key Benefits_: 
    * Fast deployment through web console or SDKs
    * Frequent retraining to stay up-to-date with new data
* Google Cloud offers several pretrained models for:
    * Vision AI
    * Video AI
    * Natural Language AI
    * Translation AI
    * Speech-to-Text and Text-to-Speech

---

### 10.2.1. Vision AI

* _**Vision AI**: Provides convenient access to ML algorithms for image processing and analysis_
* * **Predictions:**
	+ Object detection
	+ Image labels (e.g. Table, Furniture)
	+ Dominant colors for organization
	+ "Safe Search" classification (e.g. Adult, Spoof)

---

### 10.2.2. Video AI

**Video AI API**

* Recognizes objects, places, and actions in videos (20,000+ entities)
* Can be applied to stored or streaming video for real-time results
* **Use Cases:**
    * Building a video recommendation system using labels and user history
    * Creating an index of video archives with metadata
    * Improving user experience by comparing advertisement relevance

---

### 10.2.3. Natural Language AI

**Google Natural Language AI**
### Overview
* **Entity Extraction**: identifies entities such as people, organizations, products, events, locations with additional information like links to Wikipedia articles.
* **Sentiment Analysis**: provides positive, negative, or neutral scores for each sentence, entity, and text.
* **Syntax Analysis**: identifies part of speech, dependency between words, lemma, and morphology.

### Use Cases
* **Customer Sentiment Analysis**: measure customer opinion on products using emails, chat, social media.
* **Healthcare Text Analysis**: extract medical insights like drugs, dosage, conditions from clinical notes or research documents.

---

### 10.2.4. Translation AI

**Translation AI Service**
* Detects over 100 languages using Google Neural Machine Translation (GNMT) technology
* Two levels: Basic and Advanced, with key differences including glossary support and document translation capabilities
* Translates text in ASCII or UTF-8 format, as well as real-time audio translations using the Media Translation API

---

### 10.2.5. Speech‐to‐Text

* **Speech-to-Text Service**: Converts recorded audio or streaming audio into text
* *Used for creating subtitles for video recordings and streaming video*
* *Can be combined with translation services to generate subtitles in multiple languages*

---

### 10.2.6. Text‐to‐Speech

* **Text-to-Speech Service**: Provides realistic speech with humanlike intonation using DeepMind's AI expertise
* _Supported Languages and Voices_: 40+ languages, 220+ voices, including the ability to create a unique voice for your brand
* *_Voice List Available Here_*: https://cloud.google.com/text-to-speech/docs/voices

---

## 10.3. AutoML

*_Automated Machine Learning (AutoML)_*

* Automates model training for popular ML problems like image classification, text classification, and more
* User provides data and configures settings; rest of training is automated through web console or SDKs in Python, Java, or Node.js.

---

### 10.3.1. AutoML for Tables or Structured Data

**Structured Data and Machine Learning Training** ============================================= ### Overview Structured data is stored in a well-defined schema, typically in a rectangular table format. ### Training Methods * **BigQuery ML**: A SQL-based approach using BigQuery, suitable for data analysts who write SQL queries. + Train and make predictions simultaneously. + Serverless approach for ML training and prediction. * **Vertex AI Tables**: Triggered by Python, Java, or Node.js, or REST API. + Deploy model on an endpoint and serve predictions through a REST API. ### AutoML Tables Algorithms ------------------------------ * **Classification**: + Data Type: Table (IID) + Metrics: AUC ROC, Logloss, Precision at Recall, Recall at Precision * **Regression**: + Data Type: Table (IID) +

---

Metrics: RMSE, RMSLE, MAE * **Time-series data Forecasting**: + Data Type: Time series data + Metrics: RMSE, RMSLE, MAPE, Quantile loss

---
### 10.3.2. AutoML for Images and Video

**Vertex AI AutoML for Image and Video Data**
* **Automates Machine Learning**: Makes it easy to build models for image and video data
* **Available Algorithms**: 
    * Image: classification, multiclass classification, object detection, segmentation
    * Video: classification, action recognition, object tracking
* **Edge Models**: Deployable on edge devices (iPhones, Android phones, Edge TPU devices)

---

### 10.3.3. AutoML for Text

### Machine Learning for Text with AutoML
* **Text Classification**: predict one or multiple correct labels on documents
* *_Text to Text Translation_*: convert text from source language to target language
* **Entity Extraction**: identify entities within text items

---

### 10.3.4. Recommendations AI/Retail AI

**Google Cloud Retail AutoML Solution**
* Provides a Google-quality search with customizable intent and context
* Offers image search using Vision API Product Search
* Employs Recommendations AI to drive engagement through relevant product recommendations
    * Models are continuously fine-tuned based on customer behavior data
    * Customers are charged for model training and each 1,000 requests

---

### 10.3.5. Document AI

### **Document AI**

* Extracts details from digitized scanned images and handwritten documents
* Handles variability in text, handwriting, and document structure
* Produces structured data that can be stored in a database and analyzed

### **Key Components**

* **Processors**: Interface between documents and machine learning models
	+ General processors
	+ Specialized processors (e.g. procurement, identity)
	+ Custom processors (trainable by providing training datasets)
* **Document AI Warehouse**: Platform for storing, searching, organizing, governing, and analyzing documents with structured metadata

---

### 10.3.6. Dialogflow and Contact Center AI

* **Google Cloud Service**: Dialogflow offers conversational AI for Google Cloud, including chatbots and voicebots.
* _Contact Center AI Solution_: Integrates into telephony services and other solutions to provide a CCAI (Contact Center AI) solution.
* *_Features and Capabilities_*: Provides customizable chatbot and voicebot capabilities for various industries and applications.

---

## 10.4. Custom Training

* **Accelerating Deep Learning**: GPUs significantly speed up deep learning model training with massive parallel architectures.
* **Training Time Reduction**: Training on a single CPU can take weeks or months, while GPU acceleration reduces time by an order of magnitude.
* *_Harnessing Compute Intensity_*: Models requiring compute-intensive operations like matrix multiplications benefit greatly from running on GPUs.

---

### 10.4.1. How a CPU Works

**CPU Architecture**

* A general-purpose processor that supports various operations
* Loads data from memory, performs operation, and stores result back into memory for each operation
* **Inefficient for parallel computations**, such as those found in machine learning model training

---

### 10.4.2. GPU

**GPUs in Vertex AI**
* **Overview**: Specialized chip designed to rapidly process data in memory for image processing and machine learning tasks.
* **Parallel Processing**: Thousands of arithmetic logic units (ALUs) enable parallel processing, significantly improving speed.
* **Availability and Restrictions**: Must use compatible instance types, regions, and machine configurations to ensure sufficient virtual CPUs and memory.

---

### 10.4.3. TPU

### **TPUs (Tensor Processing Units)**

* Designed by Google for machine learning workloads
* Specialized hardware accelerators with multiple matrix multiply units (MXUs)
* Each MXU has 128x128 multiply/accumulators, performing 16,000 operations per cycle
* Primary task is matrix processing for neural network training

---

## 10.5. Provisioning for Predictions

### Predictions Phase
* **Hardware Provisioning Overview**: Prediction phase involves deploying models on servers or processing large volumes of data as batch jobs.
* **Workload Characteristics**:
	+ _Online prediction_ requires near real-time response, continuous scaling, and low cost.
	+ _Batch prediction_ prioritizes completion time and cost over responsiveness.

---

### 10.5.1. Scaling Behavior

* *_Autoscaling in Vertex AI_* 
  * Automatically scales prediction nodes when CPU usage is high
  * Requires proper configuration of GPU node triggers due to multi-resource monitoring (_CPU, memory, GPU_)

---

### 10.5.2. Finding the Ideal Machine Type

**Cloud Deployment Considerations**
* Deploy custom prediction containers as Docker containers to Compute Engine instances directly
* Benchmark instance performance and determine optimal machine type based on QPS cost per hour
* Consider limitations of single-threaded web servers, resource utilization, latency, and throughput requirements

**Cloud GPU Restrictions**
* GPUs only compatible with TensorFlow SavedModels or custom containers designed for GPU acceleration
* Scikit-learn and XGBoost models not supported
* Limited availability in some regions

---

### 10.5.3. Edge TPU

### **Edge Devices in IoT**

* _Collect and process real-time data locally_
* _Enable decision-making and action on limited-bandwidth devices_
* _Use edge inference to accelerate machine learning operations_

---

### 10.5.4. Deploy to Android or iOS Device

**ML Kit**
* A mobile platform that integrates Google's machine learning expertise
* Optimized for iOS and Android devices, enabling fast and efficient predictions
* Allows deployment of trained models directly into apps for personalized experiences

---

## 10.6. Summary

* **Overview**: Learned about pretrained models, AutoML, and available hardware options on Google Cloud for model training and prediction.
    * **Hardware Options**: Utilize GPUs, TPUs, or deploy to edge devices
        + _Cloud-based_: Google Cloud provides a variety of hardware accelerators.

---

## 10.7. Exam Essentials

### Choosing the Right ML Approach
* **Pretrained Models**: Use pre-trained models for tasks requiring less custom development.
* **AutoML**: Employ AutoML tools for automated machine learning when flexibility is needed.
* **Custom Models**: Design custom models when specific requirements cannot be met by other approaches.

Note: The above summary highlights the key aspects of choosing an ML approach, without going into details.

---

# 11. Chapter 5Architecting ML Solutions

---

## 11.1. Designing Reliable, Scalable, and Highly Available ML Solutions

### Overview of ML Pipeline Automation on GCP

To design reliable, scalable, and available ML solutions, you need to automate the following steps:

* **Data Collection**: Store data in Google Cloud Storage, Pub/Sub (streaming data), and BigQuery.
* **Model Deployment & Monitoring**: Deploy models with Vertex AI Prediction, monitor with Vertex AI Model Monitoring.
* _Automate the entire pipeline using Vertex AI Pipelines_

---

## 11.2. Choosing an Appropriate ML Service

### **Cloud AI/ML Layered Services**

* The Google Cloud ML services are divided into three layers based on ease of use and implementation:
	+ Top layer: _AI Solutions_ (SaaS) for easy implementation and management with no code.
	+ Middle layer: _Vertex AI_ services, including pretrained APIs, AutoML, and Workbench for building custom models.
	+ Bottom layer: _Infrastructure_ for managing compute instances, containers, and storage.

---

## 11.3. Data Collection and Data Management

* **Data Stores in Google Cloud**
    * **Overview**: Google Cloud offers various data stores to cater to different needs:
        - **Google Cloud Storage**: Suitable for large-scale, high-throughput data storage.
        - *_Vertex AI_*: Manages training and annotation datasets for machine learning models.
        - **Vertex AI Feature Store**: Stores pre-trained machine learning models and features.
    * **NoSQL Data Store**: A flexible option for handling varying requirements.

---

### 11.3.1. Google Cloud Storage (GCS)

* **Google Cloud Storage (GCS)**: a service for storing objects in Google Cloud
    * Storing various data types, including images, videos, audio, and unstructured data
        * Large files (> 100 MB) can be stored in shards to improve performance

---

### 11.3.2. BigQuery

### **BigQuery Storage Best Practices**

* Store tabular data in BigQuery for better speed
* Use tables instead of views for training data

---

### 11.3.3. Vertex AI Managed Datasets

**Managed Datasets for Training Models**
* _Use Vertex AI managed datasets instead of ingesting data directly from storage_ 
    * Supports 4 primary data formats: image, video, tabular (CSV, BigQuery), and text
* **Advantages**: Centralized dataset management, integrated data labeling, lineage tracking, model comparison, and data statistics generation

---

### 11.3.4. Vertex AI Feature Store

* **Vertex AI Feature Store**: A fully managed repository for organizing and serving ML features
    * **Features**:
        * Search and retrieve values from the Feature Store
        * Create new features if they don't exist
        * Retrieve feature values in real-time for fast online predictions
    * **Benefits**: 
        * No need to compute and save feature values manually
        * Helps detect data drifts and skew

---

### 11.3.5. NoSQL Data Store

**Static Reference Features**
* Collected, prepared, and stored in NoSQL databases optimized for low-latency singleton read operations
* Examples: Memorystore, Datastore, Bigtable
    * **Use Cases**:
        - Real-time bidding with submillisecond retrieval time
        - Media and gaming applications using precomputed predictions
        - Product recommendation systems based on logged-in user information

---

## 11.4. Automation and Orchestration

**Machine Learning Workflows**

* Machine learning projects follow specific phases:
	+ Data collection
	+ Data preprocessing
	+ Model training and refinement
	+ Evaluation
	+ Deployment to production
* _Orchestration_ of pipeline steps is crucial for integration in a production environment.
* **Automation** of pipelines enables continuous model training.

---

### 11.4.1. Use Vertex AI Pipelines to Orchestrate the ML Workflow

### Vertex AI Pipelines Summary
* **Serverless ML Workflow Orchestration**: Automate, monitor, and govern your ML systems using Vertex AI Pipelines.
* **Artifact Storage**: Store workflow artifacts in Vertex ML Metadata for lineage analysis.
* **Multi-Frame Workload Support**: Run pipelines built with Kubeflow Pipelines SDK or TensorFlow Extended.

---

### 11.4.2. Use Kubeflow Pipelines for Flexible Pipeline Construction

### Kubeflow Pipelines Summary

* **Kubeflow Pipelines**: An open source Kubernetes framework for developing and running portable ML workloads.
    * Composes, orchestrates, and automates ML systems
    * Deploys to local, on-premises, or cloud environments (e.g. Google Cloud)
* _Flexible architecture_ using simple code to construct pipelines
* Integrates with Google Cloud services like Vertex AI (AutoML) and Cloud SQL for metadata storage

---

### 11.4.3. Use TensorFlow Extended SDK to Leverage Pre‐built Components for Common Steps

* **Summary**: TensorFlow and TFX provide tools for building machine learning workflows, with TensorFlow Extended SDK recommended for existing TensorFlow users.
    * **Key Benefits**:
        - For existing TensorFlow users
        - Ideal for structured and textual data
        - Suitable for large datasets
    * _Recommended Tools_: 
        - TensorFlow
        - TFX Extended SDK

---

### 11.4.4. When to Use Which Pipeline

* **Vertex AI Pipelines**: Run Kubeflow Pipelines v1.8.9+ or TensorFlow Extended v0.30.0+
* *_Why TFX_*: Recommended for large-scale data processing due to its support for distributed processing backends
* *_Orchestrators like Kubeflow_*: Provide GUIs and easy configuration, operation, monitoring, and maintenance of ML pipelines

---

## 11.5. Serving

**Types of Machine Learning Predictions**

* *_Offline Prediction_*: The model makes predictions on a dataset without real-time data.
* *_Online Prediction_*: The model makes predictions on new, incoming data as it arrives.
* **Real-time deployment is not mentioned, but it's worth noting that online prediction may involve continuous training and updating of the model.**

---

### 11.5.1. Offline or Batch Prediction

**Offline Batch Prediction**
* Predicts data in batches, ideal for large-scale tasks like recommendations and demand forecasting
* Runs using trained model on batches of data stored in BigQuery or Google Cloud Storage

---

### 11.5.2. Online Prediction

**Real-Time Prediction Overview**
* _Types of Online Predictions_:
  + Synchronous: caller waits for prediction before proceeding
  + Asynchronous: model generates predictions without direct query from end-user
* _Key Considerations_:
  + Minimizing latency to serve prediction in real-time use cases

---

## 11.6. Summary

**Summary of Best Practices for ML on GCP**
* Design reliable, scalable, and highly available ML solutions
* Choose the right service from the GCP AI/ML stack
* Optimize data storage with submillisecond and millisecond latency options (e.g. NoSQL)

---

## 11.7. Exam Essentials

* **Scalable ML Solutions**: Design reliable, scalable, and highly available ML solutions using Google Cloud AI/ML services.
    * Choose an appropriate ML service based on the use case and expertise with ML.
    * Understand data collection, management, and various types of data stores for storing data for different ML use cases.
* **Deployment Best Practices**
    * Implement automation and orchestration for pipeline management.
    * Understand best practices for deploying models, including batch vs. real-time prediction and managing latency.

---

# 12. Chapter 6Building Secure ML Pipelines

---

## 12.1. Building Secure ML Systems

* **Cloud Security Measures**
  * _Encryption of stored data_ (at rest) ensures confidentiality
  * _Encryption of data in transit_ protects data while it is being transferred over the internet

---

### 12.1.1. Encryption at Rest

**Data Encryption in Google Cloud**

* Data stored in Cloud Storage and BigQuery tables is encrypted by default, with Google managing encryption keys.
* Customer-managed encryption keys can be used as well.

* **Encryption Methods:**
	+ Server-side encryption: data arrives already encrypted, undergoes server-side encryption
	+ Client-side encryption: data is not encrypted before sent to Cloud Storage and BigQuery
* Hashes are available for integrity checking: CRC32C and MD5.

---

### 12.1.2. Encryption in Transit

* _Google Cloud uses Transport Layer Security (TLS)_ to protect data during internet transfers.
* TLS ensures secure communication between users and servers.
* This security measure safeguards user data from unauthorized access.

---

### 12.1.3. Encryption in Use

### Data Protection in Use

* **Confidential Computing**: Encrypts data in memory to prevent compromise.
* Protects sensitive information while it's being processed
* Can be used with Confidential VMs and GKE Nodes.

---

## 12.2. Identity and Access Management

### Identity and Access Management (IAM) in Google Cloud
* **Vertex AI IAM**: Manages access to data and resources using project-level and resource-level roles.
* **Roles**:
  * Predefined roles: Grant related permissions at the project level (e.g., Vertex AI Administrator, Vertex AI User)
  * Custom roles: Allow for specific permission sets creation and assignment
* **Resource-level policies**: Grant access to specific resources only (currently supported for Vertex AI Feature Store and entity type resources)

---

### 12.2.1. IAM Permissions for Vertex AI Workbench

**Vertex AI Workbench Summary**
* _Google Cloud Platform (GCP) offers Vertex AI Workbench, a data science service that leverages JupyterLab_
* Two types of notebooks: user-managed and managed
	+ User-managed: highly customizable, but requires more setup and management time
	+ Managed: Google Cloud-managed, with advantages including integration with Cloud Storage and BigQuery, and automatic shutdown
* Access modes: Single User Only and Service Account, with permissions for managing notebooks

---

### 12.2.2. Securing a Network with Vertex AI

**Understanding Shared Responsibility and Shared Fate Models**
===========================================================

* **Key Points**:
	+ **Shared Responsibility Model**: Google Cloud provider is responsible for monitoring security threats, while end users are responsible for protecting their data.
	+ _Shared Fate Model_ aims to improve security through continuous partnership between cloud provider and customer.
* 
* **Shared Fate Components**:
  * Help getting started: Secure blueprints with security recommendations
  * Risk protection program: Assured workloads and governance

---

## 12.3. Privacy Implications of Data Usage and Collection

### Sensitive Data in Google Cloud

* **Personal Identifiable Information (PII)**:
  - Includes name, address, SSN, date of birth, financial info, passport number, phone numbers, email addresses
* _Protected Health Information (PHI)_ is handled under the HIPAA Privacy Rule, which balances individual rights with necessary disclosures for patient care.
* Google Cloud provides strategies to deal with sensitive data.

---

### 12.3.1. Google Cloud Data Loss Prevention

### **Google Cloud Data Loss Prevention (DLP) API**

* **Overview**: Removes identifying information from sensitive data, such as PII.
	+ Uses techniques like masking, tokenization, encryption, and bucketing.
* **Key Concepts**:
	+ **Data Profiling**: Identifies sensitive and high-risk data across organization.
	+ **Risk Analysis**: Computes re-identification risk metrics to determine effective de-identification strategy.
	+ **Inspection (Jobs and Triggers)**: Runs jobs to scan content for sensitive data or calculate re-identification risk.

---

### 12.3.2. Google Cloud Healthcare API for PHI Identification

### HIPAA PHI De-Identification in Google Cloud Healthcare API

* The Google Cloud Healthcare API provides a de-identify operation to remove Protected Health Information (PHI) from healthcare data.
* The API detects sensitive data, such as protected PHI, and applies de-identification transformations to mask or obscure the data.
* _De-identified health information is no longer considered protected under HIPAA Privacy Rule._

---

### 12.3.3. Best Practices for Removing Sensitive Data

### Strategies for Handling Sensitive Data

* **Structured Data**: Create a view that restricts access to sensitive columns in structured datasets.
* **Unstructured Content**: Use Cloud DLP and NLP APIs to identify and mask sensitive data, such as email and location.
    * _Use techniques like PCA or dimension-reducing methods to combine features and reduce the impact of sensitive data._
* **Numerical Fields**:
  - Zero out the last octet of IP addresses
  - Bin numeric quantities, such as age and birthdays, into ranges
  - Coarsen zip codes to include only the first three digits

---

## 12.4. Summary

* **Data Security in Machine Learning**
  * Best practices for encrypting data at rest and transit
  * Managing access with IAM for Vertex AI Workbench
  * Secure ML development techniques (federated learning, differential privacy)
* _Managing PII and PHI Data_
  * Using Cloud DLP and Healthcare APIs
  * Scaling PII identification and de-identification on large datasets

---

## 12.5. Exam Essentials

**Secure Machine Learning on Google Cloud**
* Understand encryption at rest and in transit for Cloud Storage and BigQuery
* Set up IAM roles and network security for Vertex AI Workbench
* Learn about differential privacy, federated learning, and tokenization to protect ML models and data
* Use the Data Loss Prevention (DLP) API to identify and mask PII-type data
* Utilize the Google Cloud Healthcare API to mask PHI-type data

---

# 13. Chapter 7Model Building

---

## 13.1. Choice of Framework and Model Parallelism

* _Modern deep learning models require larger datasets and increased computational power_ 
* **Multinode training is necessary for complex models on large datasets**
* **Data and model parallelism are used to distribute computations across multiple nodes**

---

### 13.1.1. Data Parallelism

* **Data Parallelism**: Split dataset across multiple GPUs or nodes, using same parameters for forward propagation
    * _Batch is sent to each node, gradient computed and sent back to main node_
    * **Synchronous/Asynchronous Strategies**: Distributed training approaches with varying computation coordination

---

### 13.1.2. Model Parallelism

### Model Parallelism
* **Split model into parts**: Partition each model into separate parts, placing each part on a GPU.
* **Increase accuracy**: Benefits include training larger models that exceed single-GPU memory limits.
* _Overcomes GPU memory limitations_

---

## 13.2. Modeling Techniques

Let's go over some basic terminology in neural networks that you might see in exam questions.

---

### 13.2.1. Artificial Neural Network

### **Artificial Neural Networks (ANNs)**

* _Simplest type of neural network_
* One hidden layer
* Feedforward architecture (_classic example_)
* Primarily used for supervised learning on numerical and structured data, such as regression problems

---

### 13.2.2. Deep Neural Network (DNN)

### **Deep Neural Networks (DNNs)**

* _Definition:_ ANNs with multiple hidden layers
* _Qualifying feature:_ At least two layers between input and output

---

### 13.2.3. Convolutional Neural Network

**What is a Convolutional Neural Network (CNN)?**

* A type of Deep Neural Network (DNN) specifically designed for image inputs
* Primarily used for **image classification tasks**
* Can also be applied to various other tasks requiring image processing

---

### 13.2.4. Recurrent Neural Network

**Recurrent Neural Networks (RNNs)**
* _Process sequences of data_
* Effective for natural language processing and time-series forecasting
* Popular type: Long Short-Term Memory (LSTM) networks

**Training a Neural Network**
* Loss function: measures prediction error (e.g., 0 for perfect, > 0 otherwise)
* Gradients: calculated from loss function
* **Gradients Update Weights**: essential for training

---

### 13.2.5. What Loss Function to Use

**Activation Function and Loss Function**

* **Binary classification**: Sigmoid activation with Binary cross-entropy or Categorical hinge loss
* **Multiclass classification**: Softmax activation with:
	+ Categorical cross-entropy for one-hot encoded data
	+ Sparse categorical cross-entropy for mutually exclusive classes

---

### 13.2.6. Gradient Descent

* **Gradient Descent Algorithm**:
    * Calculates the gradient of the loss curve at the starting point.
    * Takes a step in the direction of the negative gradient to minimize loss.
* *_Key Idea_*: Reducing loss by moving against the steepest increase direction.

---

### 13.2.7. Learning Rate

* **Gradient Descent**: The learning rate determines how much to adjust the current point based on the gradient vector's magnitude.
    * _Learning Rate (α) = scalar value multiplying the gradient_
    * _Determines step size in each iteration_

---

### 13.2.8. Batch

* **Batch Size**: The total number of examples used to calculate the gradient in a single iteration.
* _A small batch size can lead to slow convergence_, while a very large batch size can be computationally expensive.
* **Batch Considerations**: Using the entire dataset as a batch may not be efficient; instead, consider using smaller batches or mini-batches.

---

### 13.2.9. Batch Size

* **Batch Size**: The number of examples in a batch
* Typical batch sizes: 
  * Mini-batch: 10-1,000
  * SGD: 1
* Dynamic batch size allowed with TensorFlow

---

### 13.2.10. Epoch

* **Definition**: An epoch is a single iteration for training a neural network with all available training data.
* **Key Points**:
	+ All data is used exactly once in an epoch
	+ A forward pass and backward pass are counted as one pass per epoch
	+ Epochs are composed of multiple batches

---

### 13.2.11. Hyperparameters

### Hyperparameters for Machine Learning Model Training

* **Hyperparameter Tuning**: Adjusting loss, learning rate, batch size, and epoch to optimize training.
	+ _Learning Rate_: Balancing speed and convergence
	+ **Batch Size** affects training time, with larger batches potentially slowing down computation
	+ **Epochs**: Determining number of iterations without overfitting

---

## 13.3. Transfer Learning

* **Transfer Learning**: Using knowledge gained from one problem to solve a related but different problem
    * In deep learning, a neural network is first trained on a similar problem and then used as a starting point for training on the target problem
        * Enables faster development of models with limited data

---

## 13.4. Semi‐supervised Learning

**Semi-Supervised Learning**
* Machine learning approach combining labeled and unlabeled data
* Involves a small number of labeled examples and large number of unlabeled examples
* Falls between unsupervised and supervised learning.

---

### 13.4.1. When You Need Semi‐supervised Learning

### Semi-Supervised Learning Techniques
* **Increase dataset size** without additional resources to improve model accuracy.
* **Use labeled data** you know about and unlabeled data to train a semi-supervised model.
* _Limitedly trustworthy results_ compared to traditional supervised techniques due to potential labeling errors.

---

### 13.4.2. Limitations of SSL

* **Semi-supervised learning** uses _limited_ labeled data and large amounts of unlabeled data for good results in classification tasks.
* However, its effectiveness depends on the _representativeness_ of the labeled data to the overall distribution.
* Poorly chosen labeled data can lead to poor performance.

---

## 13.5. Data Augmentation

* **Data Augmentation**: A technique used to increase the size of a dataset by applying minor alterations (flips, translations, rotations) to existing images, creating synthetic modified data.
* _Purpose:_ To train neural networks with limited data or to add more relevant examples to an existing dataset.
* **Types:** Offline augmentation: applied before training; online augmentation: applied during training.

---

### 13.5.1. Offline Augmentation

### Offline Augmentation
* _Preprocesses data beforehand_
* _Increases dataset size proportional to transformation count_

---

### 13.5.2. Online Augmentation

### **Data Augmentation**

* Performs transformations on mini-batches before feeding them to machine learning models
* Preferred for large datasets due to efficiency and effectiveness
* Examples:
	+ _Image augmentation techniques_:
		- Flip
		- Rotate
		- Scale
	+ Adding Gaussian noise to enhance learning capability

---

## 13.6. Model Generalization and Strategies to Handle Overfitting and Underfitting

**Key Concepts:**

* **Bias**: Difference between average prediction and correct value (error rate of training data)
	+ High bias: Model oversimplifies, pays little attention to data
* **Variance**: Error rate of testing data (high variance = high error on test data)
	+ Low variance: Model generalizes well on unseen data

---

### 13.6.1. Bias Variance Trade‐Off

* **Balancing Model Complexity**: There is a trade-off between bias and variance in machine learning models.
* **Overfitting and Underfitting**:
  * _Underfitting_: Insufficient capacity leads to high bias, low variance; increasing capacity can fix.
  * _Overfitting_: Excessive capacity leads to high variance, low bias; requires specialized techniques to address.

---

### 13.6.2. Underfitting

**Underfit Model**
An underfit model fails to learn a problem, leading to poor performance on training and testing datasets.

* High bias, low variance
* Causes:
	+ Inadequate training data cleaning
	+ High model bias
* Solutions:
	+ Increase model complexity
	+ Improve feature engineering

---

### 13.6.3. Overfitting

### **Overfitting**

* _What is Overfitting?_
An overfit model has low bias and high variance, causing poor performance on new examples.
* **Causes of Overfitting**
	+ Learning training data too well
	+ Insufficient training examples
* **Methods to Avoid Overfitting**
	1. **Regularization Techniques**
	2. _Probabilistic Model Pruning (Dropout)_
	3. **Early Stopping**: Stop training when model performance degrades on validation set

---

### 13.6.4. Regularization

### **Regularization Techniques**

* _L1 Regularization_: Combats overfitting by shrinking parameters toward 0, making some features obsolete.
* _L2 Regularization_: Forces weights to be small but not exactly 0, improving generalization in linear models.

### **Common Issues and Solutions**

* _Exploding Gradients_:
	+ Prevent with Batch Normalization and lower learning rates
* _Dead ReLU Units_:
	+ Prevent with lower learning rates
* _Vanishing Gradients_:
	+ Help with ReLU activation function
* _Losses are not improving, try:_
	+ Increase depth and width of neural network
	+ Decrease learning rate

---

## 13.7. Summary

* **Model Parallelism and Data Parallelism**: Techniques for training neural networks in parallel using models and data
* **Neural Network Training Concepts**:
	+ Gradient Descent: optimization algorithm for neural network training
	+ Hyperparameters: important parameters (e.g., learning rate, batch size) affecting training
* **Additional Topics**:
	+ Transfer Learning: advantages of reusing pre-trained models on new tasks
	+ Semi-Supervised Learning: limitations and applications
	+ Data Augmentation: techniques for increasing dataset size using rotation, flipping, and online/offline augmentation

---

## 13.8. Exam Essentials

* **Overview**: Learn about framework parallelism, multinode training, distributed TensorFlow training, modeling techniques, hyperparameter tuning, transfer learning, semi-supervised learning, data augmentation, model generalization, and regularization.
    * _Key concepts_: gradient descent, batch size, epoch, loss functions (sparse cross-entropy, categorical cross-entropy), L1/L2 regularization
    * **Strategies**: tune hyperparameters, use transfer learning to overcome limited data, apply semi-supervised learning and data augmentation for improved generalization.

---

# 14. Chapter 8Model Training and Hyperparameter Tuning

---

## 14.1. Ingestion of Various File Types into Training

**Data Types for Training**

* _Structured data_ (e.g., tables from an on-premise database or CSV files)
* _Semi-structured data_ (e.g., PDFs, JSON files)
* _Unstructured data_ (e.g., chats, emails, audio, images, videos)

**Data Sources and Sizes**

* Data can be batch or real-time streaming
* Can range from small to petabyte scale

**Data Preparation**
Before training, data must be cleaned and transformed.

---

### 14.1.1. Collect

**Google Cloud Services for Batch and Streaming Data Collection**

* **Pub/Sub and Pub/Sub Lite**: Scalable serverless messaging and real-time analytics services with integration to processing services (Dataflow) and analytics services (BigQuery).
* **Datastream**: Serverless change data capture and replication service for synchronizing data across heterogeneous databases and applications.
* **BigQuery Data Transfer Service**: Automates loading of data from various sources, including external cloud storage and SaaS apps, into BigQuery.

---

### 14.1.2. Process

* **Data Processing Tools**
* * _Data transformation and cleaning_
* * _Data integration and aggregation_
* *_Data visualization and feature engineering_*

---

### 14.1.3. Store and Analyze

### Data Storage for Machine Learning on GCP
* **Use scalable storage solutions**: BigQuery, Google Cloud Storage, Vertex AI, and Feature Store to store different types of data.
* _Avoid storing data in block storage like NFS or VMs, and read from databases directly._
* **Store large files (>100 MB) in container formats**: Cloud Storage for image, video, audio, and unstructured data; sharded TFRecord files for TensorFlow workloads.

---

## 14.2. Developing Models in Vertex AI Workbench by Using Common Frameworks

**Creating a Jupyter Notebook Environment with Vertex AI Workbench** =========================================================== ### Overview * **Vertex AI Workbench**: A Jupyter Notebook-based development environment for data science workflows. * Two types of notebooks: Managed and User-Managed Notebooks. ### Key Differences between Managed and User-Managed Notebooks --------------------------------------------------------- * **Managed Notebook**: * Automated shutdown for idle instances * UI integration with Cloud Storage and BigQuery * Automated notebook runs (using Cloud Scheduler) * Custom containers * Dataproc or Serverless Spark integration * Preinstalled frameworks (e.g., TensorFlow, PyTorch, R, Spark, Python) * **User-Managed Notebook**: * More control and fewer features compared

---

to Managed Notebooks * No automated shutdown for idle instances * Limited UI integration with Cloud Storage and BigQuery * Custom containers * Dataproc or Serverless Spark integration (private preview) * Preinstalled frameworks (e.g., TensorFlow, R, PyTorch, Spark, Python)

---
### 14.2.1. Creating a Managed Notebook

### Enabling and Creating a Managed Notebook in Vertex AI

* Enable all APIs on the GCP Console > Vertex AI > Workbench.
* Create a new notebook using the default settings.
* Wait for the notebook to be created, then click "Open JupyterLab" to access your managed environment.

* _Managed notebook instances have two disks: one for booting and one for data._
* Upgrade options are available; this process upgrades the boot disk while preserving data on the data disk.

---

### 14.2.2. Exploring Managed JupyterLab Features

### **Getting Started with JupyterLab**

* Opens to a screen showing available frameworks, including Serverless Spark and PySpark.
* Finds tutorials notebooks in the `tutorials` folder for model building and training.
* Includes a terminal option for running command-line commands on the entire notebook.

---

### 14.2.3. Data Integration

* **Load Data**: Click the Browse GCS icon to access cloud storage folders.
* *_Navigation_*: _Figure 8.7 illustrates the process._
* *_Functionality_*: Load data from cloud storage into your managed notebook.

---

### 14.2.4. BigQuery Integration

* _Access BigQuery Data_:
    * Click the BigQuery icon on the left
    * Get data from your BigQuery tables

---

### 14.2.5. Ability to Scale the Compute Up or Down

* _Modify Jupyter Environment Hardware_
    * Click on "n1-Standard-4" to enable hardware modification
    * Attach GPU to instance within the environment

---

### 14.2.6. Git Integration for Team Collaboration

* To access project collaboration settings, click the left navigation branch icon.
    * Integrate an existing Git repository or clone a new one using the terminal with `git clone <your-repo-name>`.
        *_Alternative method:_* Using the integrated Git settings screen.

---

### 14.2.7. Schedule or Execute a Notebook Code

### **Executing Python Code in Google Colab**

* Run cells manually by clicking the triangle black arrow
* Execute notebooks automatically using the "Execute" feature
* Scheduling allows setting up Vertex AI training jobs or deploying Prediction endpoints without leaving Jupyter

---

### 14.2.8. Creating a User‐Managed Notebook

### User-Managed Notebooks in TensorFlow

* **Choose a framework**: Select from options like Python 3, TensorFlow, R, JAX, Kaggle, PyTorch during notebook creation.
* _No GPU required_: Create a user-managed notebook without GPUs and explore advanced networking options with shared VPCs.
* **Key benefits**: Git integration, terminal access, and flexibility to trigger Vertex AI training or run predictions using Vertex AI Python SDKs.

---

## 14.3. Training a Model as a Job in Different Environments

**Vertex AI Training Types**
* *_AutoML_*: Minimal technical effort for model creation and training
* *_Custom Training_*: Full control over training application functionality to target specific objectives and algorithms

---

### 14.3.1. Training Workflow with Vertex AI

**Vertex AI Training Options**

* _Training Pipelines_: Automate AutoML or custom model training with workflow orchestration.
	+ Create custom-trained models with pipeline steps like dataset addition and prediction serving.
* Custom Jobs: Define how Vertex AI runs custom training code, including worker pools and machine types.
* Hyperparameter Tuning Jobs: Optimize hyperparameters for custom-trained models.

Note: Not supported by AutoML models.

---

### 14.3.2. Training Dataset Options in Vertex AI

### Data Storage Options for Training

* **No Managed Dataset**: Use data stored in Google Cloud Storage or BigQuery
	+ Benefits include easier data management, automatic dataset splitting, and high-throughput access to remote files
* _Managed Dataset_: Preferred option for training on Vertex AI, offering advantages such as:
	+ Centralized dataset management
	+ Automated labeling and task creation
	+ Governance and iterative development tracking

---

### 14.3.3. Pre‐built Containers

### **Prebuilding a Training Job with Vertex AI**

* To set up a training job with a prebuilt container, you need to organize your code according to the application structure.
* Upload your training code as a Python source distribution to a Cloud Storage bucket before starting training.

    * Create a `setup.py` file in the root folder and specify any standard dependencies or libraries not in the prebuilt container.
    * Use the `sdist` command to create a source distribution (e.g., `python setup.py sdist --formats=gztar,zip`)

---

### 14.3.4. Custom Containers

### **Custom Container Benefits and Usage**
* _Faster start-up time_ and _use of preferred ML framework_
* *_Extended support for distributed training_* using any ML framework
* *_Use the newest version_* of an ML framework with custom containers

---

### 14.3.5. Distributed Training

**Distributed Training with Vertex AI**
* To run a distributed training job with Vertex AI, specify multiple machines (nodes) and allocate resources accordingly.
* A group of replicas with the same configuration is called a **worker pool**.
* Use an ML framework that supports distributed training and define custom worker pools to configure any custom training job.

---

## 14.4. Hyperparameter Tuning

**Hyperparameters**
* Parameters of the training algorithm not learned directly from the data
* Examples include learning rate, regularization strength, batch size
* Must be set before training can begin

---

### 14.4.1. Why Hyperparameters Are Important

*_Hyperparameter Selection_*
* **Importance**: Hyperparameters heavily influence neural network behavior
* *_Challenges_*:
	+ Efficiently searching hyperparameter space
	+ Managing large sets of experiments
* *_Solutions_*:
	+ Algorithms like Grid Search, Random Search, and Bayesian Optimization to optimize hyperparameter search

---

### 14.4.2. Techniques to Speed Up Hyperparameter Optimization

### Hyperparameter Optimization Techniques

* **Speed up hyperparameter optimization**:
	+ Use a simple validation set instead of cross-validation (factor ~k)
	+ Parallelize across multiple machines with distributed training (factor ~n)
* **Reduce computation overhead**: 
	+ Pre-compute and cache results
	+ Decrease number of hyperparameter values to consider

---

### 14.4.3. How Vertex AI Hyperparameter Tuning Works

* **Hyperparameter Tuning**
* Uses multiple trials with specified hyperparameters and target variables to optimize performance.
* Configured using `gcloud CLI` or custom code, requiring communication between training application and Vertex AI.

**Configuring Hyperparameter Tuning Jobs**

* Create a YAML file (`config.yaml`) specifying:
  * **Metric ID**: name of the metric to optimize
  * **Goal**: maximize or minimize the metric value
  * **Hyperparameter ID**: name of the hyperparameter to tune
  * **Container specifications**: image URI, machine type, and replica count for training.

---

### 14.4.4. Vertex AI Vizier

**Vertex AI Vizier**
### A Black-Box Optimization Service

* **Key Criteria:**
	+ No known objective function to evaluate
	+ Too costly to evaluate using objective function due to system complexity
* **Use Cases:**
	+ Hyperparameter tuning for complex ML models (e.g., neural network recommendation engine)
	+ Other optimization tasks, such as model parameter tuning and system evaluation

---

## 14.5. Tracking Metrics During Training

* **Debugging Machine Learning Model Metrics**
  * Tracking and debugging machine learning model metrics using tools like:
    * Interactive shell
    * TensorFlow Profiler
    * What-If Tool

---

### 14.5.1. Interactive Shell

**Interactive Shell in Vertex AI**
* Enabling an interactive shell allows for debugging and troubleshooting of training code and Vertex AI configuration
* Accessible only while job is in RUNNING state
* Losing access occurs when job finishes or changes to COMPLETED state

---

### 14.5.2. TensorFlow Profiler

### Vertex AI TensorBoard Profiler
* **Overview**: enterprise-ready managed version of TensorBoard for monitoring and optimizing model training performance.
* *Key Benefits*: Pinpoint and fix performance bottlenecks to train models faster and cheaper.
* *_Access Methods_*: 
  + From custom jobs page
  + From experiments page

---

### 14.5.3. What‐If Tool

**Using the What-If Tool**
* Access interactive dashboard for AI Platform Prediction models
* Integrates with TensorBoard, Jupyter Notebooks, Colab notebooks, and JupyterHub
* Preinstalled on Vertex AI Workbench user-managed notebooks and TensorFlow instances
### **How to use:**

* Install `witwidget` library
* Configure `WitConfigBuilder` for model inspection or comparison
* Pass `config_builder` to `WitWidget` with display height

---

## 14.6. Retraining/Redeployment Evaluation

### Model Performance Changes Over Time

Machine learning models degrade over time due to:
* *_Data Drift_*: Shifts in data distribution
* *_Concept Drift_*: Change in the underlying relationships between variables
This leads to decreased model performance, requiring periodic retraining or updating.

---

### 14.6.1. Data Drift

**Data Drift**: Change in statistical distribution of production data from baseline used to train models
* **Detection Methods**:
  * Examine feature distribution
  * Check correlation between features
  * Monitor data schema against baseline
* _Causes_: Changes in input data, such as unit changes.

---

### 14.6.2. Concept Drift

**_Concept Drift_**
Change in statistical properties of the target variable over time.

* **Cause**: Shifts in user sentiment or behavior.
* **Effect**: Deployed model accuracy degradation.
* **Solution**: Monitor deployed models using Vertex AI Model Monitoring.

---

### 14.6.3. When Should a Model Be Retrained?

**Retraining Strategies**
* **Periodic Training**: Retrain model at set intervals (e.g., weekly, monthly, yearly) based on updated training data.
* *_Performance-Based Trigger_*: Automatically trigger retraining if model performance falls below a specified threshold.
* **Data Changes Trigger**: Re-train model when data drift occurs in production.

---

## 14.7. Unit Testing for Model Training and Serving

**Testing Machine Learning Models**
* *Unit Testing*: testing individual units of code to ensure they function as expected.
* *Model Testing*: checking explicit behaviors, such as output shapes and ranges, and ensuring model performance with untrained parameters.
* Key Tests:
  - Verifying model output aligns with labels
  - Ensuring output ranges match expectations
  - Checking single gradient step decreases loss

---

### 14.7.1. Testing for Updates in API Calls

* **Testing API Updates**: Use a _unit test_ to simulate random input data and perform a single step of gradient descent to check for runtime errors.
    * This approach is more efficient than retraining the model, while still ensuring updates are thoroughly tested.
    * _Reduces resource requirements_ compared to full-scale testing.

---

### 14.7.2. Testing for Algorithmic Correctness

### Algorithmic Correctness Checklist

* **Train Model**: Verify that loss decreases after initial iterations.
* _Regularization is disabled_ to ensure memorization of training data.
* Test specific subcomputations for correctness, such as:
	+ Part of CNN runs once per element of input data.

---

## 14.8. Summary

**File Ingestion and Processing for AI/ML Workloads**

* File ingestion: Collect, process, store, and analyze files
* Services:
	+ Pub/Sub and Pub/Sub Lite for real-time data collection
	+ BigQuery Data Transfer Service and Datastream for data migration
* Model training:
	+ Vertex AI training with frameworks: scikit-learn, TensorFlow, PyTorch, XGBoost
	+ Prebuilt and custom containers for model deployment
* Testing and validation:
	+ Unit testing and model validation for machine learning
* Hyperparameter tuning:
	+ Search algorithms and Vertex AI Vizier

---

## 14.9. Exam Essentials

**Ingesting and Processing Data for AI/ML Workloads**

* **File Ingestion**: Understand various file types (structured, unstructured, semi-structured) and their storage in Google Cloud.
* **Data Analytics Platforms**: Use Pub/Sub, BigQuery, and other services to collect, process, store, and analyze data.
* **Vertex AI Workbench**: Create user-managed or managed notebooks, train models using various frameworks (e.g. scikit-learn, TensorFlow), and set up distributed training.

---

# 15. Chapter 9Model Explainability on Vertex AI

---

## 15.1. Model Explainability on Vertex AI

* _Explainability in ML is crucial when predictions have significant business outcomes_
* **Model developers must provide explanations for critical predictions**, such as loan approvals or medication dosages.
* _Gaining visibility into the training process is essential to develop human-explainable ML models_.

---

### 15.1.1. Explainable AI

**Explainability in Machine Learning**
=====================================

* **Definition**: The extent to which an ML or deep learning system can be explained in human terms.
* **Types**:
    * **Global Explainability**: Makes the overall ML model transparent and comprehensive.
    * **Local Explainability**: Explains individual predictions.
* **Benefits**:
    * Builds trust and improves adoption
    * Increases comfort level of consumers with model predictions

---

### 15.1.2. Interpretability and Explainability

* **Key Difference**: 
    * _Interpretability_ refers to associating a cause to an effect
    * _Explainability_ explains the model's internal workings justifying its results

---

### 15.1.3. Feature Importance

* **Feature Importance**: Measures the value of each feature in predicting an outcome.
    * **Benefits**:
        * _Variable selection_: Eliminate non-contributing variables to save compute, infrastructure, and training time.
        * **Target/label leakage prevention**: Avoid adding target variable to training dataset to prevent biased models.

---

### 15.1.4. Vertex Explainable AI

### **Vertex Explainable AI**

* Integrates feature attributions into Vertex AI for classification and regression tasks
* Provides insights into model's outputs, including how much each feature contributed to the predicted result
* Supported services:
	+ AutoML image models (classification only)
	+ AutoML tabular models (classification and regression only)

---

### 15.1.5. Data Bias and Fairness

**Data Bias and Fairness**
* _Biased data can lead to skewed outcomes and unfair models_
* _Machine learning models can perpetuate systemic prejudice if biased data is used_
* **Tools for detecting bias and fairness:**
	+ Vertex AI's Explainable AI feature attributions technique
	+ The What-If Tool in AutoML tables with features overview functionality
	+ Open source Language Interpretability Tool for NLP models

---

### 15.1.6. ML Solution Readiness

**ML Solution Readiness**
* _Key Concepts:_ Responsible AI Model Governance and Explainable AI
* Google provides best practices, tools, and references for implementing responsible AI principles in ML models.
* Key aspects of model governance include:
	+ **Human In the Loop**: Reviewing model output for sensitive workloads
	+ **Model Versioning and Data Lineage**: Maintaining model cards for transparent tracking
	+ _Fairness Indicators_: Evaluating models against bias detection tools

---

### 15.1.7. How to Set Up Explanations in the Vertex AI

* **Configuring Models**: For custom-trained models (TensorFlow, Scikit, XGBoost), configure explanations to support Vertex Explainable AI.
* **AutoML Explanations**: No specific configuration required for AutoML tabular classification or regression.
* _**Explaination Methods**: 
  * Online explanations: synchronous requests to the Vertex AI API
  * Batch explanations: asynchronous requests to the Vertex AI API with `generateExplanation` set to true
  * Local kernel explanations: perform within User-Managed Vertex AI Workbench notebook

---

## 15.2. Summary

* **Explainable AI**: Explainable AI (XAI) is a subset of interpretability that provides insights into AI decisions
    * Focuses on feature importance and data bias to improve model fairness and reliability
    * Enables ML solution readiness by providing transparency into AI performance

---

## 15.3. Exam Essentials

### **Explainability on Vertex AI**

* **What is Explainability?**: Understanding how machine learning models make predictions
* **Why is it Important?**: Identifying biases, fairness, and feature importance to build trust in models
* _**Supported Feature Attribution Methods:**_ 
  * Sampled Shapley algorithm
  * Integrated gradients
  * XRAI

---

# 16. Chapter 10Scaling Models in Production

---

## 16.1. Scaling Prediction Service

### Deploying a TensorFlow Model

* **Deploying a Trained TensorFlow Model**
  * A saved model contains trained parameters and computation.
  * It's useful for sharing or deploying with other frameworks (e.g., TensorFlow Lite, TensorFlow.js).
* **Saved Models**
  * Created by calling `tf.saved_model.save()`.
  * Stored as a directory on disk.
  * Includes the protocol buffer `saved_model.pb`.

---

### 16.1.1. TensorFlow Serving

### Setting up TensorFlow Serving
* **Install**: Install TensorFlow Serving with Docker for ease of use.
* **Setup Steps**:
	+ Train and save a model with TensorFlow.
	+ Serve the saved model using TensorFlow Serving.
* _Alternative_: Use a managed TensorFlow prebuilt container on Vertex AI.

---

## 16.2. Serving (Online, Batch, and Caching)

* **Best Practices for Serving Strategy**
	+ Choose between batch prediction and online prediction based on data volume and complexity
	+ Consider scalability and latency requirements when selecting an option

---

### 16.2.1. Real‐Time Static and Dynamic Reference Features

**Real-Time vs. Static Reference Features**

* *_Static Reference Features_*:
	+ Values do not change in real time
	+ Updated in batches
	+ Examples: customer ID, movie ID
	+ Stored in NoSQL databases optimized for singleton lookup operations (e.g., Firestore)
* *_Dynamic Real‐Time Features_*:
	+ Computed on the fly in event-stream processing pipelines
	+ Aggregated values available for specific windows or sessions
	+ Use cases: real-time sensor data, product recommendations, news article suggestions

---

### 16.2.2. Pre‐computing and Caching Prediction

**Offline Pre-Computing for Reduced Latency**

* **Key Concept**: Store pre-computed predictions in a low-latency data store like Memorystore or Datastore.
* *_Benefits_*:
  * Improved online prediction latency
  * No need to call model for online prediction
* *_Challenges_*:
  * Handling high cardinality (many entities) for efficient pre-computation

---

## 16.3. Google Cloud Serving Options

* **Vertex AI Deployment Options**: 
  * Online predictions
  * Batch predictions
* Supported deployment types: 
  * AutoML
  * Custom models

---

### 16.3.1. Online Predictions

**Setup a Real-time Prediction Endpoint**
* Import existing model or deploy custom trained AutoML/Custom Model on Google Cloud.
	+ Verify model file naming conventions match required formats (e.g., `.pb`, `.joblib`, or `.bst`)
	+ Push custom container image to Artifact Registry using Cloud Build (if needed)
* Follow these steps:
  * Deploy model resource to endpoint
  * Make prediction
  * Undeploy model resource if not in use

---

### 16.3.2. Batch Predictions

**Batch Prediction Overview**

* Run a job on model and input data in Google Cloud Storage
* Input data must be formatted according to AutoML or custom model requirements
* Output can be saved in BigQuery table or Cloud Storage bucket

  * _Format Options_
    + JSON Lines: Store instances as JSON Lines file in Cloud Storage bucket
    + TFRecord: Compressed files with gzip, stored in Cloud Storage bucket
    + CSV: Input instance per row, first row header, string values enclosed in double quotes
    + File list: Text file with Cloud Storage URIs to input files

---

## 16.4. Hosting Third‐Party Pipelines (MLflow) on Google Cloud

**Hosting MLflow Pipelines on Google Cloud**

* Use **Google Cloud Vertex AI Experiments** and **PostgreSQL DB** to track experiments.
* Create a **Google Cloud Storage bucket** to store artifacts.
* Run MLFlow using a **Google Cloud plug-in**, or install it on a **Compute Engine instance** or **Kubernetes** with a **Docker environment**.

---

## 16.5. Testing for Target Performance

### Testing Your Model for Performance in Production

* **Verify model stability**: Test that weights and outputs are numerically stable, checking for NaN or null values.
* **Monitor performance over time**: Track model age and performance throughout the ML pipeline to detect potential issues.
* **Use automated testing tools**: Utilize services like Vertex AI's Model Monitoring and Feature Store to simplify testing and debugging.

---

## 16.6. Configuring Triggers and Pipeline Schedules

### Triggering Training or Prediction Jobs on Vertex AI
* **Cloud Scheduler**: Set up a cron job schedule to automate training or prediction jobs.
* **Vertex AI Managed Notebooks**: Execute and schedule jobs using Jupyter Notebook, referencing [https://cloud.google.com/vertex-ai/docs/workbench/managed/schedule-managed-notebooks-run-quickstart](https://cloud.google.com/vertex-ai/docs/workbench/managed/schedule-managed-notebooks-run-quickstart)
* **Cloud Build**: Use for custom training and deployment to Cloud Run, leveraging CI/CD capabilities.

---

## 16.7. Summary

### TF Serving Scaling Prediction Service Summary

* **Overview**: Scalable prediction service using TF Serving
	+ Predict function and SignatureDef for output retrieval
	+ Online serving architectures (static & dynamic) and pre-computing/caching
* **Deployment**: Models deployed via online & batch mode with Vertex AI Prediction and Google Cloud options
* **Production Performance**: Factors causing degradation, including training-serving skew & data quality changes

---

## 16.8. Exam Essentials

**TensorFlow Serving Overview** * _TF Serving is a model serving system that deploys trained TensorFlow models for prediction._ * It supports multiple deployment options, including Docker, and provides scalability features. **Key Concepts** * **Online, Batch, and Caching Scales**: Online serves predictions in real-time, batch serves predictions in batches, while caching improves latency. * _Caching vs. Batch Serving_: Caching is a type of batch serving that stores models in memory for faster access. * **Real-Time Endpoints**: Set up endpoints using Google Cloud Vertex AI Prediction for custom or external models. **Performance and Automation** * Test target performance and understand model degradation issues. * Use Vertex AI services, such as Model Monitoring, to address performance

---

issues. * Configure triggers and schedule pipelines using Cloud Scheduler and Vertex AI managed notebooks scheduler.

---
# 17. Chapter 11Designing ML Training Pipelines

---

## 17.1. Orchestration Frameworks

* **ML Pipeline Orchestration**: An orchestrator manages multiple steps in an ML pipeline, including data cleaning, transformation, and model training.
* **Benefits**:
  * Automates execution of each step
  * Facilitates efficient experimentation and deployment
* **Key Use Cases**:
  * Development phase: automates experiment execution for data scientists
  * Production phase: automates pipeline execution based on schedules or conditions

---

### 17.1.1. Kubeflow Pipelines

### What is Kubeflow Pipelines?

Kubeflow Pipelines is a platform for building, deploying, and managing multistep ML workflows based on Docker containers.

* **Components:**
	+ **User Interface (UI)**: Manage and track experiments, jobs, and runs
	+ **Engine**: Schedules multistep ML workflows
	+ **SDK**: Defines and manipulates pipelines and components
* **Deployment Options:** 
  * Google Cloud on GKE or managed Vertex AI Pipelines
  * On-premises or local systems for testing purposes

---

### 17.1.2. Vertex AI Pipelines

### **Vertex AI Pipelines Overview**

* Run Kubeflow and TensorFlow Extended pipelines serverless with automated infrastructure management
* Provides data lineage, artifact lineage, and monitoring capabilities
* Portable, scalable, and container-based pipeline components

### **Key Features**

* **Data Lineage**: track movement of data over time
* **Artifact Lineage**: understand factors resulting in artifacts (training data, hyperparameters)
* **AutoML Integration**: use Google Cloud pipeline components for feature-rich workflows

---

### 17.1.3. Apache Airflow

* **Apache Airflow**: An open source workflow management platform for data engineering pipelines.
    * Developed by Airbnb in 2014 to manage complex workflows
        * _Built around directed acyclic graphs (DAGs)_
            * Comprises tasks with dependencies and data flows

---

### 17.1.4. Cloud Composer

*_Cloud Composer_* 
* A fully managed workflow orchestration service built on Apache Airflow
* Benefits from Airflow with no installation or management overhead
* Designed for data-driven workflows (ETL/ELT) and batch workloads with low latency

---

### 17.1.5. Comparison of Tools

### Comparison of Orchestration Tools

* **Orchestration Features**
	+ *_Kubeflow Pipelines_*: Orchestrates ML workflows in supported frameworks (TensorFlow, PyTorch, MXNet) using Kubernetes.
	+ *_Vertex AI Pipelines_*: Runs Kubeflow Pipelines with built-in failure management on metrics.
	+ *_Cloud Composer_*: Managed serverless pipeline for Kubeflow Pipelines or TFX Pipelines; supports Apache Airflow workflows.
* **Support and Management**
	+ *_Managed Infrastructure_*: No need to manage infrastructure for Cloud Composer and Vertex AI Pipelines.
	+ *_No Out-of-the-Box Failure Handling_*: Handle failures on metrics as it's not supported out of the box.

---

## 17.2. Identification of Components, Parameters, Triggers, and Compute Needs

* **Triggering an MLOps Pipeline**
    * Automate retraining of ML models with new data.
    * Triggers can include:
        - Availability of new data
        - Model performance degradation
        - Significant changes in statistical data properties

---

### 17.2.1. Schedule the Workflows with Kubeflow Pipelines

### **CI/CD Overview for Kubeflow Pipelines**

* **Kubeflow Pipelines** provides a Python SDK to operate pipelines programmatically.
* **Invocation services**: Cloud Scheduler (on schedule), Pub/Sub and Cloud Functions (responding to events)
* **Integration with other tools**:
	+ Cloud Composer
	+ Cloud Data Fusion
	+ Argo for recurring pipelines
	+ Apache Airflow (general-purpose workflows)

---

### 17.2.2. Schedule Vertex AI Pipelines

**Scheduling Vertex AI Pipeline**

* _Two ways to schedule pipeline execution:_
  * Using **Cloud Scheduler**: schedule with Cloud Function and HTTP trigger
  * Using **Pub/Sub trigger**: schedule with Cloud Function and Pub/Sub topic
* Both methods allow scheduling precompiled pipelines for automated execution

---

## 17.3. System Design with Kubeflow/TFX

**System Design with Kubeflow DSL and TFX**
* Two approaches for designing systems: Kubeflow DSL and TFX
* Comparison and discussion of each approach in separate sections

---

### 17.3.1. System Design with Kubeflow DSL

**Kubeflow Pipelines Overview**
* *Kubeflow Pipelines uses Argo Workflows by default.*
* *Exposes a Python DSL for authoring pipelines.* 
* *Allows simple Python functions as pipeline stages without explicit container creation.*

---

### 17.3.2. System Design with TFX

**TFX Overview**

* _TFX is a Google production-scale machine learning platform based on TensorFlow._
* _Provides configuration framework, shared libraries, and components for building and managing ML workflows._

### TFX Pipeline Components
* **Initial Input**: ExampleGen, StatisticsGen, SchemaGen
* *_ExampleGen ingests and splits input dataset_*
* *_StatisticsGen calculates statistics for the dataset_*
* *_SchemaGen creates a data schema based on statistics_*

---

## 17.4. Hybrid or Multicloud Strategies

**Multicloud and Hybrid Cloud**
### **Concepts**

* **Multicloud**: interconnection between 2+ cloud providers
* **Hybrid Cloud**: private computing environment + public cloud computing environment
* _**Anthos**: hybrid and multicloud cloud modernization platform_


### **Key Features of Anthos**

* Supports hybrid ML development with BigQuery Omni, GKE, and Cloud Run on-premises
* Enables query data without infrastructure management with BigQuery Omni
* Offers speech-to-text AI services on-premises for hybrid AI

---

## 17.5. Summary

**Orchestration for ML Pipelines**
### Tools and When to Use Them
* **Vertex AI Pipelines**: Managed serverless way to run ML workflows
* **Kubeflow**: Cloud-native platform with UI and TensorBoard for workflow automation
    * **Cloud Composer**: Run TFX pipelines on Kubeflow
    * **GKE**: Run Kubeflow Pipelines on-premises with Anthos

### Scheduling and System Design
* Schedule ML workflows using Cloud Build or Cloud Function event triggers
* Use Kubeflow's component-based system design for seamless orchestration
* Visualize components with TensorBoard

---

## 17.6. Exam Essentials

**Orchestration Frameworks for Automating ML Workflows**

* **Key Orchestration Platforms**
	+ Kubeflow Pipelines
	+ Vertex AI Pipelines
	+ Apache Airflow
	+ Cloud Composer
* **Comparison and Contrast**
	+ *Components*: Kubeflow (components, parameters, triggers) vs. TFX (TFX components, parameters)
	+ *Compute Needs*: Kubeflow (GCP, cloud resources), Vertex AI (GCP, cloud resources)
	+ *Scheduling**: Kubeflow (Cloud Build), Vertex AI (Cloud Function event triggers)
* **Integration and Deployment**
	+ Run TFX on Kubeflow and use TFX components and libraries
	+ Use Kubeflow Pipelines to orchestrate ML pipelines with any runtime or orchestrator

---

# 18. Chapter 12Model Monitoring, Tracking, and Auditing Metadata

---

## 18.1. Model Monitoring

### **Model Drift and Its Consequences**

* *_What is model drift?_*: When a machine learning model's performance degrades over time due to changes in the underlying data or environment.
* *_Types of drift_*:
	+ *_Concept drift_*: Changes in the concept or relationship between features and target variables.
	+ *_Data drift_*: Changes in the distribution of input data.
* *_Consequences_*: Models become less accurate, and deployment becomes more challenging.

---

### 18.1.1. Concept Drift

**Concept Drift**: When the relationship between input variables and predicted variables changes over time, it's called concept drift.

* This happens when underlying model assumptions change.
* Examples include email spam where spammers adapt to detection filters.

---

### 18.1.2. Data Drift

* **Data Drift**: Change in input data distribution or schema that can cause a machine learning model to deteriorate over time.
    * _Example:_ Changes in customer demographics, new product labels, or changes in column meaning (e.g., updated medical diagnostic levels).
    * **Model Monitoring:** Continuously evaluating the model with original training metrics to detect and act on data drift.

---

## 18.2. Model Monitoring on Vertex AI

**Model Monitoring in Vertex AI**
### Key Features

* **Training-Serving Skew**: detects differences between training and production data
* **Prediction Drift Detection**: monitors changes in input statistical distribution over time

### Data Types Supported

* _Categorical Features_: limited number of possible values (e.g. color, country, zip code)
* _Numerical Values_: can take any value (e.g. price, speed, distance)

---

### 18.2.1. Drift and Skew Calculation

* **Baseline Distribution Calculation**
	+ Categorical features: Count or percentage of instances per value
	+ Numerical features: Values grouped into equal-sized buckets by count or percentage
* **Distance Measure Calculation**
	+ Categorical features: L-infinity distance between baseline and latest production distribution
	+ Numerical features: Jensen-Shannon divergence between baseline and latest production distribution
* **Anomaly Detection Threshold**: When distance score hits configured threshold, Vertex AI identifies anomaly (skew or drift)

---

### 18.2.2. Input Schemas

* **Input Values**: Part of the payload in prediction requests
* *_Parsing_*: Vertex AI can parse input values with or without schema specification
    * *_Automatic Parsing_*: AutoML models don't require schema, while custom-trained models do.

---

## 18.3. Logging Strategy

**Enabling Prediction Logs in Vertex AI**

* Logging is available for AutoML tabular, image, and custom-trained models.
* Can be enabled during model deployment or endpoint creation.
* Required for regulatory domains like financial verticals for future audits.

---

### 18.3.1. Types of Prediction Logs

* **Log Types**: Three independent log types for prediction nodes
    * *_Node Statistics Log_*: Node-level statistics
    * *_Event Log_*: Event-related data
    * *_Error Log_*: Error information

---

### 18.3.2. Log Settings

**Updating Log Settings for Endpoint**
* _Can be done when creating endpoint or redeploying model_
* **Changing default settings requires redeployment**
* _Consider costs of logging with high QPS (Queries per second)_

---

### 18.3.3. Model Monitoring and Logging

**Restrictions between Model Monitoring and Request-Response Logging**

* **Cannot enable both simultaneously**
* _Enabling model monitoring disables request-response logging_
* Enabling request-response logging cannot be undone after model monitoring is enabled.

---

## 18.4. Model and Dataset Lineage

* **Metadata in ML Experimentation**: A crucial aspect of tracking and comparing different models, detecting degradation, and understanding lineage.
* *_Key benefits_*:
	+ Detects model degradation after deployment
	+ Compares effectiveness of hyperparameters
	+ Tracks lineage of artifacts (datasets, models) for audit purposes

---

### 18.4.1. Vertex ML Metadata

**Vertex ML Metadata Overview**

* _Metadata store_: Top-level container for metadata resources, regional and scoped to a Google Cloud project.
* **Data Model**:
  * _Artifacts_: Entities or data created by/for an ML workflow (e.g., datasets, models, input files).
  * _Contexts_: Groups of artifacts and executions that can be queried (e.g., optimizing hyperparameters).
  * _Executions_: Steps in a machine learning workflow with runtime parameters (e.g., training operations).
  * _Events_: Connect artifacts and executions to track lineage and origin.

---

## 18.5. Vertex AI Experiments

**Vertex AI Experiments Summary**
* **Automated Experiment Tracking**: Track steps, inputs, and outputs of experiments to analyze variations and optimize model performance.
* **Visualization in Google Cloud Console**: View experiment results, slice and dice data, and zoom into specific runs for detailed analysis.
* *_Lineage and Artifact Management_*: Use Vertex ML Metadata to track artifacts and view lineage, enabling more informed exploration and decision-making.

---

## 18.6. Vertex AI Debugging

### Debugging Issues in Vertex AI Training
* Connect to the container where training is running using the interactive Bash shell.
* Verify user permissions, especially for service accounts used by Vertex AI.
* Enable web access API field to `true` for interactive shells.

### Interactive Shell Commands

* Check service account permissions with `gcloud auth compute-projects get-token --project <PROJECT_ID>`
* Visualize Python execution with `py-spy`
* Analyze performance using `perf`

### Note
Make sure to navigate to the URI provided by Vertex AI when initiating a custom training job.

---

## 18.7. Summary

* **Model Monitoring**: Tracking deployed model performance for degradation
* **Logging Strategies**: Various options in Vertex AI
* **Tracking Models**: Lineage tracking with Vertex ML Metadata and Experiments

---

## 18.8. Exam Essentials

### Model Monitoring and Vertex AI Logging
#### Types of Degradation
* **Data Drift**: Changes in input distribution over time
* **Concept Drift**: Changes in relationship between input and output variables

#### Importance of Logging
Logging is crucial for tracking model performance, creating new training data, and accessing metadata.

#### Vertex AI Logging Strategies
* Use logging to monitor models and track deployment performance
* Utilize Vertex ML Metadata for storing and accessing metadata on GCP

---

# 19. Chapter 13Maintaining ML Solutions

---

## 19.1. MLOps Maturity

### MLOps Phases and Automation Levels #### Overview of Machine Learning Workflow * **Data Extraction**: Collect, aggregate, and prepare data for ML process. * **Analysis**: Explore data schema, distributions, and relationships. * **Model Training**: Set up training using input data and predict output. * **Evaluation**: Assess model quality based on predefined metrics. * **Deployment Serving**: Deploy model for online or batch predictions. ### MLOps Automation Levels #### Google's Classification of MLOps Phases * **MLOps Level 0 (Manual Phase)**: Manual model training and validation. * _**MLOps Level 1 (Strategic Automation Phase)**_: Automated data preparation, feature engineering, and strategic model selection. * **_MLOps Level 2 (CI/CD Automation, Transformational Phase)_**: Fully

---

automated MLOps pipeline with continuous integration, continuous deployment.

---
### 19.1.1. MLOps Level 0: Manual/Tactical Phase

* **Experimentation Phase**: Organizations experiment with ML, building proof of concepts and testing AI/ML use cases.
* *Key Activities:*
	+ Developing and training models
	+ Validating business improvement ideas through ML
	+ Using a model registry to manage and deploy trained models* 
* *Output:* Trained model deployed to serve predictions

---

### 19.1.2. MLOps Level 1: Strategic Automation Phase

### MLOps Level 1 (Strategic Phase)

* Organizations have identified business objectives and prioritized Machine Learning (ML) to solve problems.
* This phase involves:
	+ **Automated Pipeline Training**: Continuously trains models using pipelines.
	+ Automated Model Prediction Service: Delivering model predictions with validated data.
	+ New services required, including Feature Store and metadata management.

---

### 19.1.3. MLOps Level 2: CI/CD Automation, Transformational Phase

**Transformational Phase**
* Organizations use AI for innovation and agility
* ML experts work with product teams and across business units
* Data sharing is seamless across silos, and projects are collaborative between groups

---

## 19.2. Retraining and Versioning Models

**Retraining Model Frequency**
* Monitor model performance for degradation
* Collect real data for evaluation and new training datasets
* Determine optimal retraining frequency based on monitoring results

---

### 19.2.1. Triggers for Retraining

### Model Retraining Policies
#### Overview
Model performance degrades over time, triggering retraining is necessary.

#### Policies
* **Absolute Threshold**: Trigger retraining when accuracy falls below a set threshold (e.g., 90%).
* **Rate of Degradation**: Trigger retraining when there's a sudden drop in performance (e.g., >2% in a day).

#### Considerations
* **Training Costs**: Minimize frequent retraining to reduce costs.
* **Deployment Time**: Balance training time with potential degradation in deployed model.

---

### 19.2.2. Versioning Models

**Model Versioning**
================
### Problem Statement

* Multiple models with shared metadata not accessible externally
* API performance and behavior disruption when updating models
* Need for backward compatibility and version selection

### Solution

* **Deploy additional model versions** with version ID for selective access
* Enable REST endpoints for model access, separately and conveniently
* Allow monitoring of deployed model versions for comparison

---

## 19.3. Feature Store

### Feature Engineering Challenges

* _Non-reusable_ features are created ad hoc and not intended to be used by others
* **Governance** becomes complex due to diversity in feature creation methods
* Ad hoc features can't be shared, creating divisions between teams and reducing ML effectiveness

---

### 19.3.1. Solution

**Feast Feature Store**

* Centralized location for storing features and metadata, enabling collaboration between data engineers and ML engineers
* Applies software engineering principles of versioning, documentation, and access control
* Key features:
  * **Fast processing**: supports large feature sets with low latency for real-time prediction and batch access
  * **Scalable**: integrates with Google Cloud services and offers a managed service called Vertex AI Feature Store

---

### 19.3.2. Data Model

### Data Model Summary
#### Key Components:

* **Featurestore**: A container that can have one or more entity types.
* **EntityType**: Represents a certain type of feature, storing similar or related features.
* **Features**: Stored in an EntityType, must be unique and of type String.

#### Hierarchy:
Featurestore → EntityType → Feature

---

### 19.3.3. Ingestion and Serving

**Vertex AI Feature Store**

* Supports both **batch** and **streaming** ingestion
* Can be stored in BigQuery and ingested into the Feature Store
* Ingestion methods:
	+ Batch: used for model training
	+ Online: used for online inference
* Returns data at or before requested time

---

## 19.4. Vertex AI Permissions Model

### **Identity and Access Management (IAM) in ML Pipelines**

* _Define access controls_ to restrict users and applications to perform only necessary actions
* **Least Privilege**: Restrict user and application permissions to minimize risk
* **Auditing and Policy Management**: Enable audit logs and use cloud logging roles; implement policies at every level of the pipeline

---

### 19.4.1. Custom Service Account

* **Use custom service accounts** instead of default ones
    * Grant only necessary permissions to minimize security risks
        * _Access to required resources (e.g., BigQuery, GCS)_ 
        * No unnecessary access to other services or data

---

### 19.4.2. Access Transparency in Vertex AI

* **Access Logs**: required for tracking user and entity activity
    * *_Types_*:
        + Cloud Audit logs (user org activity)
        + Access Transparency logs (Google personnel activity)
    
    * [_Supported Services_](https://cloud.google.com/vertex-ai/docs/general/access-transparency)

---

## 19.5. Common Training and Serving Errors

### Common Errors in Training and Serving

* _Data Corruption_
* _Model Overfitting_
* _Framework Configuration Issues_

---

### 19.5.1. Training Time Errors

* **Common Errors During Training** 
  * _Input data must be properly transformed and encoded_
  * **Tensor Shape Mismatch**: incorrect dimensions in input data
  * *_Out-of-Memory Errors_*: large dataset size exceeds system resources (CPU, GPU)

---

### 19.5.2. Serving Time Errors

* **Serving Time Errors**: occur during deployment, with distinct error types
* Typical errors:
  * _Input Data Issues_: data not transformed or encoded
  * Signature Mismatch: incorrect parameter values passed to functions

---

### 19.5.3. TensorFlow Data Validation

* **TensorFlow Data Validation (TFDV)** 
  * Analyzes training and serving data
  * Computes statistics, infers schema, and detects anomalies to prevent errors

---

### 19.5.4. Vertex AI Debugging Shell

**Vertex AI Debugging Shell**
* Interactive shell for debugging training containers
* Inspect training code and configuration for issues
* Run tracing and profiling tools, analyze GPU utilization and validate IAM permissions.

---

## 19.6. Summary

**MLOps Maintenance**
* Automates training, deployment, and monitoring
* Balances model quality with retraining policy costs
* Addresses data sharing inefficiencies between departments
    * **Feature Stores**: solves data sharing issues by storing and managing reusable features, available in open source software or Vertex AI

---

## 19.7. Exam Essentials

*_MLOps Maturity Levels_*
* Experimental Phase: Basic automation
* Strategic Phase: Limited automation with some CI/CD practices
* Mature Phase: Fully automated CI/CD-inspired architecture

*_Key Concepts_*
* Model Versioning: Managing different versions of a model for retraining and deployment.
* Retraining Triggers: Determining when to trigger new training based on model degradation or time-based schedules.

*_Feature Store_*
* **Benefits:** Shared features across teams, reducing feature engineering costs.
* **Managed Services:** Vertex AI Feature Store (managed), Google Feast (open source).

---

# 20. Chapter 14BigQuery ML

---

## 20.1. BigQuery – Data Access

**Accessing Data in BigQuery**
* **Using Web Console**: Write a SQL query and view results
* **Using Jupyter Notebook**: Run query with `%%bigquery` magic command or Python API
    * _Run query using Python API_
        ```
        from google.cloud import bigquery
        client = bigquery.Client(location="us-central1",project="projectname")
        query = """ SELECT * FROM `projectname.dataset1.table1` LIMIT 10 """
        query_job = client.query(query, location="us-central1")
        df = query_job.to_dataframe()
        ```

---

## 20.2. BigQuery ML Algorithms

* **BigQuery ML**: A serverless service for creating machine learning models using standard SQL queries
    * No need for Python code
    * Train, test, validate, and predict with SQL queries only

---

### 20.2.1. Model Training

**Creating Models in BigQuery ML**
* To create a model, use `CREATE MODEL` statement with options for model type and input label columns.
* Available models include:
	+ Regression (e.g., linear_reg, DNN_REGRESSOR)
	+ Classification (e.g., logistic_reg, DNN_CLASSIFIER)
	+ Clustering (e.g., KMEANS)
	+ Dimensionality reduction (e.g., PCA, AUTOENCODER)
	+ Time-series forecasting (e.g., ARIMA_PLUS)
* Use `CREATE OR REPLACE MODEL` to reuse model names and modify existing models.

---

### 20.2.2. Model Evaluation

### Model Evaluation

* To evaluate a model, use `ML.EVALUATE` with a separate, unseen dataset.
* `_ML.EVALUATE_`: 
    * ```sql
    SELECT * FROM ML.EVALUATE(MODEL projectid.test.creditcard_model1, (SELECT * FROM test.creditcardtable))
```
    * Provides immediate results in the web interface.

---

### 20.2.3. Prediction

**BigQuery ML.PREDICT Function**
* Predicts output from a trained model
* Returns a table with input columns and two new columns: `predicted_<label_column_name>` and `predicted_<label_column_name>_probs`
* Example usage:
  ```
SELECT * FROM ML.PREDICT(
  MODEL `dataset1.creditcard_model1`,
  (SELECT * FROM `dataset1.creditcardpredict` LIMIT 1)
)
```
* Allows selecting only certain columns from the output

---

## 20.3. Explainability in BigQuery ML

**Explainability in BigQuery**

* _Enable explainability during training_:
	+ `enable_global_explain=TRUE`
	+ Use SQL functions like `ML.GLOBAL_EXPLAIN` or `ML.EXPLAIN_PREDICT`
* **Model Types and Explainability Methods**
	+ Linear/Logistic Regression: Shapley values, p-values
	+ Boosted Trees: SHAP, Gini-based feature importance
	+ Deep Neural Network and Wide-and-Deep: Integrated gradients
	+ Time-series decomposition: Arima_PLUS

---

## 20.4. BigQuery ML vs. Vertex AI Tables

**Overview of BigQuery ML and Vertex AI**

* **BigQuery**: Serverless data warehouse for SQL experts
    * Tables, joins, GROUP-BY statements used extensively
    * Custom queries automated using scheduled queries and visualization tools
* **Vertex AI**: Platform for machine learning engineers
    * Familiarity with Kubeflow, Java or Python programming
    * Jupyter Notebooks, Pandas DataFrames, and custom TensorFlow operations used daily

---

## 20.5. Interoperability with Vertex AI

* **Integration Points**: 
    * _Machine Learning Pipeline_
    * Data Ingestion
    * Model Training
    * Model Deployment
    * Data Analysis
    * Monitoring and Evaluation

---

### 20.5.1. Access BigQuery Public Dataset

### **BigQuery Public Datasets**

* _Free access to over 200 public datasets_ through Google Cloud Public Datasets Program
* Storage cost-free, query costs apply
* Access datasets from Vertex AI for machine learning training and data augmentation

---

### 20.5.2. Import BigQuery Data into Vertex AI

**Creating a Vertex AI Dataset from BigQuery**
* Directly provide BigQuery URL as `bq://project.dataset.table_name` to create a dataset
* No need to export and import data, thanks to integration with Vertex AI

---

### 20.5.3. Access BigQuery Data from Vertex AI Workbench Notebooks

* You can interact with BigQuery from a Jupyter Notebook in Vertex AI Workbench
* Directly browse your dataset, run SQL queries, or download data into a Pandas DataFrame
* Useful for exploratory analysis, visualization, and machine learning development

---

### 20.5.4. Analyze Test Prediction Data in BigQuery

**Exporting Model Predictions to BigQuery**
* _Allows for post-hoc analysis of model performance_
* Enables exporting train and test dataset predictions to BigQuery
* Facilitates further analysis using SQL methods

---

### 20.5.5. Export Vertex AI Batch Prediction Results

* **Integration with BigQuery**: Use BigQuery as input data for batch predictions in Vertex AI
* **Bi-directional data transfer**: Send predictions back to BigQuery to store them as a table

---

### 20.5.6. Export BigQuery Models into Vertex AI

* **Exporting Models from BigQuery**: Export your BigQuery ML model to Google Cloud Storage (GCS) and import it into Vertex AI for complete freedom in training and deployment.
* **Model Registration with Vertex AI Model Registry**: Register your BigQuery ML models directly into the Vertex AI Model Registry, eliminating the need to export model files to GCS.
    * Supported Models: Both BigQuery inbuilt and TensorFlow models are supported.

---

## 20.6. BigQuery Design Patterns

* **Data Science Design Patterns**: Recurring situations in data science and machine learning that require clever solutions.
* **BigQuery ML Innovations**: Elegant new approaches to address traditional problems with its revolutionary technology.

---

### 20.6.1. Hashed Feature

* **Solving Categorical Variable Issues**
	+ Incomplete vocabulary: Data might not have full set of values
	+ High cardinality: Creates scaling issues in ML models
	+ Cold start problem: New values added to categorical variable

---

### 20.6.2. Transforms

**BigQuery ML Feature Transformations**

* In BigQuery ML, feature transformations are applied to inputs before feeding into the model.
* These transformations must be applied to deployed models in production for consistent results.
* **Available transforms:**
	+ `_FEATURE_CROSS` for creating new features from existing fields
	+ `QUANTILE_BUCKETIZE` for dividing features into buckets
* Transformations are automatically added to built models, enabling pipeline consistency during prediction.

---

## 20.7. Summary

### **BigQuery ML Summary**

* _Democratized machine learning_ in the SQL community
* Enables use of ML pipelines and transformations directly using SQL for faster model creation
* Highly interoperable with Vertex AI

---

## 20.8. Exam Essentials

* **BigQuery and Machine Learning (ML)**
+ Learn how to train, predict, and explain models using SQL
+ Understand the differences between BigQuery ML and Vertex AI services

**Design Patterns and Integrations**
+ Leverage BigQuery's solutions for common machine learning problems (hashing, transforms, serverless predictions)
+ Seamlessly integrate BigQuery ML with Vertex AI

---

# 21. AppendixAnswers to Review Questions

---

## 21.1. Chapter 1: Framing ML Problems

### **Unsupervised Learning and Supervised Learning**

* **Unsupervised learning**: used for unlabeled data, cluster documents, provide keywords *Topic modeling*
* **Supervised learning**: used for labeled data, use for downstream prediction
    * Can be semi-supervised when mixed with unlabeled and labeled data

### **Model Evaluation Metrics**

* _**Accuracy is incorrect for imbalanced datasets**_: choice between precision and recall instead
* Recall is chosen to minimize false negatives

---

## 21.2. Chapter 2: Exploring Data and Building Data Pipelines

### Data Leakage Issues

* **Causes**:
	+ Oversampling
	+ Downsample majority class with unweighting to create balanced samples
	+ Model training on hospital name (label leakage)
* **Solutions**:
	+ Transform data before splitting for testing and training
	+ Removing features with missing values
	+ Retraining the model after monitoring skew and adjusting data distribution

---

## 21.3. Chapter 3: Feature Engineering

* **Data Preprocessing**
    * Normalizing data converts range to normalized format, aiding convergence of models.
    * One-hot encoding converts categorical features to numeric features.

* **Model Evaluation and Bias**
    * AUC PR minimizes false positives in imbalanced datasets compared to AUC ROC.
    * Training data performance indicates case of data leakage; cross-validation helps.

* **Data Loading and Processing**
    * Prefetching and interleaving improve processing time with TensorFlow Data.

* **ETL and Data Transformation**
    * Cloud Data Fusion is UI-based tool for ETL.
    * TensorFlow Transform is most scalable way to transform training/testing data.

---

## 21.4. Chapter 4: Choosing the Right ML Infrastructure

### **Understanding Technical Texts** * **Pretrained Models**: Always start with a pre-trained model and evaluate its performance. If that doesn't work, consider AutoML, and custom models as a last resort. * Use Google Translate's glossary feature to create a list of terms for translation * **Vertex AI Services**: * No "translated subtitle" service available, but you can combine services like Android app deployment and ML Kit. * **Vertex AI Custom Jobs** are recommended instead of AutoML or pretrained APIs (available on GCP). * **GCP Pricing**: Choose between 1 TPU or 8 GPUs based on your needs. TPUs are more efficient for sparse matrices, while GPUs are better for high-precision predictions. * Avoid using TPU and GPU in the same instance * **Content Recommendation**: * **"Frequently

---

bought together"**: Showed at checkout to help customers add more items * **"Recommended for you"**: Display on home pages to bring attention to likely products * **"Similar items"**: Based on product information only, helps customers choose between similar products * **Engagement and Recommendations**: * To encourage customer engagement, use "Frequently bought together" at checkout. * Use "Recommended for you" on home pages to bring attention to likely products. * Display "Similar items" based on product information only. * **Browsing Data and User Events**: When available, create recommendations based on user events. Without browsing data, rely on project catalog information. ### Key Concepts | Concept | Description | | --- | --- | | **AutoML** | Automated machine learning solution for

---

model development and deployment. | | **Vertex AI** | Google Cloud's artificial intelligence platform for building, deploying, and managing machine learning models. | | **GCP Pricing** | Google Cloud Platform pricing options, including TPUs (Tensor Processing Units) and GPUs (Graphics Processing Units). | ### Additional Notes * The Natural Language API does not accept voice input. * A Vertex AI custom job is the most appropriate solution for certain use cases. * You cannot have TPU and GPU in a single instance.

---
## 21.5. Chapter 5: Architecting ML Solutions

**Solution Overview**
* Use App Engine for deployment and call Vertex AI Prediction model endpoint
* *Use Cloud BigQuery as NoSQL store for scalability and low latency*
* Use Dataflow for preprocessing, Vertex AI platform for training and serving
* *Use Kubeflow Pipelines for automated retraining and experiment tracking*

**Infrastructure Considerations**
* *Avoid using Block and File Storage due to latency issues*
* Use Cloud Storage for document data lake
* *Use a Cloud Storage trigger with Pub/Sub topic to automate GKE training jobs*

**Prediction and Experimentation**
* Use batch prediction functionality for aggregated data
* *Use Kubeflow experiments for training and executing experiments*
* Use TensorFlow's BigQueryClient for efficient data reading

---

## 21.6. Chapter 6: Building Secure ML Pipelines

* **Federated Learning**: Deploy ML model on device where data is stored.
    * Use service account key for authentication with GOOGLE_APPLICATION_CREDENTIALS
* **Data Protection**:
    * Cloud DLP: redact and mask PII, manage VPC security
    * Vertex AI IAM access needed
* **Compute and Storage**:
    * Use Vertex AI-managed notebook for auto shutdown of idle instances

---

## 21.7. Chapter 7: Model Building

* **Concept 1**: Model performance suffers when training data distribution changes, necessitating monitoring and retraining. * **Training Data Skew**: Good performance on training data but poor validation data suggests training data skew. * **Memory Error**: Increasing batch size resolves out-of-memory error caused by large image sizes. * _Batch Size Adjustment_ * **Model Overcorrection**: High learning rate leads to oscillating loss curves and repeated overcorrection. * _Learning Rate Synchronization_ * **Regularization**: Regularizing models with L2 helps mitigate overfitting issues. * _L2 Regularization_ * **Data Augmentation**: Techniques like data augmentation can help with limited data. * _Data Augmentation Methods_ * **Bias-Variance Tradeoff**: Both parameters must be considered

---

during model training. * _Parameter Tuning_ * **Hyperparameter Tuning**: Adjusting hyperparameters, such as learning rate and batch size, is crucial for model performance. * _Hyperparameter Optimization_

---
## 21.8. Chapter 8: Model Training and Hyperparameter Tuning

### **Cloud-based Training Options**

* Use BigQuery SQL for least manual intervention and computation time
* **Custom Vertex AI training** provides full control over code
* **Vertex AI hyperparameter tuning** configures trials, search algorithm, and parameter ranges

### **Data Ingestion and Processing**

* **Pub/Sub with Cloud Dataflow** ingests streaming data
* Preprocess data in Dataflow, send to ML training, store in BigQuery
* Use **BigQuery magic** or BigQuery client to convert data into DataFrame

### **Monitoring and Performance**

* **TensorFlow Profiler** tracks metrics and monitors performance
* Use interactive shell for real-time evaluation

---

## 21.9. Chapter 9: Model Explainability on Vertex AI

### Feature Attribution Methods for Nondifferentiable Models
* **Sampled Shapley**: Used for nondifferentiable models due to TensorFlow's non-differentiable operations.
* **Integrated Gradients**: Supported feature attribution technique, but not recommended for image data.
* _XRAI_: Supported feature attribution technique, but exact usage requires further information.

---

## 21.10. Chapter 10: Scaling Models in Production

### Architecture Overview
* **Use Bigtable** for dynamic feature lookup with low-latency serving requirement.
* Implement caching predictions in a _**Cloud Datastore**_ for faster lookup.
* Utilize Pub/Sub to Cloud Function for notification when user account balance drops below threshold.

### Batch Prediction and Notification
* Leverage **BigQuery** for batch prediction with Vertex AI.
* Create daily batch prediction job using the schedule function in Vertex AI managed notebooks.
* Set up notification with **Pub/Sub to Cloud Function** architecture for timely alerts.

---

## 21.11. Chapter 11: Designing ML Training Pipelines

**TFX and Kubeflow Pipeline Management**

* _Use TFX Evaluator or ModelValidator for model performance benchmarks_
* Use event-based Cloud Storage triggers to schedule Vertex AI Pipelines
    * Least effort setup: serverless Kubeflow Pipelines using Vertex AI
* Load Kubeflow BigQuery component URL to query BigQuery in Kubeflow Pipelines
* Orchestrate TFX pipeline with Apache Airflow or Kubeflow
* Automate Kubeflow Pipeline testing with Cloud Build

---

## 21.12. Chapter 12: Model Monitoring, Tracking, and Auditing Metadata

### **Key Concepts** * **Data Drift**: When input distribution changes over time. * Example: Distribution of "height" feature changed by 2 standard deviations * **Concept Drift**: When relationship between input and predicted value changes. * Example: Fraudsters changed their modus operandi to evade detection * **Distance Metrics**: * L-infinity distance (Greatest distance between two vectors) * Chebyshev distance * **Metadata Store**: * Artifact: A specific piece of information stored in a metadata store * Example: Train and test dataset * Context: Relevant information for model execution, not an artifact * Example: Environment variables or user credentials * Execution: The process of executing the model on input data * **Logging Options**: * Container logging * Access logging *

---

Request-response logging

---
## 21.13. Chapter 13: Maintaining ML Solutions

**MLOps Level**

* The correct MLOps level for this organization is *_Level 2_*, as they are dealing with many models in parallel and experimenting with different algorithms and technologies.
* Other options (A, B, C) are valid in *Level 1*, but not suitable for the maturity level of this organization.

**Feature Store**

* The correct Feature Store solution is *_Vertex AI Feature Store_*.
* This managed service provides a real-time serving capability, is easy to maintain and hand off, and fits the use case best.

---

## 21.14. Chapter 14: BigQuery ML

**BigQuery Integration with Vertex AI**
* *_Option D_* is the correct answer because it effectively uses a Vertex AI and BigQuery integration to solve the problem.
* *_Option C_* is also possible but less effective due to data transfer requirements.
* *_Option A_* is wrong due to unnecessary data transfer, while *_Option B_* involves redundant retraining of models.

---

# 22. Index

---

# 23. Online Test Bank

---

## 23.1. Register and Access the Online Test Bank

### Registering Your Book for Online Access
* Go to www.wiley.com/go/sybextestprep and select your book from the list.
* Complete registration information and answer security verification to receive a pin code.
* Enter the pin code on the test bank site to activate it.

### Creating an Account and Accessing the Test Bank

* Create an account or log in with existing credentials.
* Find your book in the test bank and click "Register or Login".
* Activate PIN and refresh page if needed.

---

# 24. WILEY END USER LICENSE AGREEMENT

---

