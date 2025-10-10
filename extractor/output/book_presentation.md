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

---# 2. Official Google Cloud CertifiedProfessional Machine Learning EngineerStudy Guide

---

# 7. Chapter 1Framing ML Problems

---

## 7.1. Translating Business Use Cases

* _Identify the impact, success criteria, and data available for a use case_
    * Match the use case with a machine learning problem (algorithm and metric)
    * Understand the stakeholders' expectations and their ability to communicate the value of the project
* _Determine the feasibility of solving the problem using machine learning_
    * Evaluate existing technology, available data, and budget
    * Familiarize yourself with the latest advancements in the relevant field (e.g. natural language processing)
* _Select an ML learning approach that fits the use case_
    * Consider multiple options and evaluate their potential impact on the business

---

## 7.2. Machine Learning Approaches

* *_Machine Learning Landscape_*
  * _Multiple Approaches exist with varying levels of research and application_
  * _Hundreds or thousands of techniques are available, requiring knowledge of categories and problem types_
  * _Each approach solves a specific class of problems, distinguished by data type and prediction method_

---

### 7.2.1. Supervised, Unsupervised, and Semi‐supervised Learning

* **Machine Learning Classification**: Machine learning approaches can be classified into two main types: supervised and unsupervised.
*   *Supervised Learning*: Uses labeled datasets to train models, e.g., image classification, sentiment analysis.
*   *Unsupervised Learning*: Groups or classifies data without labels, e.g., clustering algorithms, autoencoders.

---

### 7.2.2. Classification, Regression, Forecasting, and Clustering

* **Classification**: Predicting labels or classes in images, videos, or text.
    * _Types:_ Binary (2 labels) and Multiclass (more than 2 labels)
    * _Applications:_ Cloud Vision API
* **Regression**: Predicting numerical values for continuous outcomes.
    * _Example:_ House price prediction, rainfall amount prediction
    * _Data types:_ Structured data with rows and columns
* **Forecasting**: Using time-series data to predict future values.
    * _Types:_ Can be converted to regression or classification problems by modifying the data

---

## 7.3. ML Success Metrics

* **Choosing an ML Metric**
+ **Metric Selection**: Select a metric that aligns with the business success criteria, considering factors like recall, precision, and F1 score.
+ *_Recall_*: Measures the percentage of true positives predicted correctly. A higher score indicates better performance in detecting rare or important classes.
    *   _Formula:_  True Positives / (True Positives + False Negatives)
* **Metrics Comparison**
+ *_Precision_*: Measures the percentage of true positives among all positive predictions. A higher score indicates fewer false positives.
    *   _Formula:_  True Positives / (True Positives + False Positives)

---

+ *_F1 Score_*: Harmonic mean of precision and recall, providing a balanced measure of both false positives and false negatives.
    *   _Formula:_  2 x (Precision x Recall) / (Precision + Recall)

---
### 7.3.1. Area Under the Curve Receiver Operating Characteristic (AUC ROC)

* **ROC Curve**: graphical plot summarizing binary classification model performance
    * The ideal point is the top-left corner (100% TP, 0% FP)
    * A diagonal line is the worst case, and we want the curve to be far from it
        * Calculating the area under the curve (AUC) compares two models
* **Advantages of AUC**:
    * Scale-invariant: measures ranked predictions, not absolute values
    * Classification threshold-invariant

---

### 7.3.2. The Area Under the Precision‐Recall (AUC PR) Curve

* **Area Under the Precision-Recall Curve (AUC PR)**:
	+ Measures relationship between recall and precision
	+ Best AUC PR: horizontal line at top, with 100% precision and recall at optimal point (_top-right corner_)
	+ Preferred in imbalanced datasets to avoid skewing

---

### 7.3.3. Regression

* **Regression Metrics**: Used to evaluate the quality of predictions
    * *_Mean Absolute Error (MAE)_*: Average absolute difference between actual and predicted values
    * *_Root-Mean-Squared Error (RMSE)_*: Square root of average squared difference between target and predicted values
        + Penalizes large overpredictions, ranges from 0 to infinity
* *_Asymmetric Metric_*: 
    + *_RMSLE_*: Root-mean-squared logarithmic error, penalizes underprediction, uses natural log of predicted and actual values (+1)
        - Ranges from 0 to infinity
* *_Proportional Error_*:

---

    + *_Mean Absolute Percentage Error (MAPE)_*: Average absolute percentage difference between labels and predicted values

---
## 7.4. Responsible AI Practices

* **AI Responsibility**: Consider fairness, interpretability, privacy, and security in ML solutions
    * _Fairness_: measure bias with statistical methods, use model explanations for transparency
    * _Interpretability_: choose algorithms with inherent interpretability (e.g. linear regression), use model explanations to quantify feature contributions
    * _Privacy_ and _Security_: detect and prevent data leakage, minimize threats from data collection, training, and deployment phases

---

## 7.5. Summary

* *_Understanding Business Use Cases_* 
  * Identifying key stakeholders and their needs
  * Framing machine learning problems for optimal solution design

---

## 7.6. Exam Essentials

* **Machine Learning Fundamentals**
  * Understand business use cases and problem types (regression, classification, forecasting)
  * Know data types and popular algorithms for each type
  * Be familiar with ML metrics (e.g. precision, recall, RMSE) and understand how to match metric to use case
* **Responsible AI Practices**
  * Understand Google's Responsible AI principles
  * Familiarize yourself with recommended practices for AI in fairness, interpretability, privacy, and security

---

# 8. Chapter 2Exploring Data and Building Data Pipelines

---

## 8.1. Visualization

* *_Data Visualization_*: Exploratory technique to find trends and outliers
    * *_Applications_*:
        * **Data Cleaning**: Identify imbalanced data and clean it visually
        * **Feature Engineering**: Visualize features to select the most influential ones
    * *_Visualization Techniques_*:
        + Univariate Analysis (box plots, distribution plots)
        + Bivariate Analysis (line plots, bar plots, scatterplots)

---

### 8.1.1. Box Plot

* A box plot displays data distribution by showing *25th, 50th*, and *75th quartiles*
    * It represents maximum observations within the *interquartile range*
        * Outliers are points outside the *whiskers (max/min)*

---

### 8.1.2. Line Plot

* **Line Plot**: A graph that displays the relationship between two variables, showing trends in data changes over time.
    * _Used for analysis and visualization_ 
    * _Helps to identify patterns and correlations_
    * _Commonly used in time-series data analysis_

---

### 8.1.3. Bar Plot

* A bar plot displays trends and comparisons between categorical data
    * Typically used for sales figures, visitor numbers, and revenue data over time
    * Facilitates visual analysis of changes and patterns in the data

---

### 8.1.4. Scatterplot

* A scatterplot is the most common plot used in data science.
* It visualizes clusters in a dataset.
* It shows the relationship between two variables.

---

## 8.2. Statistics Fundamentals

* **Measures of Central Tendency**: 
    * Mean: average value of a dataset
    * Median: middle value when data is sorted
    * Mode: most frequently occurring value

---

### 8.2.1. Mean

Mean is the accurate measure to describe the data when we do not have any outliers present.

---

### 8.2.2. Median

* *_Median calculation_* 
    * Find median by arranging data values from lowest to highest
    * For even numbers, calculate average of two middle values
    * For odd numbers, use the middle value as median

---

### 8.2.3. Mode

* _Mode_: Value(s) occurring most frequently
    * Used to identify outliers when majority of data points are similar
    * In example: dataset has three occurrences of value 5, making it the mode

---

### 8.2.4. Outlier Detection

* **Mean is sensitive to outliers**, affecting its calculation.
    * In the dataset [15, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9, 210], the outlier 210 significantly changes the mean compared to median and mode.
        * **Mean:** with outlier: 12.72, without outlier: 29.16

---

### 8.2.5. Standard Deviation

* **Statistics Definitions**
  * _Standard Deviation_: square root of variance
  * Standard deviation measures how spread out data is, identifying outliers with values >1 SD from mean
  * *_Covariance_*: measure of how two random variables vary from each other

---

### 8.2.6. Correlation

* **Correlation Explained**: Correlation is a normalized form of covariance, measuring the strength and direction of linear relationship between two variables.
    * _Types of Correlation_:
        * Positive: increase in one variable leads to increase in another
        * Negative: increase in one variable leads to decrease in another
        * Zero: no substantial impact on each other

---

## 8.3. Data Quality and Reliability

* *_Data Quality Matters_*: The quality of your model depends on reliable and high-quality data.
    * *_Data Issues_*:
        * Missing values
        * Duplicate values
        * Noisy features (e.g., GPS measurements)
    * *_Ensuring Data Reliability_*: Check for label errors, noise in features, and outliers to ensure data quality.

---

### 8.3.1. Data Skew

* **Data Skew**: Asymmetrical data distribution, often with outliers and skewness in either direction.
    * _Causes_: Outliers, non-normal distribution
    * _Effects_: Poor model performance due to extreme values
        * _Solutions_:
            * Log transformation and normalization
            * Synthetic Minority Oversampling Technique (SMOTE)

---

### 8.3.2. Data Cleaning

* **Normalization**: transforms features to a consistent scale, improving model performance and training stability
    * _Reduces feature dominance_ 
    * _Improves model accuracy_ 
    * _Enhances model robustness_

---

### 8.3.3. Scaling

* Scaling converts feature values to a standard range (e.g., 0-1) to improve deep neural network convergence and avoid "NaN traps".
* Benefits include:
	* Faster gradient descent
	* Reduced impact of outlier features
	* Improved model performance with uniformly distributed data

---

### 8.3.4. Log Scaling

* **Log Scaling**: used when data varies by large amounts, e.g. 10,000 vs 100
* *Transforms values into a common range*, e.g. log of 100,000 = 100, log of 100 = 10
* *Reduces spread in data, making it suitable for modeling*

---

### 8.3.5. Z‐score

* *Z-score calculation*: `Scaled value = (value − mean) / stddev`
* *_Outlier detection_*: values beyond ±3 standard deviations from the mean are considered outliers
* **Scalability**: z-scores are used to normalize data and compare values across different datasets

---

### 8.3.6. Clipping

* **Clipping Extreme Outliers**: Cap feature values above or below to a fixed value using feature clipping.
* _**Before/After Normalization**_: Can be applied before or after other normalization techniques.

---

### 8.3.7. Handling Outliers

* An _outlier_ is a value that significantly differs from other data points
    * Visualization techniques like box plots and Z-score can detect outliers
    * Techniques also include clipping and IQR methods

---

## 8.4. Establishing Data Constraints

* **Schema**: A defined structure for the ML pipeline that describes the property of the data, including data type, range, format, and distribution.
    * **Advantages**:
        * Enables metadata-driven preprocessing
        * Validates new data and catches anomalies like skews and outliers during training and prediction

---

### 8.4.1. Exploration and Validation at Big‐Data Scale

* *_TFDV: Scalable Data Validation for ML_* 
    * A tool for detecting data anomalies and schema anomalies in large datasets
    * Part of the TensorFlow Extended (TFX) platform, providing libraries for data validation and schema validation
    * Used in two phases:
        * Exploratory Data Analysis Phase: producing a data schema to understand the data for the ML pipeline
        * Production Pipeline Phase: defining a baseline to detect skew or drift in the model during training and serving

---

## 8.5. Running TFDV on Google Cloud Platform

* *_TFDW core APIs_*
  * Built on **Apache Beam SDK**
  * Runs in **Google Cloud Dataflow**, a managed batch and streaming pipeline service
  * Integrates with **BigQuery** and **Google Cloud Storage** for data warehousing and lakes, as well as *_Vertex AI Pipelines_* for machine learning

---

## 8.6. Organizing and Optimizing Training Datasets

* *_Dataset Splitting_*
  * **Purpose**: Divide dataset into training, test, and validation sets for model training, testing, and hyperparameter tuning.
  * **Key Roles**:
    * _Training Dataset_: Actual training data used to learn from
    * _Validation Dataset_: Subset used for hyperparameter tuning and evaluation of model behavior
    * _Test Dataset_: Sample data used to test and evaluate model performance

---

### 8.6.1. Imbalanced Data

* **Handling Imbalanced Data**: 
  * Downsample majority class and upweight minority class
  * Example weight: increases importance of minority class examples during training
  * **Advantage**: faster model convergence due to increased proportion of minority class examples

---

### 8.6.2. Data Splitting

* **Random Splitting Issue**: Randomly splitting data can cause skew because it may group similar topics together, such as stories written in the same time period.
    * _Solution: Time-based Splitting_ 
        - Split data by month or specific dates
        - For example, train on April and test on May 1-7

---

### 8.6.3. Data Splitting Strategy for Online Systems

* *_Time-based splits are recommended for online systems_*
    * _Ensure validation set mirrors lag between training and prediction_
        * _Use domain knowledge to choose split approach (time vs random)_

---

## 8.7. Handling Missing Data

* **Handling Missing Data**: There are several ways to handle missing data in a dataset
    * _Delete rows or columns with missing values_
    * _Replace missing values with mean, median, or mode of remaining values_
        * This method can prevent data loss compared to removing or deleting columns/rows
    * Use machine learning to predict missing values based on non-missing variables

---

## 8.8. Data Leakage

* *_Data Leakage_**: Model learns from test data during training, underperforming on unseen data.
    * _Causes of Data Leakage_: 
        * Incorrectly using target variable as a feature
        * Including test data in training data sets
    * *_Prevention Measures_*:
        * Select non-correlated features with target variable
        * Split data into test, train, and validation sets

---

## 8.9. Summary

* **Data Visualization**
  * Box plots, line plots, and scatterplots are used to visualize data
  * Statistical fundamentals such as mean, median, mode, and standard deviation are discussed
* **Data Cleaning and Validation**
  * Data cleaning techniques: log scaling, scaling, clipping, z-score
  * TFDV for validating data schema at scale
  * Importance of defining a data schema in an ML pipeline
* **Data Splitting and Handling**
  * Strategies for splitting data: time-based, clustered
  * Dealing with missing data and data leakage

---

## 8.10. Exam Essentials

* **Data Visualization**: Understand box plots, line plots, scatterplots, and statistical terms like mean, median, mode, and standard deviation.
    * **Data Quality**: Check for outliers, skew, and define a data schema; validate data using TFDV for scalability.
    * **Data Preparation**: Split dataset into training, test, validation data; apply sampling strategy for imbalanced data; handle missing values.

---

# 9. Chapter 3Feature Engineering

---

## 9.1. Consistent Data Preprocessing

* **Pretraining Data Transformation**: 
  * Performs transformation on complete dataset before model training
  * Advantages: single computation, examines entire dataset for transformations
  * Disadvantages: same transformation needs to be reproduced at prediction time, updates slow and compute-intensive
* _Inside Model Data Transformation_
  * Transformations part of the model code, applied during training
  * Decouples data and transformation, easy changes if needed
  * Potential disadvantage: increases model latency for large or computation-heavy transforms

---

## 9.2. Encoding Structured Data Types

* A good feature should be related to business objectives
* Features can be categorized into two types:
  * *_Categorical_*: defined by a limited number of values (e.g., yes/no, male/female)
  * *_Numeric_*: represented as scalar or continuous data (e.g., observations, recordings)

---

### 9.2.1. Why Transform Categorical Data?

* *_Categorical data limitation_*: Most ML algorithms can't work with label data directly.
    * **Conversion requirements**: Categorical variables need to be converted to numeric data for use in most algorithms.
        * *_Back-conversion_*: Numeric outputs must also be converted back to categorical data during predictions.

---

### 9.2.2. Mapping Numeric Values

* *_Numeric Data Transformation_*
  * _Normalizing_: adjusts data range to a common scale
  * _Bucketing_: groups data into discrete categories for analysis

---

### 9.2.3. Mapping Categorical Values

* *_Encoding_*: converting categorical data into numerical values
* *_Label Encoding_*: assigning a unique number to each category
* *_One-Hot Encoding_*: representing categories as binary vectors

---

### 9.2.4. Feature Selection

* **Feature Selection**: Selecting a subset of useful features to improve model prediction
    * Techniques include:
        * _Reducing dimensions using dimensionality reduction_ (reduces noise and overfitting)
        * Selecting most important features through techniques like backward selection and Random Forest
    * Alternatives: 
        * Combining new features through PCA and t-SNE

---

## 9.3. Class Imbalance

* **Classification Model Outcomes**
  * True positive: Correctly predicts the positive class
  * True negative: Correctly predicts the negative class
  * False positive: Incorrectly predicts the positive class
  * False negative: Incorrectly predicts the negative class, leading to missed diagnoses of actual patients

---

### 9.3.1. Classification Threshold with Precision and Recall

* **Classification Threshold**: A value that determines when to classify as positive or negative
    * _Influences false positives and false negatives_
    * _Problem-specific and must be fine-tuned_
* **Precision Recall Curve**:
  * _Precision: true positives / total positive predictions_
  * _Recall (TPR): percentage of correct positives out of total samples for that class_

---

### 9.3.2. Area under the Curve (AUC)

* **ROC AUC**: used for balanced datasets
    * measures precision at different recall levels
* **PR AUC**: used for imbalanced datasets 
    * such as credit card transactions with few false positives

---

## 9.4. Feature Crosses

**Feature Cross**
* **Definition**: Creating a new feature by multiplying two or more existing features
* *_Example_*: Combining latitude and longitude to predict crowded streets
* **Purpose**: Enhance predictive ability, represent nonlinearity in linear models

---

## 9.5. TensorFlow Transform

* **Data Pipelines in TensorFlow**: Efficient input pipelines are crucial for increasing TensorFlow model performance
* *_TF Data API_*: provides a high-level interface for building data pipelines, simplifying data processing and preparation
* **TensorFlow Transform**: enables the automated transformation of data pipelines, reducing manual effort and improving scalability

---

### 9.5.1. TensorFlow Data API (tf.data)

* **Data Input Pipeline Best Practices**
    * _Prefetch Transformation_: Overlap preprocessing and model execution using `tf.data.Dataset.prefetch`
    * _Parallelize Data Reading_: Use `tf.data.Dataset.interleave` to mitigate data extraction overhead
    * _Cache Data_: Use `tf.data.Dataset.cache` to cache data in memory during the first epoch

---

### 9.5.2. TensorFlow Transform

* **TensorFlow Transform**: a library that enables transformations before model training and emits a reproducible TensorFlow graph during training.
    * It addresses the **training-serving skew** issue in machine learning pipelines.
        + Key steps: data analysis, transformation, evaluation, metadata production, model feeding, and serving data.

---

## 9.6. GCP Data and ETL Tools

* **Google Cloud Tools for Data Transformation**
  * _Cloud Data Fusion_: A web-based service for building ETL/ELT pipelines, supporting Cloud Dataproc and MapReduce Spark streaming
  * _Dataprep by Trifacta_: A serverless, UI-based tool for visually exploring, cleaning, and preparing data for analysis, reporting, and machine learning

---

## 9.7. Summary

* **Feature Engineering**: Transforming numerical and categorical features is crucial for model training and serving.
    * **Dimensionality Reduction**: Techniques like PCA can help reduce feature dimensions while preserving important information.
    * **Classification**: AUC PR is more effective than AUC ROC for imbalanced classes, highlighting the importance of precision and recall in classification models.

---

## 9.8. Exam Essentials

* _Data Preprocessing Fundamentals_
  * Understand when and how to transform data before training
  * Know techniques for encoding structured data types
    * **Feature Selection**: reduce dimensionality and improve model performance
  * Use tools such as Cloud Data Fusion and Cloud Dataprep for ETL and data cleaning

* _Understanding Classification Metrics_
  * True Positive, False Positive, Accuracy, AUC, Precision, Recall in classification problems
  * Effectively measure accuracy with class imbalance using metrics like ROC-AUC score

---

# 10. Chapter 4Choosing the Right ML Infrastructure

---

## 10.1. Pretrained vs. AutoML vs. Custom Models

* **Pretrained Models**: * Easy to use and speed up ML incorporation; no need to think about algorithm, training, or deployment; widely used by thousands of users; available through APIs in Python, Java, and Node.js SDKs.
* *_Use Pretrained Models First_*; consider using custom models if needed
* **AutoML (Vertex AI)**: Build own model using own data; no need team of ML experts; suitable for unique use cases without AutoML options

---

## 10.2. Pretrained Models

* *_Pretrained Models_* 
  * Machine learning models trained on large datasets and performing well in benchmark tests
  * Available on Google Cloud platform for easy use through web console, CLI, SDKs, etc.
* *_Google Cloud Pretrained Models_*
  * Vision AI
  * Video AI
  * Natural Language AI
  * Translation AI
  * Speech-to-Text
  * Text-to-Speech
* *_Additional Platforms_* 
  * Document AI
  * Contact Center AI

---

### 10.2.1. Vision AI

* **Vision AI**: provides image processing and analysis using machine learning algorithms
    * _Convenient access_ through the Google Cloud platform
    * Performs tasks like:
        * Object detection and classification
        * Handwriting recognition (optical character recognition)
    * Provides four types of predictions:
        * Object detection in images
        * Image labels (e.g. Table, Furniture)
        * Dominant colors for image organization
        * "Safe Search" classification (e.g. Adult, Spoof)

---

### 10.2.2. Video AI

* **Video AI API**: Recognizes objects, places, and actions in videos
    * Supports over 20,000 different objects, places, and actions
    * Can be applied to stored or streaming videos with real-time results
* **Use Cases**
    * Video recommendation system using labels and user viewing history
    * Indexing video archives using metadata
    * Improving ad relevance by comparing content and ads

---

### 10.2.3. Natural Language AI

### **Google Cloud Natural Language AI**

* *_Provides_:_ 
  * Entity extraction
  * Sentiment analysis
  * Syntax analysis
  * Categorization into 700+ predefined categories
* *_Insights:_*_ 
  * Extracts entities with additional info (e.g. Wikipedia links)
  * Analyzes sentiment and syntax to identify emotions, parts of speech, dependencies, etc.
* *_Use Cases:_*_ 
  * Measure customer sentiment towards products
  * Extract medical insights from healthcare text

---

### 10.2.4. Translation AI

* _Translation AI service_ detects over 100 languages using Google's GNMT technology.
* Two levels: Basic and Advanced, with differences including glossary support and document translation capabilities.
* **Pricing:** differs between Basic and Advanced versions
* _Additional features:_ supports ASCII/UTF-8 text translation, real-time audio translation via Media Translation API.

---

### 10.2.5. Speech‐to‐Text

* *_Speech-to-Text service_* converts audio to text
* Used for creating subtitles for video recordings and streaming video
* Often combined with translation services for multi-language subtitles

---

### 10.2.6. Text‐to‐Speech

* *_Text‐to‐Speech service_*: provides realistic humanlike speech with 220+ voices across 40+ languages and variants
* *_Voice creation_*: create a unique voice to represent your brand at all touchpoints
* *_Language support_*: https://cloud.google.com/text-to-speech/docs/voices

---

## 10.3. AutoML

* **AutoML**: automates model training tasks 
    * _available for popular ML problems like image classification, text classification, and more_
        * _can be configured through web console or SDKs in Python, Java, or Node.js_

---

### 10.3.1. AutoML for Tables or Structured Data

* **Overview of BigQuery ML**: BigQuery ML is a SQL-based approach to training models, suitable for data analysts comfortable with writing SQL queries.
    * _Can be used for both training and prediction_
    * Automatically adds predictions to tables

* **Vertex AI Tables**: Trains ML models using Python, Java, or Node.js, or REST API, providing the ability to deploy models on an endpoint and serve predictions through a REST API.
    * Supports various AutoML algorithms for different data types and problems
        * _Table (IID)_: Regression, Classification
        * _Time-series data_: Forecasting

* **AutoML Job Configuration**: 

---

    * "Budget" specifies the maximum hours for training; if not completed within budget, uses best model trained within it
    * "Enable early stopping" ends job if deemed complete, saving only used node hours

---
### 10.3.2. AutoML for Images and Video

* **AutoML for Image Data**:
    * _Image Classification_: Predict one correct label from a list
    * _Multiclass Classification_: Predict all correct labels per image
    * _Object Detection_: Locate objects within an image
    * _Image Segmentation_: Label per-pixel areas of an image

* **AutoML for Video Data**:
    * _Video Classification_: Get label predictions for entire videos
    * _Action Recognition_: Identify action moments in a video
    * _Object Tracking_: Track objects with labels and timestamps

---

### 10.3.3. AutoML for Text

* *_Machine Learning Models for Text_*
  * _Built using Vertex AI AutoML text, providing easy model creation_
  * _Solves popular problems:_
    * *_Text Classification_*: predict one correct label per document
    * *_Multi-Label Classification_*: predict multiple labels per document
    * *_Entity Extraction_*: identify entities within text items
    * *_Text to Text Translation_*: convert text from source language to target

---

### 10.3.4. Recommendations AI/Retail AI

* **Retail Search**: Customizable search solution using Google's understanding of user intent and context, including image search via Vision API
    * *Trains on product images and metadata for accurate results*
    * *Can be integrated with other Retail AI solutions*

* **Recommendations AI**:
    * *Analyzes customer behavior and product data to drive engagement and sales*
    * *Uses machine learning models to provide personalized recommendations, including next products to buy and similar items*

---

### 10.3.5. Document AI

* _Document AI_ extracts details from scanned images of documents, handling variability in text styles.
    * **Key Tasks:**
        * _Text and layout information extraction_
        * _Entity recognition and normalization_
    * *_Processors_* are interfaces between documents and machine learning models, including general, specialized, and custom processors.

---

### 10.3.6. Dialogflow and Contact Center AI

* Dialogflow offers a conversational AI platform 
  * Integrating with Google Cloud for a contact center AI solution
  * Providing chatbots and voicebots for various applications

---

## 10.4. Custom Training

* **GPU Training**: Custom training allows for flexible hardware options, including GPUs, which accelerate compute-intensive deep learning operations.
* *_Compute-Intensive Operations_*: Matrix multiplications benefit from massively parallel architectures like GPUs, reducing training time by an order of magnitude.
* *_Training Time Reduction_*: Offloading computations to a GPU can reduce training time from days or months to hours.

---

### 10.4.1. How a CPU Works

* A **CPU** is a versatile processor that handles various tasks, including software and data applications.
* It loads data from memory, processes the value, and stores the result back into memory for each operation.
* *This architecture is suitable for general-purpose software applications, but inefficient for serially performing complex calculations like those in machine learning (ML) model training.*

---

### 10.4.2. GPU

* **GPUs**: Specialized chips designed to process data in parallel
    * _Key feature_: Thousands of arithmetic logic units (ALUs) for faster processing
    * **GPU limitations**: Must be used with compatible instance types and machine configurations
        * _Restrictions_: Number of GPUs per instance, virtual CPUs, and memory requirements

---

### 10.4.3. TPU

* **GPUs Limitations**: Semi-general-purpose processors with many applications, resulting in slower performance for ML workloads due to access to registers and shared memory.
    * _Increased data transfer overhead_ 
    * _Reduced ALU performance per cycle_
* **TPUs (Tensor Processing Units)**: Specialized hardware accelerators designed by Google for ML workloads.
    * **Matrix Multiply Units (MXUs)**: 128x128 multiply/accumulators, capable of 16,000 operations per cycle
    * **High-performance matrix processing**: Thousands of directly connected multiply-accumulators forming a large physical matrix.

---

## 10.5. Provisioning for Predictions

* **Prediction Phase**: Prediction workload is different from training, focusing on response time and cost.
    * _**Online Prediction**: Model deployment on a server, fast response times (~ near real-time), continuous scaling required.
        * _Scalability_ is crucial for online predictions.
    * _**Batch Prediction**: Initiated as batch jobs, reasonable response times (~ near normal time), cost optimization key.

---

### 10.5.1. Scaling Behavior

* **Autoscaling in Vertex AI**: automatically scales prediction nodes when CPU usage is high
    * _Configures monitoring for multiple resources: CPU, memory, and GPU_
        * Trigger configurations must be set to ensure proper scaling

---

### 10.5.2. Finding the Ideal Machine Type

* Use a Compute Engine instance with a custom container as a Docker container for cost-effective prediction.
    * Monitor CPU utilization and queries per second (QPS) to determine optimal machine type and avoid high costs due to single-threaded web server limitations.
    * Consider GPU acceleration, but be aware of restrictions on usage, regions, and DeployedModel resources.

---

### 10.5.3. Edge TPU

* *_Edge Devices_*: collect real-time data, make decisions, take action, and communicate with other devices or with the cloud.
    * Accelerated by Google's Edge TPU
    * Single Edge TPU performs 4 trillion operations per second on 2 watts of power
        * Used for edge inference, deploying models at the edge

---

### 10.5.4. Deploy to Android or iOS Device

* **ML Kit**: Easy-to-use machine learning package for mobile apps
    * Brings Google's expertise to developers
    * Optimized for device, saving bandwidth and enabling offline prediction

---

## 10.6. Summary

* **Google Cloud Models**: Pretrained models available on Google Cloud for various scenarios.
    * **Hardware Options**: GPUs, TPUs, and edge computing for training and prediction workloads.
        * **Edge Computing**: Deploying models to edge devices for real-time processing.

---

## 10.7. Exam Essentials

* **Choosing the Right Approach**
  * Pretrained models
  * AutoML
  * Custom models: select based on solution readiness, flexibility, and approach
* **Provisioning Hardware**
  * Training: GPU, TPU hardware with specific instance types (e.g., NVIDIA Tesla V100, Google Cloud TPUs)
  * Deployment: CPU and memory constraints in the cloud; TPUs in edge devices
* **Serverless Approach**
  * Use pretrained models and solutions for scalability and domain-specific problem-solving

---

# 11. Chapter 5Architecting ML Solutions

---

## 11.1. Designing Reliable, Scalable, and Highly Available ML Solutions

* **Overview**: Designing a reliable ML pipeline requires automation of data collection, transformation, training, tuning, deployment, and monitoring.
    * _Key components:_
      * Data storage (Google Cloud Storage)
      * Scalable infrastructure (Dataflow, Compute Engine)
      * Model training (Custom models with Vertex AI Training and AutoML)
* **Pipeline stages**:
  * **Training**: Automating model training with custom models using Vertex AI Training or AutoML
  * **Tuning and experiment tracking**: Using Vertex AI hyperparameter tuning and Experiments to track results
    * _Key benefits:_
      + Faster model selection

---

  * **Deployment and monitoring**: Scaling prediction endpoints in production with Vertex AI Prediction and Model Monitoring
    * _Key benefits:_
      + Fully managed scalability

---
## 11.2. Choosing an Appropriate ML Service

* **Google Cloud AI/ML Stack** 
    * The top layer consists of managed SaaS offerings such as Document AI, Contact Center AI, and Enterprise Translation Hub.
        *_Serverless and scalable APIs_*
    * Vertex AI services are the middle layer, including:
        *_AutoML for custom model creation_
        *_Workbench for data science workflow development_
    * The bottom layer consists of infrastructure such as compute instances and containers with TPUs, GPUs, and storage.
        *_Managed by user for scalability and reliability_*

---

## 11.3. Data Collection and Data Management

* **Data Stores in Google Cloud**
  * *_Google Cloud Storage_* 
  * *_BigQuery_* 
  * *_Vertex AI's datasets_* 
  * *_Vertex AI Feature Store_* 
  * *_NoSQL data store_*

---

### 11.3.1. Google Cloud Storage (GCS)

* _Google Cloud Storage (GCS)_ is a service for storing objects in Google Cloud.
* You can store various data types, including images, videos, audio, and unstructured data.
* Large files can be split into 100-10,000 shards to improve read and write throughput.

---

### 11.3.2. BigQuery

* _Use BigQuery for tabular data storage_ 
    * For training data, store as tables instead of views for better speed
    * Available tools: Google Cloud console, bq command-line tool, BigQuery REST API, Vertex AI, Jupyter Notebooks

---

### 11.3.3. Vertex AI Managed Datasets

* **Use Managed Datasets for Custom Model Training**
  * Provides benefits such as central location management, integrated data labeling, easy lineage tracking, and comparison of model performance.
  * Supports four primary data formats: image, video, tabular (CSV, BigQuery tables), and text.
  * Allows automatic splitting of data into training, test, and validation sets.

---

### 11.3.4. Vertex AI Feature Store

* _**Centralized Repository for ML Features**:_ Vertex AI Feature Store is a fully managed repository for organizing, storing, and serving ML features.
    * **Benefits:**
        * No need to compute feature values and save them in multiple locations
        * Helps detect drifts and mitigate data skew
        * Enables fast online predictions with real-time retrieval of feature values

---

### 11.3.5. NoSQL Data Store

* Static feature lookup requires optimized database for low-latency singleton read operations
    * **Options:** Memorystore, Datastore, Bigtable
        * Memorystore: submillisecond latency with Redis offering
        * Datastore: high performance, ease of application development, and scalability
        * Bigtable: massively scalable, high throughput, and low-latency workloads

---

## 11.4. Automation and Orchestration

* _Machine Learning Workflows_: Phases include data collection, preprocessing, model training, evaluation, and deployment.
    * Typically implemented in an ML project pipeline
    * Requires orchestration and automation for production environment integration
    * **Kubeflow Pipelines**: A serverless pipeline solution built atop Kubernetes
        * Automates execution for continuous model training

---

### 11.4.1. Use Vertex AI Pipelines to Orchestrate the ML Workflow

* *_Vertex AI Pipelines_* 
    * A managed service for automating, monitoring, and governing ML systems
    * Orchestrate ML workflow in a serverless manner using Kubeflow Pipelines SDK or TensorFlow Extended
    * Stores workflow artifacts in Vertex ML Metadata for lineage analysis

---

### 11.4.2. Use Kubeflow Pipelines for Flexible Pipeline Construction

* **Kubeflow Overview**: An open source Kubernetes framework for developing and running portable ML workloads
    * _Key Components_: Kubeflow Pipelines (compose, orchestrate, automate ML systems)
        + Supports local, on-premises, cloud deployments
    * _Flexibility_: Allows simple code construction of pipelines with Google Cloud Pipeline Components

---

### 11.4.3. Use TensorFlow Extended SDK to Leverage Pre‐built Components for Common Steps

### TensorFlow and Vertex AI Workflow
* **Overview**: TensorFlow provides components for common steps in the Vertex AI workflow, including data ingestion, validation, and training.
* **TFX**: Provides frameworks and libraries for defining, launching, and monitoring machine learning models in production.
* **Recommended SDK**: *TensorFlow Extended SDK*: For users already using TensorFlow, working with structured data, or handling large datasets.

---

### 11.4.4. When to Use Which Pipeline

* **Vertex AI Pipelines**: Compatible with Kubeflow Pipelines v1.8.9+ or TensorFlow Extended v0.30.0+
* *Built-in support for ML operations*: Tracks ML-specific metadata and lineage, essential for validating pipeline correctness.
* *Recommended over other orchestrators*: Due to built-in support and easier configuration, monitoring, and maintenance compared to TFX.

---

## 11.5. Serving

* **Model Deployment**:
  * Trained models are deployed to production for predictions.
  * Models can make predictions in two modes:
    *_Offline_*: No real-time data input
    *_Online_*: Real-time data input, often with continuous learning capabilities

---

### 11.5.1. Offline or Batch Prediction

* *_Offline Batch Prediction_*: predicting data in batches before receiving the data
    * Use cases: recommendations, demand forecasting, segment analysis
    * Can be done using Vertex AI batch prediction with BigQuery or Cloud Storage storage

---

### 11.5.2. Online Prediction

* **Real-Time Prediction**: _near real-time predictions are made when a request is sent to a deployed model endpoint_
    * Can be used in applications such as _real-time bidding_ and _sentiment analysis of Twitter feeds_
        + Minimizing latency at the model level: _build smaller models, use accelerators like Cloud GPU and TPU_
        + Minimizing latency at the serving level: _use low-latency data stores, precompute predictions, cache results_

---

## 11.6. Summary

* *Designing a reliable ML solution on GCP involves best practices for scalability and high availability*
* _Key services include Vertex AI, BigQuery, and NoSQL data stores_
* **Automation techniques**: Vertex AI Pipelines, Kubeflow Pipelines, TFX pipelines to manage ML pipelines

---

## 11.7. Exam Essentials

* Design reliable, scalable, and highly available machine learning (ML) solutions using Google Cloud AI/ML services
    * Choose an appropriate ML service based on your use case and expertise
        * Understand the AI/ML stack of GCP and when to use each layer
* Implement data collection, management, and storage for various ML use cases
    * Use suitable data stores for different scenarios
* Automate and orchestrate workflows using automation and orchestration techniques
    * Select between Vertex AI Pipelines, Kubeflow, and TFX pipelines as needed

---

# 12. Chapter 6Building Secure ML Pipelines

---

## 12.1. Building Secure ML Systems

* *_Encryption_*: A process that protects data from unauthorized access
* *Google Cloud provides*_ 
  * *_Encryption at Rest_*: Protects data while it is stored on servers
  * *_Encryption in Transit_*: Protects data while it is being transmitted between servers

---

### 12.1.1. Encryption at Rest

* **Data Encryption**: Data stored in Cloud Storage and BigQuery is automatically encrypted at rest by default, with option to use customer-managed encryption keys.
    * _Encryption methods_: Authenticated Encryption with Associated Data (AEAD) encryption functions used for individual table values in BigQuery
    * _Server-side vs Client-side encryption_: Server-side encryption occurs before data is written to disk and stored, while client-side encryption occurs after data is received but not yet stored.

---

### 12.1.2. Encryption in Transit

* *Transport Layer Security (TLS)*
  is used by _Google Cloud_ to secure data transmission over the internet.
* **Protects** against eavesdropping, tampering, and man-in-the-middle attacks.
* Ensures confidentiality and integrity of data during read and write operations.

---

### 12.1.3. Encryption in Use

* *_Encryption in Use_*: Protects data from compromise while being processed
    * Example: _Confidential Computing_, using *_Confidential VMs_* and *_GKE Nodes_*
        * Learn more at: https://cloud.google.com/blog/topics/developers-practitioners/data-security-google-cloud

---

## 12.2. Identity and Access Management

* **Vertex AI IAM**: Manage access to data and resources in Google Cloud using Identity and Access Management (IAM).
    * _Project-level roles_: Grant access to resources by assigning one or more roles to a principal.
    * **Customize permissions**: Use project-level and resource-level policies to grant specific permissions, such as read/write access to Vertex AI Feature Store.

---

### 12.2.1. IAM Permissions for Vertex AI Workbench

* **Vertex AI Workbench** is a data science service offered by Google Cloud Platform, leveraging JupyterLab to explore and access data.
    * **Types of Notebooks**: User-managed notebook instances offer high customizability, while managed notebooks are Google-Cloud managed and less customizable.
        + _Advantages of Managed Notebooks_: Integration with Cloud Storage and BigQuery, automatic shutdown when not in use.
* **Access Modes**: Two ways to set up user access modes:
    * Single User Only: grants access only to a specified user
    * Service Account: grants access to a service account, requiring secure setup and management.

---

### 12.2.2. Securing a Network with Vertex AI

* **Understanding Shared Responsibility**: The cloud provider monitors security threats, while end users (you) are responsible for protecting data and assets.
    * _Shared Fate Model_: Fosters a partnership between the cloud provider and customer to continuously improve security through ongoing collaboration.
        * **Key Components**:
            * Help getting started with secure blueprints
            * Risk protection program and assured workloads

---

## 12.3. Privacy Implications of Data Usage and Collection

* **Sensitive Data Types**
  * _Personally Identifiable Information (PII)_: name, address, SSN, date of birth, financial info, passport number, phone numbers, email addresses
* *_Protected Health Information (PHI)_*: covered under HIPAA Privacy Rule for patient care and other purposes
* **HIPAA's dual approach**: balances individual rights with necessary disclosure for patient care

---

### 12.3.1. Google Cloud Data Loss Prevention

* **Google Cloud Data Loss Prevention (DLP) API**: removes identifying information from text content, including tables
    * _Uses multiple techniques_: masking, tokenization, encryption, bucketing
    * _Provides risk analysis and inspection features_ to monitor sensitive data

---

### 12.3.2. Google Cloud Healthcare API for PHI Identification

* **HIPAA De-Identification**: PHI linked to 18 identifiers must be treated with special care under HIPAA.
    * **Cloud Healthcare API**: Highly configurable de-identification removes sensitive info from various data types, including text, images, FHIR, and DICOM.
        * _Data masking, deletion, or obfuscation_ of PHI targeted by the de-identify command.

---

### 12.3.3. Best Practices for Removing Sensitive Data

* **Handling Sensitive Data in Datasets**
  * **Structured Data**: Restrict access to sensitive columns using views.
  * **Unstructured Content**: Use Cloud DLP, NLP API, and other APIs to identify and mask sensitive data.

* **Protecting Multiple Columns**: Apply PCA or dimension-reducing techniques to combine features, then train ML models on the resulting vectors.

* **Coarsening Techniques**:
  * *_IP Addresses_*: Zero out the last octet of IPv4 addresses.
  * *_Numerical Quantities_*: Bin numbers to reduce individual identification.
  * *_Location_*: Use city, state, or zip code identifiers, or large ranges to obfuscate unique characteristics.

---

## 12.4. Summary

* **Data Security in Machine Learning**
  * Encryption at rest and transit
  * IAM for access management to Vertex AI Workbench
* **Secure ML Development**
  * Federated learning and differential privacy
* **PII and PHI Data Management**
  * Cloud DLP and Cloud Healthcare APIs

---

## 12.5. Exam Essentials

* **Secure ML Systems**: Understand encryption at rest and transit for ML data in Cloud Storage and BigQuery
* *   Set up IAM roles for Vertex AI Workbench management
    *   Network security setup for Vertex AI Workbench
        *   Protect against unauthorized access and data breaches
* 
* _Privacy Implications_
  *   Understand differential privacy, federated learning, and tokenization concepts
  *   Data usage and collection best practices
* 
* **Data Protection**
  *   Google Cloud Data Loss Prevention (DLP) API for PII type data identification and masking
    *   Google Cloud Healthcare API for PHI type data identification and masking

---

# 13. Chapter 7Model Building

---

## 13.1. Choice of Framework and Model Parallelism

* **Distributed Deep Learning**: Modern deep learning models require larger datasets and more parameters.
* **Multinode Training**: To train these models efficiently, multinode training with _data parallelism_ and _model parallelism_ is necessary.

---

### 13.1.1. Data Parallelism

* *_Data Parallelism_*: splitting dataset into parts, assigning to parallel machines/GPUs, using same parameters and computing gradients
* **Strategies for Distributed Training**: synchronous, asynchronous
* *_Trade-off in Data Parallelism_*: adjusting learning rate to maintain smooth training with multiple nodes

---

### 13.1.2. Model Parallelism

* **Model Parallelism**: *   _Splitting a model into parts to train on individual GPUs_ 
    *   Benefits: increases model size limit, improves vision model accuracy
    *   Example: splitting ResNet50 across 10 GPUs for training
* **Distributed Training Strategies using TensorFlow**
    *   _MirroredStrategy_: Synchronous training on multiple GPUs on one machine
    *   _TPUStrategy_: Synchronous training on multiple TPU cores

---

## 13.2. Modeling Techniques

Let's go over some basic terminology in neural networks that you might see in exam questions.

---

### 13.2.1. Artificial Neural Network

* **Artificial Neural Networks (ANNs)**: Simplest type of neural network with one hidden layer
* *Typically used for supervised learning and numerical data*
* *Example: Regression problems*

---

### 13.2.2. Deep Neural Network (DNN)

* _Deep Neural Networks (DNNs):_ ANNs with multiple hidden layers between input and output
* Characterized by at least two layers
* Examples: deep nets

---

### 13.2.3. Convolutional Neural Network

* _Convolutional Neural Networks (CNNs)_ 
    * Designed for image input and classification tasks
    * Can also handle other image-based tasks

---

### 13.2.4. Recurrent Neural Network

* **Recurrent Neural Networks (RNNs)**: designed for processing sequences of data
* **Applications**:
  * Natural language processing
  * Time-series forecasting
  * Speech recognition
* **Training**: stochastic gradient descent, loss function selection to optimize model weights and biases

---

### 13.2.5. What Loss Function to Use

* _Key point_: The choice of loss function is closely tied to the activation function in the output layer of a neural network.
    * **Loss functions**:
        * Regression: Mean squared error (MSE)
        * Binary classification: Binary cross-entropy, categorical hinge loss, and squared hinge loss
        * Multiclass classification: Categorical cross-entropy (one-hot encoded) or sparse categorical cross-entropy

---

### 13.2.6. Gradient Descent

* *The gradient descent algorithm minimizes loss by moving in the opposite direction of the gradient, which points to the steepest increase.* 
    * _The goal is to find the minimum point on the loss curve._ 
    * _A single step in the negative direction reduces the magnitude of the gradient,_

---

### 13.2.7. Learning Rate

* The _gradient vector_ has both direction and magnitude.
    * It determines the next point by multiplying its magnitude with a **learning rate** (step size) to move in a new direction.
        + Example: the next point is moved **0.025 units** from the previous one.

---

### 13.2.8. Batch

* **Gradient Descent Batch**: The size of the batch used to calculate the gradient in a single iteration.
* *_Batch Size Matters_*: Using too many examples in a single iteration can significantly slow down computation.
* *_Optimal Batch Size_*: A balance between speed and accuracy must be found.

---

### 13.2.9. Batch Size

* *_Batch size_*: number of examples in a batch
    * Typically fixed during training and inference, but can be dynamic with TensorFlow
    * _Common values_: 10 to 1,000 for mini-batches

---

### 13.2.10. Epoch

* *_Epoch_*: Iteration for training the neural network with all training data
    * Consists of one or more *_batches_*
        + A batch includes exactly one forward and one backward pass
        + All data is used once in each epoch

---

### 13.2.11. Hyperparameters

* *_Hyperparameters:_* Loss, learning rate, batch size, and epoch are the key parameters for training an ML model.
* *Optimal values:*_ A high learning rate may lead to too slow convergence, while a small one can result in slow learning.
* **Batch size:** Choosing an optimal batch size can impact computation time.

---

## 13.3. Transfer Learning

* **_Transfer Learning_**: storing knowledge gained from one problem and applying it to a different related problem.
    * *Example*: using knowledge of car recognition to recognize trucks, or vice versa.
    * *Benefit*: saves time, improves performance, and enables model development with limited data.

---

## 13.4. Semi‐supervised Learning

* _Semi-supervised learning_: Combines a small amount of labeled data with a large number of unlabeled examples
* Uses both for training, falling between *unsupervised* and *supervised* machine learning approaches
* Requires **small amount of labeled data** to improve model performance

---

### 13.4.1. When You Need Semi‐supervised Learning

**Semi-Supervised Techniques for Data Augmentation**

* Use unlabeled data to train a model and augment a smaller labeled dataset
* Examples:
    * Fraud detection: label known instances of fraud, retrain model on new data
    * Anomaly detection: identify patterns in large datasets to detect outliers
    * Speech recognition: use unlabeled audio to improve model performance

---

### 13.4.2. Limitations of SSL

*Semi-supervised learning* uses _limited labeled data_ and _abundant unlabeled data_ for promising results in classification tasks. However, its applicability depends on the *representativeness* of the labeled data to the entire dataset.

---

## 13.5. Data Augmentation

* **Data Augmentation**: Enhance training data with minimal modifications (flips, rotations, translations) to artificially increase size and improve performance.
* _Key goal_: Collect sufficient examples for complex tasks while dealing with limited available data.
* *_Benefits_*:
	+ Increase relevant data in existing datasets
	+ Improve model's ability to generalize

---

### 13.5.1. Offline Augmentation

* Offline augmentation *increases dataset size by performing transformations beforehand*
    * Preferring this method for smaller datasets due to scalability concerns
    * Example: Rotating all images doubles the dataset size

---

### 13.5.2. Online Augmentation

* **Online Augmentation**: Performing data augmentation on mini-batches before feeding them to the model.
* **Advantages**:
  * **Large Datasets**: Suitable for large datasets
  * *_Improves Learning Capability_* 
  * **Real-time Transformation**: Allows for real-time transformation without modifying the entire dataset.

---

## 13.6. Model Generalization and Strategies to Handle Overfitting and Underfitting

* **Key Concepts**: 
  * _Bias_: difference between prediction and correct value, represents error rate of training data
  * _Variance_: error rate of testing data, affects model's ability to generalize
* **Trade-offs**:
  + High bias: oversimplifies model, performs poorly on test data
  + High variance: pays too much attention to training data, overfits

---

### 13.6.1. Bias Variance Trade‐Off

* *_**Bias-Variance Tradeoff**_*: Model complexity affects bias and variance
* *_**Overfitting vs Underfitting**_*
  * Overfitting: too many parameters, high variance, low bias
  * Underfitting: too simple, high bias, low variance

---

### 13.6.2. Underfitting

* **Underfit Model**: fails to learn the problem, performs poorly on training and test datasets.
    * High bias, low variance
    * Can be caused by: poor data cleaning, high bias in model, overfitting
    * Solutions:
        * Increase model complexity
        * Remove noise from data

---

### 13.6.3. Overfitting

* **Overfit Model**: has low bias, high variance
    * _Common issues with overfit models_
        + performance varies widely on new data or noise
    * _Ways to reduce overfitting_ 
        + Increase training examples
        + Change network complexity and parameters
    * Supported methods in BigQuery ML:
        + Early stopping
        + Regularization

---

### 13.6.4. Regularization

* **Regularization Techniques**
  * _L1_ regularization: shrinks coefficients towards 0, useful for feature selection
  * _L2_ regularization: forces weights to be small but not 0, improves generalization in linear models

* **Common Issues with Backpropagation and Regularization**
  * Exploding gradients: prevent with batch normalization and lower learning rate
  * Dead ReLU units: prevent with lowering learning rate or using a different activation function like ReLU
  * Vanishing gradients: prevent with ReLU activation function or dropout regularization

---

## 13.7. Summary

* **Training Neural Networks**
  * Gradient descent is used to minimize the loss function
  * Hyperparameters such as learning rate and batch size affect training quality
  * _Overfitting_ occurs when a model is too complex for the data, while _underfitting_ happens when it's too simple

---

## 13.8. Exam Essentials

* **Parallelism Strategies**: Choose between data parallel or model parallel multinode training strategies for large neural networks.
* * **Distributed Training in TensorFlow**: Use distributed training techniques for efficient model deployment, such as Google's Cloud AI Platform.
* **Hyperparameter Tuning and Optimization**:
  * Learn key hyperparameters like learning rate, batch size, and epoch.
  * Apply regularization techniques to handle overfitting and underfitting.

---

# 14. Chapter 8Model Training and Hyperparameter Tuning

---

## 14.1. Ingestion of Various File Types into Training

* **Data Types**: structured (database tables), semi-structured (PDFs, JSON), unstructured (chats, emails, audio, images, videos)
* **Data Sources**: batch data, real-time streaming data, IoT sensors
* **Data Volumes**: small (few megabytes) to petabyte scale

---

### 14.1.1. Collect

* **Batch and Streaming Data Collection**: Use Google Cloud services for collecting batch or streaming data from various sources.
    * **Pub/Sub**:
        * Serverless scalable service for messaging and real-time analytics
        * Publish and subscribe across the globe with deep integration to processing services (Dataflow) and analytics services (BigQuery)
    * _Optimized for predictable workloads_, use Pub/Sub Lite instead
    * Datastream for on-premise database replication, supports Oracle and MySQL databases

---

### 14.1.2. Process

* *_Data preprocessing and transformation are essential steps in machine learning (ML) training_* 
  * **Tools for data processing:**
    * _e.g., Pandas, NumPy, SciPy_
    * _e.g., OpenRefine, Trifacta_
  * **Data standardization:** normalizing data to a common format

---

### 14.1.3. Store and Analyze

* _Data Storage Guidance_
    * Use **BigQuery** or **Google Cloud Storage** for tabular data
    * Store image, video, audio, and unstructured data in large container formats on Google Cloud Storage (avoid block storage and direct database reads)
    * Use TFRecord files for TensorFlow workloads, Avro files for other frameworks, and Parquet format with TF I/O

---

## 14.2. Developing Models in Vertex AI Workbench by Using Common Frameworks

* **Vertex AI Workbench Overview**
  * A Jupyter Notebook-based development environment for the entire data science workflow
  * Interact with Vertex AI and other Google Cloud services from within a Jupyter Notebook

* **Managed Notebooks vs User-Managed Notebooks**
  * Managed notebooks:
    * Automatic shutdown for idle instances
    * UI integration with Cloud Storage and BigQuery
    * Automated notebook runs (using Cloud Scheduler)
    * Custom containers, Dataproc or Serverless Spark integration, and frameworks preinstalled
  * User-managed notebooks:
    * No automatic shutdown for idle instances
    * Limited UI integration with Cloud Storage and BigQuery

---

    * No automated notebook runs
    * No custom containers, Dataproc or Serverless Spark integration, but can add own framework

---
### 14.2.1. Creating a Managed Notebook

* **Create Managed Notebook**:
  * Go to Vertex AI, enable all APIs, and create a new notebook
  * Set default settings and click Create
* **Managed Notebook Details**:
  * Notebook created with monthly billing estimate
  * Upgrade option available to upgrade the boot disk
* _Note: Upgrading managed notebooks is a manual process that preserves data on the data disk._

---

### 14.2.2. Exploring Managed JupyterLab Features

* You are redirected to a screen showing available frameworks after clicking Open JupyterLab
* The notebooks folder contains existing tutorials to help you get started with building and training models
* A terminal option is available to run terminal commands on the entire notebook

---

### 14.2.3. Data Integration

* Open the _Browse GCS icon_ in the left navigation bar
* Load data from cloud storage folders to integrate with a managed notebook 
* Use this feature to access data stored in Google Cloud Storage

---

### 14.2.4. BigQuery Integration

* _Access BigQuery data_ 
    * Click BigQuery icon on left
    * Use *_Open SQL editor_* option

---

### 14.2.5. Ability to Scale the Compute Up or Down

* *_Modifying Jupyter Environment_* 
    * Access click n1-standard-4
    * Modify Jupyter environment hardware
    * Attach GPU to instance (if applicable)

---

### 14.2.6. Git Integration for Team Collaboration

* _Integrate existing git repository or clone new one_
    * Use left navigation branch icon or run `git clone` command in terminal
    * Clone your repository for project collaboration

---

### 14.2.7. Schedule or Execute a Notebook Code

* To execute code in a Jupyter notebook, click "Execute" to submit it to an Executor.
    * This functionality is used to set up or trigger Vertex AI training jobs or deploy scheduling of a Vertex AI Prediction endpoint from within the Jupyter interface.
        *_Automatic execution_* can be enabled by clicking the triangle black arrow and selecting "Run all cells".

---

### 14.2.8. Creating a User‐Managed Notebook

* **Create User-Managed Notebook**: Choose execution environment during creation, picking from options like Python 3, TensorFlow, R, etc.
    * **Advanced Options**: Configure networking with shared VPCs available
    * _Access to JupyterLab and Git integration_

---

## 14.3. Training a Model as a Job in Different Environments

* *_Vertex AI Training Types_*
  * AutoML: Minimal technical effort for model creation and training
  * Custom Training: Full control over training application functionality, targeting specific objectives and algorithms

---

### 14.3.1. Training Workflow with Vertex AI

* **Vertex AI Training Options**
  * _Training Pipelines_: Orchestrates custom training jobs and hyperparameter tuning for AutoML or custom models
  * Custom Jobs: Specifies settings for running custom training code, including worker pools and machine types
  * Hyperparameter Tuning Jobs: Searches for optimal hyperparameter values for custom-trained models

---

### 14.3.2. Training Dataset Options in Vertex AI

* **Dataset Management**: Choose between *_No Managed Dataset_* (use Cloud Storage or BigQuery) or *_Managed Dataset_* (central location for managing datasets)
    * Advantages of *_Managed Dataset_*:
        - Easily create labels and annotation sets
        - Track lineage for governance and development
    * *_Prebuilt Containers_* vs *_Custom Containers_*

---

### 14.3.3. Pre‐built Containers

* **Setting up a prebuilt container**:
    * Organize your code according to the application structure
    * Upload training code as Python source distribution to Cloud Storage bucket
    * Specify dependencies in `setup.py`
* **Creating a custom job**:
    * Use command: `gcloud ai custom-jobs create` with parameters such as region, display name, machine type, replica count, executor image URI, working directory, and script path.

---

### 14.3.4. Custom Containers

* **Custom Containers on Vertex AI**: _Use custom containers to run your training application with faster start-up time, use of ML frameworks not available in prebuilt containers, and extended support for distributed training_
    * **Benefits**:
        * Faster start-up time
        * Use of preferred ML framework
        * Extended support for distributed training
* **Creating a Custom Container**: _Build a custom container using a Dockerfile and push it to an Artifact Registry. Then specify the dataset, compute instances, and custom container image URI in Vertex AI training_
    * Steps:
        1. Create a custom container and training file

---

        2. Set up files with required folder structure
            - Create a root folder
            - Create a Dockerfile and trainer/ folder
                - Create task.py (training code) file
        3. Build the Dockerfile and push it to an Artifact Registry

---
### 14.3.5. Distributed Training

* **Distributed Training**: Run multiple replicas on different nodes (worker pools) for a single job.
    * **Task Roles**: 
        - Primary: manages others, reports status
        - Secondary: performs work as designated in job configuration
        - Parameter Servers: stores model parameters and coordinates shared state
        - Evaluators: evaluates the model

---

## 14.4. Hyperparameter Tuning

* **Hyperparameters**: parameters of the training algorithm not learned during training
    * _Set before training begins_
    * **Example:** learning rate, must be chosen in advance for gradient descent

---

### 14.4.1. Why Hyperparameters Are Important

* _Hyperparameter selection is crucial for neural network success_
    * _Search algorithms: Grid Search, Random Search, and Bayesian Optimization_ 
        * **Grid Search**: Exhaustive search through manually specified hyperparameters
        * **Random Search**: Random combinations of parameters to find the best solution
            * _Advantage:_ Higher chances of finding optimal parameters without aliasing

---

### 14.4.2. Techniques to Speed Up Hyperparameter Optimization

* **Speed up hyperparameter optimization**:
  * Use a simple validation set instead of k-fold cross-validation for large datasets
  * Parallelize across multiple machines to speed up by ~n for n machines
  * Pre-compute or cache results to avoid redundant computations

---

### 14.4.3. How Vertex AI Hyperparameter Tuning Works

* **Hyperparameter Tuning**: Optimizes target variables (numeric metrics) by running multiple trials with adjusted hyperparameters.
  * **Job Configuration**:
    + `config.yaml`: defines API fields for HyperparameterTuningJob, including metric ID, goal, parameter ID, and container image URI.
  * **gcloud CLI Commands**: Create a custom job to start hyperparameter tuning with commands such as `gcloud ai hp-tuning-jobs create`

---

### 14.4.4. Vertex AI Vizier

* *_Vertex AI Vizier_*: Black-box optimization service for tuning hyperparameters in complex ML models
    * _Criteria for using Vertex AI Vizier:_
        * No known objective function available
        * Too costly to evaluate using traditional methods
    * _Uses and examples:_ 
        * Optimizing neural network parameters
        * Testing user interface arrangements
        * Identifying optimal buffer size and thread count for computing resources

---

## 14.5. Tracking Metrics During Training

* *Tracking machine learning model metrics: Interactive shell, TensorFlow Profiler, and What‐If Tool*
* Debugging techniques for ML models
* Tools covered in this section:
  * _Interactive Shell_
  * TensorFlow Profiler
  * What‐If Tool_

---

### 14.5.1. Interactive Shell

* **Debugging and Inspection**: Use an interactive shell to browse file systems, run debugging utilities, and analyze GPU usage in Vertex AI containers.
    * _Enable Web Access_: Set `enableWebAccess` API field to true while setting up custom jobs programmatically or checking Enable training debugging in the Vertex AI console.
    * **Interactive Shell Limitations**: Only available while job is in RUNNING state; links appear on job details page for each node.

* **Logging and Monitoring**:
    * _Vertex AI Console_: View logs by clicking on View logs link on Vertex AI training page.

---

    * _Cloud Monitoring_: Logs are exported to Cloud Monitoring, with some metrics also shown in the Vertex AI console.

---
### 14.5.2. TensorFlow Profiler

* **Vertex AI TensorBoard Profiler**: A managed version of TensorBoard for monitoring and optimizing model training performance
    * _Helps identify and fix performance bottlenecks_
    * Allows remote profiling of Vertex AI training jobs on demand
    * Visualizes results in Vertex TensorBoard

---

### 14.5.3. What‐If Tool

* **Using What-If Tool**
  * The What-If Tool allows inspection of AI Platform Prediction models through an interactive dashboard.
  * It integrates with TensorBoard, Jupyter Notebooks, Colab notebooks, and JupyterHub.

  * To use WIT:
    * Install the `witwidget` library
    * Configure `WitConfigBuilder` to inspect a model or compare two models

---

## 14.6. Retraining/Redeployment Evaluation

* *_Machine learning models_* suffer performance degradation over time due to changes in *_data_* and *_user behavior_*
* Causes of performance degradation include *_Data Drift_* and/or *_Concept Drift_*, which vary in speed of decay.

---

### 14.6.1. Data Drift

* *_Data Drift_*: Change in statistical distribution of production data from baseline data used to train or build a model.
* *Causes:* Changes in feature attribution or data itself, e.g., unit change in temperature data.
* *_Detection Methods:_* Examine feature distribution, correlation between features, or check data schema over baseline using a monitoring system.

---

### 14.6.2. Concept Drift

* **Concept Drift**: Statistical properties of a target variable change over time
    * _Example_: Sentiment analysis models may need to adapt as people's opinions shift
    * **Detection and Mitigation**: Model monitoring is crucial to detect drift and ensure model performance remains accurate.

---

### 14.6.3. When Should a Model Be Retrained?

* **Retraining Strategies**
  * *_Periodic Training_*: Retrain model at set intervals (e.g. weekly, monthly) based on updated training data.
  * *_Performance-Based Trigger_*: Retrain model when performance falls below threshold (ground truth or baseline data).
  * *_Data Changes Trigger_*: Retrain model in response to data drift (changes in production).

---

## 14.7. Unit Testing for Model Training and Serving

* **Testing ML Models**: Testing code, data, and model for expected behavior
    * *_Key Tests:_* 
        - Model output shape aligns with labels
        - Output ranges meet expectations
        - Single gradient step decreases loss
        - Data integrity through assertions

---

### 14.7.1. Testing for Updates in API Calls

* *_Testing API Updates_*: Test for runtime errors using unit tests with random input data
    * Run a single step of gradient descent to validate the model's behavior
        * _No retraining required, reducing resource intensity_

---

### 14.7.2. Testing for Algorithmic Correctness

* _Check Algorithmic Correctness_: Train model, verify decreasing loss
* _Test Specific Subcomputations_*: Test individual parts of the algorithm to ensure correctness
* *_Monitor for Memorization_*: Regularly train without regularization to detect if the model is memorizing training data

---

## 14.8. Summary

* **File Ingestion**: File types (structured, unstructured, semi-structured) are ingested into GCP using services such as Pub/Sub, BigQuery Data Transfer Service, and Datastream.
    * *_Collect_*: Collecting real-time data with Pub/Sub and Pub/Sub Lite
    * *_Process_*: Transforming data with Cloud Dataflow, Cloud Data Fusion, etc.
* **Model Training**: Model is trained using Vertex AI with frameworks like scikit-learn, TensorFlow, PyTorch, and XGBoost
    * *_Hyperparameter Tuning_*: Using search algorithms and Vertex AI Vizier for optimal results.

---

## 14.9. Exam Essentials

* **File Ingestion**: Understand various file types (structured, unstructured, semi-structured) for AI/ML workloads in GCP.
    * Use Pub/Sub, BigQuery Data Transfer Service, and Datastream for data collection and migration.
    * Transform data using Cloud Dataflow, Cloud Data Fusion, Cloud Dataproc, and Cloud Composer.

* **Vertex AI Workbench**: 
    * Create managed and user-managed notebooks with common frameworks (e.g., scikit-learn, TensorFlow).
    * Use AutoML and custom training options for model development.
    * Set up distributed training using custom jobs and track metrics during training.

---

    * Contribute to retraining/redeployment evaluation and handle bias-variance trade-off.

* **Model Training and Serving**: 
    * Unit test data and models for machine learning.
    * Test for API updates and algorithm correctness.
    * Understand hyperparameter tuning (grid search, random search, Bayesian search) and regularization (L1, L2).

---
# 15. Chapter 9Model Explainability on Vertex AI

---

## 15.1. Model Explainability on Vertex AI

* **Explainability in Model Development**: As business outcomes rely on model predictions, developers are responsible for explaining model decisions.
* _Human Explainable ML (XAI) is crucial_ to address questions like "Why was my loan rejected?" or "Why should I take 10 mg of this drug?"
* **Visibility into Training Process** is essential to develop trustworthy and understandable machine learning models.

---

### 15.1.1. Explainable AI

* **Explainability**: ability to understand internal mechanics of an ML system
    * **Types**:
        * Global: transparency and comprehensiveness of overall ML model
        * Local: explanations for individual predictions
    * Increases trust, improves adoption, and helps debugging complex models

---

### 15.1.2. Interpretability and Explainability

* **Key difference between interpretability and explainability**: 
    * _Interpretability_ focuses on associating a cause to an effect
    * _Explainability_ focuses on justifying the model's results with hidden parameters

---

### 15.1.3. Feature Importance

* **Feature Importance**: Explains the value of each feature in constructing the model
    * _Benefits_:
        * *_Variable Selection_*: Identify unimportant features, reduce compute costs, and training time
        * *_Data Leakage Prevention_*: Avoid adding target variable to training dataset, reducing bias

---

### 15.1.4. Vertex Explainable AI

* **Vertex Explainable AI**: integrates feature attributions into Vertex AI, providing insights into model outputs for classification and regression tasks.
    * **Supported Services**:
        * AutoML image models (classification)
        * AutoML tabular models (classification & regression)
        * Custom-trained TensorFlow models on tabular & image data

---

### 15.1.5. Data Bias and Fairness

* _Data Bias_: occurs when certain parts of the data are not collected or misrepresented, leading to skewed outcomes.
    * **Causes**: surveys, systemic/historical beliefs, non-random/small data samples
* _ML Fairness_: ensures models treat individuals fairly and avoid biases based on characteristics like race/gender/disabilities/sexual orientation.
    * **Tools for Detection**:
        * Explainable AI: attributions technique, features overview functionality
        * What-If Tool (Vertex AI)
        * Language Interpretability Tool (open source)

---

### 15.1.6. ML Solution Readiness

* **Google's Responsible AI approach**: adheres to principles of fairness, transparency, and accountability in AI development and deployment.
    * _Best practices include model cards, Explainable AI tools, and human-in-the-loop reviews_
* **Model Governance**: provides guidelines and processes for implementing AI principles, including avoiding bias and justifying decisions.
    * _Key practices include responsibility assignment matrices, model versioning, and fairness evaluation_

---

### 15.1.7. How to Set Up Explanations in the Vertex AI

* To use Vertex Explainable AI with a custom-trained model, you need to configure explanations, which is not required for AutoML models.
    * For online explanations, send a `projects.locations.endpoints.explain` request instead of `projects.locations.endpoints.predict`.
    * Local kernel explanations can be performed in the User-Managed Vertex AI Workbench notebook using the preinstalled Explainable AI SDK.

---

## 15.2. Summary

* **Explainable AI**: *Explaining AI decisions using data insights to build trust*
    * _Technique:_ Model feature attribution
        + Sampled Shapley
        + XRAI (eXplainable Regression AI)
        + Integrated Gradients

---

## 15.3. Exam Essentials

* **Explainability on Vertex AI**: Model explainability helps understand complex predictions, ensuring fairness and accountability.
    * _Key Features:_
        * Sampled Shapley algorithm
        * Integrated gradients
        * XRAI
    * Supports Explainable AI SDK for TensorFlow prediction container and AutoML models.

---

# 16. Chapter 10Scaling Models in Production

---

## 16.1. Scaling Prediction Service

* _A saved model is a complete TensorFlow program containing trained parameters and computation._
* **Saved Models**: Stored as directories on disk, with `saved_model.pb` being a protocol buffer describing the function.
* *Saved models can be used for sharing or deploying with other frameworks like TensorFlow Lite, TensorFlow.js, and TensorFlow Serving.*

---

### 16.1.1. TensorFlow Serving

* **TensorFlow Serving**: Hosts trained TensorFlow models as API endpoints through a model server.
    * Allows loading models from different sources
    * Supports REST and gRPC API endpoints
    * Can be installed with Docker for ease of use

---

## 16.2. Serving (Online, Batch, and Caching)

* *_Serving Options_*: Batch prediction and online prediction
* *_Best Practices_*
  * _Caching Strategy_: Implement a cache layer to improve serving performance
  * _Scalability_: Design architectures that scale with the model's complexity
  * _Monitoring_: Continuously monitor system performance for optimization

---

### 16.2.1. Real‐Time Static and Dynamic Reference Features

* **Input Features**:
    * *_Static Reference Features_*: fixed, batch-updated values (e.g., customer ID)
        * Available in data warehouses
        * Used for estimating prices, recommending products
    * *_Dynamic Real-Time Features_*: computed on the fly (e.g., sensor data)
        * Use cases: predicting engine failure, recommending news articles

---

### 16.2.2. Pre‐computing and Caching Prediction

* **Batch Scoring Architecture**: Pre-compute predictions in batch scoring jobs and store them in a low-latency data store for online serving.
* **Types of Lookup Keys**:
  * _Specific Entity_: predictions based on unique IDs (e.g., customerid, deviceid)
  * _Hybrid Approach_: precompute for top-N entities and use model for remaining entities
* **Hashed Combination of Input Features**: create a hashed key for combination of input features and store corresponding prediction value

---

## 16.3. Google Cloud Serving Options

* _Deployment Options in Google Cloud_ 
  * Online and batch predictions available
  * Both AutoML and custom models supported

---

### 16.3.1. Online Predictions

* To deploy a real-time prediction endpoint, import models from elsewhere into Google Cloud.
    * *_Models must be in compatible formats_*:
        + TensorFlow SavedModel: `saved_model.pb`
        + scikit-learn: `model.joblib` or `model.pkl`
        + XGBoost: `model.bst`, `model.joblib`, or `model.pkl`
* Follow these steps to set up a prediction endpoint: 
    * Deploy the model resource to an endpoint.
    * Make a prediction.
    * Undeploy the model resource if not in use.

---

### 16.3.2. Batch Predictions

* You can run batch predictions using AutoML models or custom-trained models.
    * Batch predictions can be prepared in JSON Lines, TFRecord, CSV files, file list, BigQuery tables, or as a Google Cloud Storage bucket.
        * To enable model monitoring for batch predictions, you need to select for output a Google Cloud Storage bucket.

---

## 16.4. Hosting Third‐Party Pipelines (MLflow) on Google Cloud

* **MLflow Overview**: *Open source platform for managing machine learning life cycle, library agnostic and language independent*
    * _Tackles four primary functions:_ 
        - Tracking experiments
        - Packaging ML code
        - Managing models
        - Model registry

---

## 16.5. Testing for Target Performance

* **Testing for Production**
  * Check training-serving skew and model quality with real-time data
  * Monitor model age and performance throughout the ML pipeline
  * Test for numerically stable weights and outputs (e.g., no NaN or null values)
  * Utilize tools like Vertex AI Model Monitoring and Feature Store to detect skew and monitor performance

---

## 16.6. Configuring Triggers and Pipeline Schedules

* **Triggering Training/Prediction Jobs**
    * Use services like Cloud Scheduler, Managed Notebooks, Cloud Build, or Cloud Run to schedule and execute training or prediction jobs on Vertex AI.
    * Consider using orchestrators like Cloud Workflows, Cloud Composer, or TFX pipelines for complex workflows.

---

## 16.7. Summary

* _TF Serving Overview_
  * Covers predict function and its relation to SignatureDef
  * Discusses online serving architecture options (static, dynamic, caching)
  * Explains model deployment methods (online, batch) with Vertex AI Prediction and Google Cloud options
  * Identifies performance degradation factors (training-serving skew, data quality changes)

---

## 16.8. Exam Essentials

### TensorFlow Serving Overview
* **What is TensorFlow Serving**: A framework for serving machine learning models in production.
* **Deployment using Docker**: Set up TF Serving with Docker containers for scalable model deployment.
* _**Scalable prediction services**: Online, batch, and caching serving modes for improved performance._

* **Model Serving Response**: Saved model's SignatureDef tensors define the prediction response format.
* _**Online vs. Batch Serving**: Real-time invocation (static or dynamic reference features) vs. batch processing with caching for faster responses._
* _**Caching Strategies**: Improve serving latency by storing frequently accessed predictions._


---

* **Google Cloud Options**: Set up real-time endpoints using Google Cloud Vertex AI Prediction and configure batch jobs.
* _**Vertex AI Services**: Monitor model performance, diagnose degradation issues, and automate pipelines._

* **Configure Triggers and Pipelines**: Schedule triggers with Cloud Scheduler and manage workflows for automated pipeline execution.

---
# 17. Chapter 11Designing ML Training Pipelines

---

## 17.1. Orchestration Frameworks

* **Orchestrator**: Manages the steps in an ML pipeline
    * Runs the pipeline in sequence based on defined conditions
    * Automates execution of each step
        * Development: Simplifies data science tasks and experiment execution
        * Production: Automates pipeline execution based on schedules or triggers

---

### 17.1.1. Kubeflow Pipelines

* **Kubeflow**: ML toolkit for Kubernetes, builds on top of Kubernetes for deploying, scaling, and managing complex systems.
    * _Key features:_
        - Supports various ML frameworks (TensorFlow, PyTorch, MXNet)
        - Deployable to clouds, local, and on-premises platforms
* **Kubeflow Pipelines**: Platform for building, deploying, and managing multistep ML workflows using Docker containers.
    * _Key components:_
        - User interface for managing experiments and tracking results
        - Engine for scheduling multistep ML workflows
        - SDK for defining and manipulating pipelines and components

---

### 17.1.2. Vertex AI Pipelines

* **Serverless Machine Learning Pipelines**: Run Kubeflow and TensorFlow Extended pipelines on Vertex AI Pipelines without provisioning servers.
    * **Automated Infrastructure Management**: Vertex AI Pipelines automatically provisions infrastructure and manages it for you.
    * **Data Lineage and Artifact Tracking**: Track data lineage and understand the factors that resulted in artifacts, such as training data or hyperparameters used.

---

### 17.1.3. Apache Airflow

* **Apache Airflow**: Open source workflow management platform for data engineering pipelines.
    * _Started at Airbnb in 2014 as a solution to manage complex workflows_
    * _Represents workflows as directed acyclic graphs (DAGs) with tasks, dependencies, and data flows._

---

### 17.1.4. Cloud Composer

* *_Cloud Composer_* 
  * A fully managed workflow orchestration service on Apache Airflow
  * Automates Airflow setup and management, freeing up infrastructure focus

* *_Key Benefits_* 
  * Supports ETL/ELT workflows with data-driven task execution

* *_Use Cases_* 
  * Batch workloads with low latency requirements

---

### 17.1.5. Comparison of Tools

* **Kubeflow Pipelines**: _orchestrates ML workflows in any framework_ (_TensorFlow, PyTorch, MXNet_) using Kubernetes.
  * Set up on-premises or in the cloud
  * No need to manage infrastructure

* **Vertex AI Pipelines**: builds on Kubeflow Pipelines and offers:
  * Integrated failure handling with built-in GCP metrics

---

## 17.2. Identification of Components, Parameters, Triggers, and Compute Needs

* **Triggering MLOps Pipelines**: Automate retraining of models with new data on demand, schedule, new data availability, model performance degradation, or statistical property changes.
    * _No new pipelines are deployed; only a new prediction service or trained model is served._
    * New pipeline deployment required for new implementation through CI/CD pipeline.

---

### 17.2.1. Schedule the Workflows with Kubeflow Pipelines

* **Kubeflow Pipelines CI/CD Overview**
  * Uses kfp.Client Python SDK for programmatically operating pipelines
  * Invocations can be scheduled using Cloud Scheduler, triggered by events with Pub/Sub and Cloud Functions, or as part of larger workflows with Cloud Composer or Data Fusion 
  * Alternative schedulers include Argo and Jenkins (via Google Cloud Marketplace)

---

### 17.2.2. Schedule Vertex AI Pipelines

* **Scheduling Vertex AI Pipeline** 
  * With Cloud Scheduler: use an event-driven Cloud Function with HTTP trigger and create a Cloud Scheduler job
  * With Cloud Pub/Sub: use an event-driven Cloud Function with Pub/Sub trigger

---

## 17.3. System Design with Kubeflow/TFX

* *_System Design Overview_* 
    * Kubeflow DSL
    * TFX

---

### 17.3.1. System Design with Kubeflow DSL

* **Overview of Kubeflow Pipelines**: Pipelines are stored as YAML files executed by Argo, with support for Python DSL.
    * _Python DSL_: Allows authoring pipelines using a Python representation of ML workflow operations.
    * _Container setup_: Create container as Python function or Docker container, then sequence and compile into YAML file.

---

### 17.3.2. System Design with TFX

* **TFX Overview**: Google's machine learning platform based on TensorFlow for production-scale workflows
    * _Configuration framework, shared libraries, and orchestration capabilities_
    * Provides TFX pipelines, libraries, and components for building and managing ML workflows
* **TFX Pipeline Components**:
  * *_ExampleGen: ingests and preprocesses input data_*
  * *_Trainer: trains the model with hyperparameter tuning_*
  * *_Evaluator: performs deep analysis and validation of training results_*

---

## 17.4. Hybrid or Multicloud Strategies

### Multicloud and Hybrid Cloud Concepts

* **Integration**: Combining at least two public cloud providers for a unified solution.
* *Examples of GCP Multicloud Features:*
  - Integration with on-premises environments through GCP AI APIs
  - Use of BigQuery Omni to run analytics on data stored in S3 or Azure Blob Storage
* **Hybrid Cloud**: Combining private computing environment and public cloud computing environment.
* *Anthos Features:*
  - BigQuery Omni for querying data without infrastructure management
  - Hybrid AI offering, including Speech-to-Text On-Prem, GKE, Kubeflow Pipelines, and Cloud Run

---

## 17.5. Summary

* **Overview of Orchestration Tools**
  * *Kubeflow, Vertex AI Pipelines, Apache Airflow, and Cloud Composer are used for ML pipeline automation.*
    * *_Each tool has its own strengths and use cases_*


  * *_Vertex AI Pipelines is a managed serverless workflow runner.*_
    
   * *_Cloud Function event triggers can schedule Kubeflow pipelines_*


  * *_Kubeflow and TFX support bringing your own orchestrator or runtime_*

---

## 17.6. Exam Essentials

* **Orchestration Frameworks**: Understanding the different frameworks needed for automating Machine Learning (ML) workflows
    * *_Frameworks_:_* 
        + Kubeflow Pipelines: Know how to run Kubeflow and TFX pipelines, and use Cloud Build to trigger deployments.
    + Vertex AI Pipelines: Understand how to schedule ML workflows, use Cloud Function event triggers, and run on GCP.
        - *_Key features:_* 
            * Trigger ML workflows with Cloud Function event triggers
            * Run on GCP for scalable deployment
    + Apache Airflow: Know how to use Cloud Composer for automating ML workflows
        - *_Key features:_* 

---

            * Use Cloud Composer for workflow automation
    + TFX Pipelines: Understand how to define and orchestrate ML pipelines, and run on Kubeflow or GCP using Vertex AI Pipelines.

---
# 18. Chapter 12Model Monitoring, Tracking, and Auditing Metadata

---

## 18.1. Model Monitoring

* _**Model Drift:**_ A machine learning model's performance may degrade over time due to changes in the environment, input data, or underlying concepts.
    * Can occur due to **concept drift**, where the model's predictions become less accurate as the underlying concept changes
    * Can also occur due to **data drift**, where the distribution of input data shifts over time
* _**Detection and Recovery:**_ It is essential to detect these types of drift and implement methods to recover or retrain the model.

---

### 18.1.1. Concept Drift

* **Concept Drift**: relationship between input variables and predicted variables changes over time
    * _Caused by changing underlying assumptions_
    * Often happens due to data shift, such as adapting to evade detection filters (e.g. email spam)

---

### 18.1.2. Data Drift

* **Data Drift**: Change in input data distribution or schema after model training, affecting performance.
    * Causes include changes in customer demographics, new product labels, or shifts in column meaning (e.g., diabetic diagnosis levels).
    * Monitoring input data and re-evaluating the model with original metrics is a direct way to address drift.

---

## 18.2. Model Monitoring on Vertex AI

* **Model Monitoring in Vertex AI**: 
    * Enables monitoring of skew and drift in model deployments
    * Can detect differences in input feature distributions between training and production data
    * Supports categorical and numerical features, with separate analysis for each

---

### 18.2.1. Drift and Skew Calculation

* **Baseline Distribution**: 
    * Calculated from training data for categorical features: count/percentage of each possible value
    * Calculated from production data for numerical features: count/percentage in buckets (full range divided into equal-sized buckets)
    
    * Latest distribution calculated using same method as baseline for comparison
    
* **Distance Measure**:
    * Categorical features: L-infinity distance between baseline and latest production distribution
    * Numerical features: Jensen-Shannon divergence between baseline and latest production distribution

---

### 18.2.2. Input Schemas

* *_Input Values_*: part of prediction payload
    * *_Parsing_*: Vertex AI parses input values with/without schema
        + *_AutoML Models_*: parsing is automatic, no schema required
        + *_Custom Trained Models_*: require manual schema specification

---

## 18.3. Logging Strategy

* *_Monitoring and logging are essential when deploying a model for prediction_*
* *_Logging is mandatory in some domains (e.g. regulated financial verticals) for future audits_*
* *_Prediction logs can be enabled during deployment or endpoint creation in Vertex AI_*

---

### 18.3.1. Types of Prediction Logs

* *_Log Types_*:
  * **Node Prediction Logs**: capture data from prediction nodes
  * **Model Summary Logs**: provide a summary of the model's state
  * **Event Logs**: record significant events during prediction

---

### 18.3.2. Log Settings

* Logging settings can be updated when creating an endpoint or deploying a model
    * Changes to log settings require undeploying and redeploying the model
* Consider the costs of logging, especially for models with high "Queries per second" (QPS) rates

---

### 18.3.3. Model Monitoring and Logging

* **Infrastructure sharing**: Both model monitoring and request-response logging use the same backend infrastructure (BigQuery table).
    * **Restrictions on enabling services**:
        * Cannot enable both if model monitoring is already active
        * Cannot modify request-response logging once model monitoring is enabled

---

## 18.4. Model and Dataset Lineage

* *_Metadata_*: recording experiment parameters, artifacts, and metrics
* **Key benefits:** 
  * Detect model degradation after deployment
  * Compare hyperparameter effectiveness
  * Track lineage of ML artifacts for auditing and tracing

---

### 18.4.1. Vertex ML Metadata

* **Vertex ML Metadata**: An open-source library for recording and querying metadata for machine learning workflows.
    * **Data Model**:
        * *_Metadata Store_*: The top-level container for all metadata resources, regional and project-scoped.
        * *_Artifacts_*: Entities created by or consumable by ML workflows (e.g. datasets, models).
        * *_Contexts_*: Groups of artifacts and executions that can be queried.
    * *_Terminology_*:
        * *_Execution_*: A step in a machine learning workflow with runtime parameters.
        * *_Events_*: Connect artifacts and executions, capturing artifact lineage and origin.

---

## 18.5. Vertex AI Experiments

* **Experiment Tracking**: Vertex AI Experiments helps track trial variations to find the best model for a use case
    * *_Key Features_*: 
        * Track steps of an experiment run (preprocessing, training, etc.)
        * Track input and output data
    * *_Benefits_*: Analyze results, understand what works, and choose direction

---

## 18.6. Vertex AI Debugging

* **Debugging GPU Issues in Vertex AI** 
    * Install interactive Bash shell in the container
    * Ensure user has correct permissions to access data
    * Enable interactive shells with `enableWebAccess API` set to true

---

## 18.7. Summary

* _Model deployment and monitoring_
	+ Tracking model performance for degradation
	+ Logging strategies in Vertex AI
* _Model tracking and lineage_
	+ Using Vertex ML Metadata
	+ Using Vertex AI Experiments
* _Additional logging strategies not specified_

---

## 18.8. Exam Essentials

* **Monitoring Model Performance**: Understanding the need to monitor model performance after deployment, including tracking changes in input data.
* * _Types of Degradation_ *
  * Data drift
  * Concept drift
* **Logging Strategies**:
  * Logging for tracking deployment performance and creating new training data
  * Using logging in Vertex AI
* **Vertex ML Metadata**: A managed solution for storing and accessing metadata on GCP, including tracking model lineage and artifacts.

---

# 19. Chapter 13Maintaining ML Solutions

---

## 19.1. MLOps Maturity

* **MLOps Journey**: Organizations go through a journey from manual machine learning to fully automated MLOps
    * _Three Phases_: Manual, Strategic Automation, and CI/CD Automation
        + Manual: Experimental training of models with manual pipeline
        + Strategic Automation: Automation using pipelines and features like feature engineering
        + CI/CD Automation: Full automation with continuous integration and deployment

---

### 19.1.1. MLOps Level 0: Manual/Tactical Phase

* **Experimentation Phase**: Organizations start experimenting with ML, building proof of concepts and testing AI/ML use cases to validate business improvements.
	* *Focus on individual or team experimentation and model development*
	* *Models are stored in a model registry for handoff to the release/deployment team*

---

### 19.1.2. MLOps Level 1: Strategic Automation Phase

* **MLOps Level 1 Phase**: Organizations have identified business objectives and prioritize ML to solve problems.
    * _Characteristics_: Automated pipeline, continuous delivery of model prediction service
    * Key services: Automated validation, Feature Store, metadata management, pipeline triggers
    * _Infrastructure_: Shared infrastructure for teams, clear distinction between dev/prod environments

---

### 19.1.3. MLOps Level 2: CI/CD Automation, Transformational Phase

* **Transformational Phase**: Organization uses AI for innovation and agility
    * ML experts in product teams and business units
    * Datasets accessible across silos for collaborative projects
    * CI/CD automation for model updates and pipeline

---

## 19.2. Retraining and Versioning Models

* *_Retraining Model_*: The question of when to retrain a model after performance degradation
    * **Drift Detection**: Monitoring model performance using Vertex AI Model Monitoring to detect changes in performance
    * *_Training Dataset Collection_*: Collecting real data for evaluation and creation of new training datasets

---

### 19.2.1. Triggers for Retraining

* **Retraining Policies**: 
    * _Scheduled training_: Train models at fixed intervals (e.g., weekly or monthly) with predictable costs.
        * Can incorporate new data on a regular basis
    * Trigger retraining based on:
        * _Absolute threshold_: Set a threshold below which accuracy falls, e.g. 90%
            * Example: trigger when accuracy drops below 92%

---

### 19.2.2. Versioning Models

* **Model Versioning**: allows deploying multiple models with version ID selection for backward compatibility
    * *_Solves disruption caused by changes in API behavior_*
    * *_Enables access to older models using REST endpoints_*
        * **Monitoring**: enables comparison of deployed model versions

---

## 19.3. Feature Store

* **Feature engineering is a costly investment**: Creating valuable features manually can take more time than experimenting with ML models.
* *_Main problems with ad hoc feature engineering_*:
    * Features are non-reusable and not automated in pipelines
    * Creates governance issues due to diversity of methods
    * Increases divisions between teams

---

### 19.3.1. Solution

* **Feature Stores**: central location for features and metadata, allowing data engineers and ML engineers to share and collaborate
    * Enables versioning, documentation, and access control using software engineering principles
    * Processes large feature sets quickly and accesses them with low latency for real-time predictions and batch access
        *_Example:* Feast, an open-source Feature Store created by Google and Gojek

---

### 19.3.2. Data Model

**Data Model Overview**

* **Time-series data storage**: data is stored as it changes over time
* **Hierarchy structure**:
	+ Featurestore (container)
	+ Entity Type (container for similar features)
	+ Feature (individual data point)
* Example: create a featurestore called `baseballfs` with an `EntityType` called `batters`, containing features `team`, `batting_avg`, and `age`.

---

### 19.3.3. Ingestion and Serving

* **Ingestion Method**: Supports both batch and streaming ingestion using BigQuery as a data source.
* **Retrieval Methods**:
  * Batch: For model training
  * Online: For online inference
* **Data Retrieval**: Returns values at or before the requested time t.

---

## 19.4. Vertex AI Permissions Model

* **Identity and Access Management (IAM)** is crucial for managing access to resources like datasets, models, and services in ML pipelines.
* Best practices include:
  * _Least privilege_: Restrict users and applications to only necessary actions
  * Manage service accounts and keys: Actively monitor and update security assets
  * Auditing: Enable audit logs and use cloud logging roles

---

### 19.4.1. Custom Service Account

* _Using default service accounts_ can grant unnecessary permissions
* _Customizing service accounts_ allows for precise permission control
* _Requiring minimal permissions reduces risk of data breaches_*

---

### 19.4.2. Access Transparency in Vertex AI

* **Cloud Audit Logs**: capture actions of users from your organization
* *_Access Transparency Logs_*: capture actions of Google personnel
* Most services are supported, but some features may not be covered.

---

## 19.5. Common Training and Serving Errors

* **Error Types**: Understanding common errors in machine learning frameworks helps effective debugging.
* *TensorFlow-specific errors*: Familiarity with these errors is crucial for resolving issues during training and serving.
* _Framework-agnostic insights_: Knowledge of error patterns can be applied across various frameworks, enhancing debugging capabilities._

---

### 19.5.1. Training Time Errors

* *_Common Training Phase Errors_*
  * *_Data Transformation Issues_*: input data must be properly transformed or encoded
  * *_Tensor Shape Mismatch_*: tensor shapes do not match the model's expectations
  * *_Out-of-Memory Errors_*: instance size exceeds available memory (CPU and GPU)

---

### 19.5.2. Serving Time Errors

* *Serving time errors occur only during deployment and have distinct natures.* 
  * Common errors include untransformed input data and signature mismatches.*
  * For a full list, refer to the official TensorFlow error documentation.*

---

### 19.5.3. TensorFlow Data Validation

* **TensorFlow Data Validation (TFDV)**: a tool to prevent and reduce errors in machine learning models
    * Analyzes training and serving data for statistics, schema inference, and anomaly detection
    * Provides full documentation at https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell

---

### 19.5.4. Vertex AI Debugging Shell

* **Interactive Shell**: Debug training with an interactive shell in Vertex AI
    * _Inspect training container_
    * Run tracing and profiling tools
    * Analyze GPU utilization

---

## 19.6. Summary

* **MLOps** is an extension of CI/CD principles for maintaining ML applications
    * **Automation**: automating training, deployment, and monitoring
        * **Retraining policy**: balancing model quality and cost to avoid inefficiencies
    * **Feature sharing**: overcoming departmental barriers with feature stores (e.g. open source or Vertex AI)

---

## 19.7. Exam Essentials

### MLOps Maturity Levels
* **Experimental Phase**: Early-stage MLOps with minimal automation.
* **Strategic Phase**: Introduction of automation tools for model versioning and retraining triggers.
* *_Fully Mature_*: CI/CD-inspired MLOps architecture with robust feature management.

### Feature Management
* *_Feature Store_*: Centralized repository for shared features, reducing duplicate efforts.

---

# 20. Chapter 14BigQuery ML

---

## 20.1. BigQuery – Data Access

* **Accessing Data**: 
  * _Three methods_:
    * Web console using SQL queries
    * Jupyter Notebook with magic command `%%bigquery`
    * Python API for running queries in a Jupyter Notebook
* **Running Queries**:
  * _Common execution locations_: Vertex AI Workbench and Google Cloud Platform

---

## 20.2. BigQuery ML Algorithms

* **BigQuery ML**: A serverless service allowing you to create and deploy machine learning models using standard SQL queries
    * No need for Python code or writing custom scripts
    * Completely serverless, making it easy to use and cost-effective

---

### 20.2.1. Model Training

* **Creating a Model**: 
  * Run `CREATE MODEL` with options for model type and input label columns
  * Use available models (regression, classification, time-series, clustering, etc.)
    * Regression: _LINEAR_REG_, _BOOSTED_TREE_REGRESSOR_, DNN_REGRESSOR, AUTOML_REGRESSION
    * Classification: _LOGISTIC_REG_, _BOOSTED_TREE_CLASSIFIER_, DNN_CLASSIFIER, DNN_LINEAR_COMBINED_CLASSIFIER, AUTOML_CLASSIFIER

* **Model Training**: 
  * Pass query result to training job
  * Automatic calculations of evaluations and metrics (e.g. ROC curve, PR curves)

---

### 20.2.2. Model Evaluation

* To evaluate a model, use `ML.EVALUATE` with a separate unseen dataset.
* Use `SELECT * FROM ML.EVALUATE()` to perform the evaluation query.
* Example: `SELECT * FROM ML.EVALUATE(MODEL 'projectid.test.creditcard_model1', ( SELECT * FROM `test.creditcardtable`))`

---

### 20.2.3. Prediction

* **ML.PREDICT Function**: Makes predictions using a trained BigQuery ML model
    * Passes an entire table to predict and returns a new table with predicted values and probabilities
    * _Example usage_: `SELECT * FROM ML.PREDICT (MODEL 'dataset1.creditcard_model1', (SELECT * FROM 'dataset1.creditcardpredict' LIMIT 1))`

---

## 20.3. Explainability in BigQuery ML

* **Explanation**: Explanation is crucial for debugging models and improving transparency in BigQuery.
* 
    * _Model Explainability Methods_: Different methods are used for explainability, including:
        - Shapley values (Linear & logistic regression)
        - Tree SHAP (Boosted Trees)
        - Integrated gradients (Deep Neural Network & Wide‐and‐Deep)
        - Time‐series decomposition (Arima_PLUS)

---

## 20.4. BigQuery ML vs. Vertex AI Tables

* **Key Difference**: *BigQuery for SQL experts, data analysts, and business users.* 
    * _Vertex AI for machine learning engineers with expertise in Python, Java, and Kubeflow._
        * Fine-grained control required for data flow and training process.

---

## 20.5. Interoperability with Vertex AI

* *_Vertex AI_* and *_BigQuery ML_* are two separate products that work together seamlessly through multiple integration points.

---

### 20.5.1. Access BigQuery Public Dataset

* _BigQuery Public Datasets_: more than 200 publicly available datasets for use on Google Cloud.
* **Access and Cost**: access through GCP project, pay only for queried data.
* *_Integrating with Vertex AI_*: combine datasets for improved machine learning models, e.g. traffic prediction using public weather dataset.

---

### 20.5.2. Import BigQuery Data into Vertex AI

* **Connecting to BigQuery datasets**: Create a Vertex AI dataset directly from a BigQuery URL using the console or Python API.
  * _No need to export and import data_, thanks to the integrated connection.
  * Use `bq://project.dataset.table_name` format in the `create()` method.

---

### 20.5.3. Access BigQuery Data from Vertex AI Workbench Notebooks

* You can browse and interact with your BigQuery dataset directly from a managed notebook instance in Vertex AI Workbench.
* Run SQL queries on the dataset
* Download data into a Pandas DataFrame for further analysis

---

### 20.5.4. Analyze Test Prediction Data in BigQuery

* **Exporting Model Predictions**: Train a model with a train and test dataset, then export test prediction results to BigQuery for analysis.
* *_Benefits_*: Further analyze test predictions using SQL methods in BigQuery.
* *_Use case_*: Utilize exported data to gain insights into model performance.

---

### 20.5.5. Export Vertex AI Batch Prediction Results

* You can make batch predictions in Vertex AI by directly linking to a BigQuery table.
* The predictions are then sent back to BigQuery for storage as a new table.
* This streamlines MLOps pipelines with standardized data.

---

### 20.5.6. Export BigQuery Models into Vertex AI

* You can export BigQuery ML models to GCS for import into Vertex AI
* Direct registration of BigQuery ML models into the Vertex AI Model Registry is now possible
    * Supported models: BigQuery inbuilt models and TensorFlow models
    * Currently limited support for ARIMA_PLUS, XGBoost, and transform-based models

---

## 20.6. BigQuery Design Patterns

* **Machine Learning Design Patterns**: Elegant solutions to frequent data science and machine learning challenges
* *Addressing common issues in data analysis and modeling*
* *_Providing innovative alternatives to traditional approaches_*

---

### 20.6.1. Hashed Feature

* *_Categorical Variable Issues_* 
    * Incomplete vocabulary: Insufficient data values for the categorical variable
    * High cardinality: Large number of unique values, causing scaling issues
    * Cold start problem: New, unseen values added to the dataset
        * *_Solution:_* Hashing with FarmHash algorithm to transform high-cardinality variables into a low-cardinality domain

---

### 20.6.2. Transforms

* **BigQuery ML Pipelines**: Models in BigQuery are built with transformations applied to inputs.
    * Transformations like `FEATURE_CROSS` and `QUANTILE_BUCKETIZE` create new features used by the model.
    * These transformations are automatically added to the model for prediction, without requiring explicit code changes.
* **Available Transforms**: BigQuery ML offers various transforms, including:
  * _POLYNOMIAL_EXPAND_
  * _HASH_BUCKETIZE_
  * _MIN_MAX_SCALER_
  * _STANDARD_SCALER_

---

## 20.7. Summary

* **BigQuery ML**: Democratized machine learning in the SQL community
    * Simplified the ML pipeline and reduced model creation time
    * Highly interoperable with Vertex AI

---

## 20.8. Exam Essentials

* **BigQuery and ML Overview**
    * Learn about BigQuery's history and innovation of integrating machine learning into data analysis
    * Understand how to train, predict, and explain models using SQL
* _Key Differences_
  * BigQuery ML for analysts/SQL experts vs. Vertex AI for ML engineers
* _Integration Points_
  * Seamlessly work between BigQuery ML and Vertex AI services

---

# 23. Online Test Bank

---

## 23.1. Register and Access the Online Test Bank

* *To access online test bank, go to www.wiley.com/go/sybextestprep and follow these steps:*
  * Click "here to register" and select your book from the list
  * Complete registration information and answer security verification for book ownership
* Enter pin code received via email, then login or create a new account
* You will be redirected to test bank site with your book listed on top of page

---

# 24. WILEY END USER LICENSE AGREEMENT

---

