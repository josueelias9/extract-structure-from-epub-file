---
marp: true
theme: default
paginate: true
footer: 'Generated from EPUB'
backgroundColor: #fff
color: #333
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
24. **WILEY END USER LICENSE AGREEMENT** (0 sections)# 1. Table of Contents

---

This book provides a comprehensive guide to Google Cloud Professional Machine Learning Engineer certification, covering essential topics such as problem framing, data exploration, feature engineering, model building, training, and deployment, with an emphasis on scalability, security, and best practices.

---

## 1.1. List of Tables

The summary is: 
Machine learning problem types include structured data, time-series data, and binary classification examples, which are analyzed using metrics such as accuracy, precision, and recall.

---

## 1.2. List of Illustrations

Machine learning (ML) involves creating algorithms to analyze data, but often this process can be tedious and require significant manual work; Google Cloud provides tools like Vertex AI and Kubeflow Pipelines to automate these tasks and streamline workflow, allowing users to easily build, train, deploy and maintain ML models.

---

## 1.3. Guide

Cover Table of Contents Title Page Copyright Dedication Acknowledgments About the Author About the Technical Editors Introduction Begin Reading Appendix: Answers to Review Questions Index End User License Agreement

---

## 1.4. Pages

There is no content to summarize. The provided text consists only of numbers (1-331).

---

# 2. Official Google Cloud CertifiedProfessional Machine Learning EngineerStudy Guide

---

# 3. Acknowledgments

---

# 4. About the Author

---

# 5. About the Technical Editors

---

## 5.1. About the Technical Proofreader

Adam Vincent is an experienced educator certified in Google Cloud, who teaches machine learning courses and automates data, in his free time enjoying hobbies like gaming, reading, and hiking.

---

## 5.2. Google Technical Reviewer

Wiley and the authors wish to thank the Google Technical Reviewer Emma Freeman for her thorough review of the proofs for this book.

---

# 6. Introduction

---

## 6.1. Google Cloud Professional Machine Learning Engineer Certification

A Professional Machine Learning Engineer designs, builds, and productionizes ML models using Google Cloud technologies, ensuring responsible AI and collaborating with cross-functional teams to create scalable solutions for business challenges.

---

### 6.1.1. Why Become Professional ML Engineer (PMLE) Certified?

The Professional Machine Learning Engineer (PMLE) certification from Google provides proof of professional achievement, increases marketability with high-paying job opportunities, recognizes Google's leadership in open-source AI, boosts customer confidence in secure and cost-effective ML solutions.

---

### 6.1.2. How to Become Certified

You can register for the Google Cloud professional certification exam, which takes 2 hours and includes 50-60 multiple-choice questions, without prior company affiliation or prerequisites.

---

## 6.2. Who Should Buy This Book

This book is a comprehensive guide to gaining expertise in machine learning on Google Cloud Platform, covering topics such as feature engineering, model training, deployment, and security, with the goal of helping readers pass the Professional Machine Learning Engineer exam.

---

## 6.3. How This Book Is Organized

This book consists of 14 chapters covering various topics on machine learning (ML) development, deployment, and maintenance, including data exploration, feature engineering, model building, training, scaling, and monitoring, with a focus on Google Cloud infrastructure and tools.

---

### 6.3.1. Chapter Features

The book provides a structured approach to preparing for the Google Cloud certification exam, including chapter objectives, review questions, assessment tests, and study materials, but warns that learning the underlying topic is key to passing the exam and its format may change or be retired in the future.

---

## 6.4. Bonus Digital Contents

The online learning environment for this book includes digital test engine questions, practice exams, flashcards, and a glossary to support in-depth study and review of exam objectives.

---

### 6.4.1. Interactive Online Learning Environment and Test Bank

You can access all these resources at www.wiley.com/go/sybextestprep .

---

## 6.5. Conventions Used in This Book

This book uses specific typographic styles such as italicized text for key terms, monospaced font for configuration files and command names, notes for peripheral information, and tips for useful shortcuts.

---

## 6.6. Google Cloud Professional ML Engineer Objective Map

The book covers a comprehensive framework of low-code machine learning (ML) solutions, including architecting, collaborating within teams, scaling prototypes, serving models, automating pipelines, and monitoring ML solutions, with an emphasis on utilizing Google Cloud services to develop, deploy, and maintain scalable and secure ML workflows.

---

## 6.7. How to Contact the Publisher

Submit any possible errors or mistakes in this book by emailing a correction to wileysupport@wiley.com with "Possible Book Errata Submission" as the subject.

---

## 6.8. Assessment Test

Here are the concise summaries: 1. Split data into training (days 1-29), validation (day 30), and test sets using random data split. 2. Choose AUC PR, AUC ROC, recall, and precision metrics. 3. Create a feature cross by swapping two or more features. 4. Use Cloud Pub/Sub to stream data in GCP and use Cloud Dataflow to transform the data. 5. Recommend Vertex AI AutoML for team members who prefer SQL coding. 6. Benefits of using a Vertex AI managed dataset: integrated data labeling, tracking lineage, automatic data splitting, and de-identification techniques. 7. Choose "Use NLP API, Cloud Speech API, Vision AI, and Video Intelligence AI to identify sensitive data..." 8. Use L1 regularization to reduce features while solving an overfitting problem with large models. 9. To avoid exploding

---

gradients, use batch normalization or lower learning rates. 10. Do not recommend BigQuery connector for real-time streaming data. 11. Recommend Cloud Dataflow connector for real-time streaming data. 12. Avoid grid search and parallelize computations to speed up hyperparameter optimization. 13. Vizier is a service for optimizing complex models with many parameters, but it's not limited to non-ML use cases. 14. Vertex AI TensorBoard and Profiler are tools to track metrics when training neural networks. 15. Use Shapley Integrated gradient XRAI (eXplanation with Ranked Area Integrals) technique to select features with structured datasets. 16. "tf.saved_model.save()" is the command that creates a TensorFlow SavedModel. 17. To deploy a TensorFlow model trained locally to set up real-time

---

prediction using Vertex AI, import the model to Model Registry, deploy the model, create an endpoint for deployed models, and enable monitoring on the endpoint. 18. Vertex AI ML metadata helps track lineage with Vertex AI pipelines. 19. "tf.keras.models.load_model()" is incorrect as it's not a command that creates a TensorFlow SavedModel. 20. To predict revenue by each agent for the next quarter, build a Vertex AI AutoML forecast model, deploy the model, and make predictions using BigQuery ML. 21. To scale an MLOps pipeline, automate training using Vertex AI Pipelines or create a Python script to train multiple models using Vertex AI. 22. There is no reason to use Vertex AI Feature Store that involves extracting features from images and videos stored in Cloud Storage. 23. To predict

---

revenue by each agent for the next quarter efficiently, build a BigQuery ML ARIMA+ model, make predictions in BigQuery, or export the model from BigQuery into the Vertex AI model repository and run predictions in Vertex AI. 24. Automating training using Vertex AI Pipelines is more efficient than retraining an existing model on Vertex AI with the same data and deploying it as part of a CD pipeline. 25. The statement that "Vertex AI supports all models trained on BigQuery ML" is incorrect.

---
## 6.9. Answers to Assessment Test

Here are the summaries: A. Time-series data requires time-based splitting to avoid data or label leakage. B. Precision-recall curves are recommended for imbalanced classes in highly skewed domains. C. A feature cross is created by multiplying two or more features and can be used with one-hot encoded features. D. Cloud Pub/Sub creates a pipeline, while Cloud Dataflow is used for data transformation. E. BigQuery ML is the right approach for performing machine learning using SQL. F. The advantages of managed datasets include integrated labeling, lineage, and automatic labeling features. G. Cloud DLP uses techniques to obscure PII data, but image data requires Vision AI. H. L1 reduces features while L2 seeks stability; vanishing gradients can hinder lower layers training. I. Batch

---

normalization helps prevent exploding gradients with a lower learning rate. J. The Cloud Storage connector is used for on-premises data without needing BigQuery. K. Random search algorithms improve performance in complex models with many parameters. L. Vertex AI Vizier optimizes complex models; it can be used for ML and non-ML use cases. M. Vertex AI hyperparameter tuning does not track metrics but tunes hyperparameters. N. Sampled Shapley is an Explainable AI method suitable for tabular or structured datasets. O. Feature importance explains the features using a score, indicating their usefulness relative to other features. P. SavedModels contain trained parameters and computation in TensorFlow models. Q. Importing models to Vertex AI's Model Registry is required before deploying. R.

---

Vertex ML Metadata records metadata and artifacts produced by an ML system. S. Kubeflow pipelines cannot be invoked using BigQuery; they are for ETL workloads only. T. Vision AI achieves pre-trained models, while options A, B, and C are unnecessarily complex. U. TPU is not suitable for this case due to custom TensorFlow operations; option C is the best choice. V. Porting data into BigQuery is not required for batch processing. W. Access logging provides access logs without timestamps. X. Vertex AI recognizes option B as the correct level of MLOps maturity and recommends automating training with pipelines. Y. A Vertex AI Feature Store deals only with structured data, not images. Z. Building a model in BigQueryML and predicting is the most efficient approach. AA. Option C is the most

---

efficient because it uses an important portability feature between BigQuery and Vertex AI's Model Registry. BB. Porting a model from BigQuery ML TRANSFORM into Vertex AI is incorrect; option A or D can be used instead. CC. Option B is a roundabout way of doing this, making option C the most efficient choice.

---
# 7. Chapter 1Framing ML Problems

---

## 7.1. Translating Business Use Cases

Identifying the impact, success criteria, and available data for a use case is crucial before matching it with a machine learning approach and determining a metric, budget, and timeline.

---

## 7.2. Machine Learning Approaches

Machine learning encompasses a vast array of methods, categorized into specific classes that solve distinct types of problems based on data and prediction types.

---

### 7.2.1. Supervised, Unsupervised, and Semi‐supervised Learning

Supervised machine learning approaches use labeled datasets to train models, while unsupervised methods like clustering and autoencoders group or classify data without prior labeling.

---

### 7.2.2. Classification, Regression, Forecasting, and Clustering

Classification predicts labels or classes from input data, with binary (2-label) and multiclass (more than 2-labels) being distinct types; regression predicts a number (e.g., house price or rainfall), whereas clustering groups similar data points together based on inherent characteristics.

---

## 7.3. ML Success Metrics

In evaluating the performance of a machine learning model, the choice of metric depends on the specific use case, but common metrics for binary classification problems include precision, recall, and F1 score, which can be calculated using formulas that depend on the number of true positives, false positives, true negatives, and false negatives.

---

### 7.3.1. Area Under the Curve Receiver Operating Characteristic (AUC ROC)

The receiver operating characteristic (ROC) curve plots the true positive rate against the false positive rate at different classification thresholds to evaluate the performance of a binary classification model.

---

### 7.3.2. The Area Under the Precision‐Recall (AUC PR) Curve

The area under the precision-recall (AUC PR) curve represents the relationship between recall and precision, with a horizontal line indicating optimal performance at 100% precision and 100% recall.

---

### 7.3.3. Regression

The four common metrics used to evaluate regression models are MAE, RMSE, RMSLE, and R2, which measure differences in absolute or squared errors, and range from 0 to infinity or perfect fit, respectively.

---

## 7.4. Responsible AI Practices

Fairness, interpretability, privacy, and security must be carefully considered in machine learning solutions to ensure they are reliable, trustworthy, and respect user experience.

---

## 7.5. Summary

In this chapter, you learned how to take a business use case and understand the different dimensions to an ask and to frame a machine learning problem statement as a first step.

---

## 7.6. Exam Essentials

To apply machine learning to business challenges, one must understand problem types such as regression, classification, and forecasting, matching data types with suitable algorithms and selecting appropriate metrics like precision, recall, and RMSE.

---

## 7.7. Review Questions

Here are the summaries: 1. When analyzing a potential use case, look for impact, success criteria, algorithm, budget, and time frames. 2. Predicting rainfall amounts is a forecasting problem using machine learning. 3. A model to detect valid support tickets should be a binary classification model. 4. Tracking balls in sports uses image detection and video object tracking. 5. Organizing academic papers without a policy can't cluster or gain insight with current data. 6. The chosen metric for linear regression is MAE, not RMSE, MAPE, or Precision. 7. For house price prediction, choose the metric RMSLE to ensure no extreme errors. 8. For plant classification, choose accuracy as the main metric. 9. Classifying engine cracks in X-ray images requires precision and recall metrics due to rare

---

event. 10. Unsupervised learning is used when data isn't labeled for identification methods. 11. Building a model with unlabeled data uses semi-supervised or collaborative filtering methods. 12. Creating a visual search engine should use supervised learning to classify image content. 13. Extending the dataset and retraining models frequently is not ideal, adding evaluation as part of the pipeline is recommended instead. 14. Hyper-supervised learning isn't a type of machine learning approach. 15. Feeding model output into another model can amplify errors or be done with proper design patterns. 16. Deploying a high-impact model should be followed by integrating it with systems and testing for biases. 17. For model interpretability, use explanations instead of relying on black box models. 18.

---

The Android app's accuracy issue might be due to dataset representation or deployment issues. 19. Training data privacy is still a concern even if the data is just scans and secured behind a firewall. 20. Creating a recommendation model should not use data for creative purposes without considering privacy concerns. 21. Periodically retraining models to adjust performance and include new products is a valid next step. 22. Using product description and images in training can enhance the model's value.

---
# 8. Chapter 2Exploring Data and Building Data Pipelines

---

## 8.1. Visualization

Data visualization is a technique that uses charts to find trends, outliers, and correlations in data, aiding in data cleaning and feature engineering processes through univariate and bivariate analysis methods.

---

### 8.1.1. Box Plot

A box plot visualizes data by displaying the 25th, 50th (median), and 75th percentiles, with the body representing the interquartile range and whiskers indicating maximum and minimum values, while points outside the whiskers are considered outliers.

---

### 8.1.2. Line Plot

A line plot plots the relationship between two variables and is used to analyze the trends for data changes over time. Figure 2.2 shows a line plot. FIGURE 2.2 Line plot

---

### 8.1.3. Bar Plot

A bar plot is used for analyzing trends in data and comparing categorical data such as sales figures every week, the number of visitors to a website, or revenue from a product every month. Figure 2.3 shows a bar plot. FIGURE 2.3 Bar plot

---

### 8.1.4. Scatterplot

A scatterplot is the most common plot used in data science and is mostly used to visualize clusters in a dataset and show the relationship between two variables.

---

## 8.2. Statistics Fundamentals

In statistics, we have three measures of central tendency: mean, median, and mode. They help us describe the data and can be used to clean data statistically.

---

### 8.2.1. Mean

Mean is the accurate measure to describe the data when we do not have any outliers present.

---

### 8.2.2. Median

The median is calculated by finding the middle value in a dataset when it contains an outlier or has an even number of values, resulting in averaging two numbers for even datasets.

---

### 8.2.3. Mode

Mode is used if there is an outlier and the majority of the data is the same. Mode is the value or values in the dataset that occur most. For example, for the dataset 1, 1, 2, 5, 5, 5, 9, the mode is 5.

---

### 8.2.4. Outlier Detection

The addition of an outlier in a dataset significantly affects the mean, causing it to deviate from both the median and mode values.

---

### 8.2.5. Standard Deviation

Standard deviation is the square root of the variance. Standard deviation is an excellent way to identify outliers. Data points that lie more than one standard deviation from the mean can be considered unusual. Covariance is a measure of how much two random variables vary from each other.

---

### 8.2.6. Correlation

The correlation coefficient ranges from –1 to +1 and indicates the degree of relationship between two variables, with 0 indicating no substantial impact, positive correlations showing a rise in one variable accompanies a rise in another, and negative correlations show an opposite trend.

---

## 8.3. Data Quality and Reliability

A model's quality is heavily dependent on the reliability and size of its training data, which requires cleaning and checking for errors such as missing values, duplicates, and noisy features.

---

### 8.3.1. Data Skew

Data skew occurs when a normal distribution curve is asymmetric, resulting in outliers or uneven data distribution, and can be analyzed using statistical measures like mean, median, and standard deviation to determine right-skewed or left-skewed distributions that affect model performance.

---

### 8.3.2. Data Cleaning

The goal of normalization is to transform features to be on a similar scale. This improves the performance and training stability of the model. (See https://developers.google.com/machine-learning/data-prep/transform/normalization .)

---

### 8.3.3. Scaling

Scaling floating-point feature values converts them into a standard range (e.g., 0 to 1), allowing gradient descent to converge better in deep neural network training and preventing "NaN traps".

---

### 8.3.4. Log Scaling

Log scaling is used to scale data that has vastly different ranges by converting large values into a comparable scale through logarithmic transformation.

---

### 8.3.5. Z‐score

The z-score is calculated as a value's distance from the mean in standard deviations, ranging from -3 to +3, with values above or below this range considered outliers.

---

### 8.3.6. Clipping

In the case of extreme outliers, you can cap all feature values above or below to a certain fixed value. You can perform feature clipping before or after other normalization techniques.

---

### 8.3.7. Handling Outliers

An outlier in a dataset is an observation that significantly deviates from the rest of the data points and can be identified through visualization techniques such as box plots, Z-score, clipping, and statistical methods like Interquartile range (IQR).

---

## 8.4. Establishing Data Constraints

A well-defined schema for an ML pipeline provides a consistent framework for feature engineering, data transformation, and validation, enabling metadata-driven preprocessing and anomaly detection.

---

### 8.4.1. Exploration and Validation at Big‐Data Scale

TensorFlow Data Validation (TFDV) uses libraries from TensorFlow Extended (TFX) platform for detecting anomalies and validating schemas of large datasets at scale to support the scalability of deep neural networks.

---

## 8.5. Running TFDV on Google Cloud Platform

TFDV's core APIs are built on Apache Beam, allowing for scalable batch and streaming pipeline management through Google Cloud Dataflow, integrated with BigQuery, Google Cloud Storage, and Vertex AI Pipelines.

---

## 8.6. Organizing and Optimizing Training Datasets

The dataset is typically divided into three parts: a training set for learning from, a validation set for hyperparameter tuning, and a test set used to evaluate model performance after training has been completed without reusing data from these sets.

---

### 8.6.1. Imbalanced Data

Downsampling the majority class (e.g., no-fraud scenarios) and upweighting it can improve model performance by increasing the proportion of minority class examples, resulting in faster model convergence due to reduced bias from oversampling or undersampling.

---

### 8.6.2. Data Splitting

Splitting dataset into training and test sets can cause skew due to natural clustering in data, such as same timeline publication date for similar topics, requiring alternative approaches like splitting based on time of year or specific dates.

---

### 8.6.3. Data Splitting Strategy for Online Systems

Splitting online system training and serving data by time is recommended for mirroring the lag between training and prediction, typically using a 30-day window with separate sets of training (days 1-29) and validation (day 30).

---

## 8.7. Handling Missing Data

Handling missing data involves methods such as removing or deleting rows/columns with null/NaN values, replacing values with mean/mode/median, imputing with most frequent category, and using machine learning algorithms like k-NN and Random Forest to predict missing values based on correlations and non-missing variables.

---

## 8.8. Data Leakage

Data leakage occurs when a machine learning model is trained on both test and training data, resulting in overfitting as the model learns from data it will not encounter during deployment.

---

## 8.9. Summary

Data visualization, statistical analysis, and data preprocessing techniques are crucial in machine learning pipelines, including data cleaning, normalization, schema definition, validation, splitting strategies, handling imbalanced datasets, missing data, and data leakage.

---

## 8.10. Exam Essentials

The ability to effectively manage and process data is crucial for building reliable machine learning (ML) pipelines, involving tasks such as data visualization, quality assessment, schema definition, data splitting, sampling strategies, missing value handling, and avoidance of data leaks.

---

## 8.11. Review Questions

1. You are the data scientist for your company, and you have a dataset of credit card transactions. To improve the performance of your classification model, you should Z-normalize all numeric features. 2. The reason your cancer prediction model performed poorly on new patient data is due to strong correlation between the feature "hospital name" and predicted result. 3. Six months after deploying your deep neural network model, it's performing poorly due to a change in input data distribution; you should create alerts to monitor for skew and retrain your model. 4. The most likely cause of reduced model accuracy is poor quality data or lack of model retraining, possibly with an incorrect data split ratio or missing training data. 5. To resolve the class imbalance problem for faulty sensor

---

readings, you can generate 10 percent positive examples using class distribution or downsample the majority data with upweighting to create 10 percent samples. 6. Your meteorological department's temperature prediction model accuracy dropped from 99% to 70% in production; this may be due to leakage from training data during testing or overfitting, and you should split the training and test data based on time rather than a random split. 7. To overcome gradient optimization difficulties with your neural-network-based project, consider using feature construction, normalization techniques, improving data cleaning by removing features with missing values, or changing hyperparameter tuning steps to reduce the dimension of the test set and have a larger training set. 8. Your model may be

---

overfitting in production due to high correlation between the target variable and a feature, removing features with missing values, or adding your target variable as an additional feature.

---
# 9. Chapter 3Feature Engineering

---

## 9.1. Consistent Data Preprocessing

Applying transformations to data before or within the model during training has two approaches: Pretraining Data Transformation, where transformation is done on the entire dataset before training, and Inside Model Data Transformation, where the transformation is part of the model code and applied during training.

---

## 9.2. Encoding Structured Data Types

A good feature in machine learning should be related to business objectives, predictable, numeric with magnitude, and supported by sufficient examples.

---

### 9.2.1. Why Transform Categorical Data?

Most machine learning (ML) algorithms cannot operate on label data directly due to requiring numeric input and output variables.

---

### 9.2.2. Mapping Numeric Values

Integer and floating‐point data don't need special encoding because they can be multiplied by a numeric weight. You may need to apply two kinds of transformations to numeric data: normalizing and bucketing.

---

### 9.2.3. Mapping Categorical Values

There are ways to convert or transform your categorical data into numerical data so that your model can understand it.

---

### 9.2.4. Feature Selection

Feature selection involves selecting or reducing a subset of input variables to predict a target variable, primarily using dimensionality reduction techniques that minimize noise, overfitting, and computational resources.

---

## 9.3. Class Imbalance

Classification models' accuracy is compromised by false negatives, where correctly identified sick patients are mislabeled as healthy.

---

### 9.3.1. Classification Threshold with Precision and Recall

A classification threshold is a value chosen by humans that determines when a model predicts the positive class, influencing false positives and false negatives; raising the threshold increases precision while lowering it increases recall.

---

### 9.3.2. Area under the Curve (AUC)

Two types of AUC measures are used for classification problems: AUC-ROC (for balanced datasets) and AUC-PR (for imbalanced datasets), which helps to identify classes with unequal examples like credit card transaction fraud detection.

---

## 9.4. Feature Crosses

A feature cross is created by multiplying two or more features to capture nonlinearity in a linear model and improve predictive ability.

---

## 9.5. TensorFlow Transform

Increasing the performance of a TensorFlow model requires an efficient input pipeline. First we will discuss the TF Data API and then we'll talk about TensorFlow Transform.

---

### 9.5.1. TensorFlow Data API (tf.data)

Incorporating best practices such as prefetching, interleaving, caching, and vectorizing into your data input pipeline can significantly improve model execution speed by reducing device idle time and mitigating data extraction overhead.

---

### 9.5.2. TensorFlow Transform

The TensorFlow Transform library allows you to perform transformations prior to model training, emitting a reproducing graph during training, and avoids training-serving skew by utilizing tools like Cloud Dataflow in Google Cloud.

---

## 9.6. GCP Data and ETL Tools

Cloud Data Fusion and Dataprep by Trifacta are Google Cloud services that simplify data transformation and ETL through a web-based, managed interface that supports pipeline building, data cleaning, and scaling without requiring code or infrastructure management.

---

## 9.7. Summary

This chapter covers feature engineering techniques, including numerical transformation (bucketing and normalization), categorical encoding methods, class imbalance and dimensionality reduction, and introduces TensorFlow's tf.data and tf.Transform pipelines for data processing on Google Cloud.

---

## 9.8. Exam Essentials

Learn about data transformation techniques such as encoding, feature selection, cross-validation, and understand when to apply them using tools like TensorFlow Transform, Cloud Data Fusion, and BigQuery.

---

## 9.9. Review Questions

Here are one-sentence summaries of each content: 1. To improve model performance, use feature cross with categorical features, one-hot encoding, and oversampling. 2. To resolve gradient optimization difficulties, combine strongest features, normalize data, clean data by removing missing values, or reduce dimension in test set. 3. For custom fraud detection model, prioritize using an optimization objective that maximizes the area under the precision-recall curve (AUC PR) value. 4. To address AUC ROC score issues, perform hyperparameter tuning to reduce the value or use nested cross-validation. 5. To speed up ResNet training with TPU, modify tf.data dataset by setting batch size equal to training profile read rate and increasing buffer size for shuffle. 6. For low-latency image processing

---

pipeline, convert data into TFRecords, store images in Cloud Storage, and prefetch transformations. 7. To analyze city-specific housing price relationships, use feature crosses such as [binned latitude x binned roomsPerPerson]. 8. To build a unified analytics environment, use a fully managed cloud-native data integration service like Google Cloud Data Fusion or Cloud Dataflow. 9. For model performance tracking with high dynamic customer behavior, use k-fold cross-validation to validate on specific subsets of data before pushing to production. 10. To improve model accuracy from 99% in training to 66% in production, apply transformation during model training, perform data normalization, remove missing values, or create a production pipeline using tf.Transform.

---
# 10. Chapter 4Choosing the Right ML Infrastructure

---

## 10.1. Pretrained vs. AutoML vs. Custom Models

You can use pretrained autoML models or build a custom model using your own data via AutoML, which chooses the best algorithm and requires only formatted data, or opt for a completely custom model with full flexibility but requiring ML expertise.

---

## 10.2. Pretrained Models

You can leverage pre-trained machine learning models available on the Google Cloud platform for various tasks such as vision, natural language, and speech recognition, and use them with just a few minutes of setup through their web console or SDKs.

---

### 10.2.1. Vision AI

Vision AI provides a cloud-based service that allows users to analyze images without creating a machine learning infrastructure, offering features such as object detection, image classification, facial recognition, and "Safe Search" classifications for restricted content.

---

### 10.2.2. Video AI

The API provides real-time object, place, and action recognition in videos, enabling applications such as video tagging, recommendation systems, indexing large archives, and improving ad relevance through comparative analysis.

---

### 10.2.3. Natural Language AI

The Google Natural Language AI provides entity extraction, sentiment analysis, syntax analysis, and categorization services, enabling applications such as customer sentiment analysis, medical insights extraction, and document classification into over 700 predefined categories.

---

### 10.2.4. Translation AI

The service uses Google Neural Machine Translation technology to detect over 100 languages and translate text, documents, and audio between any pairs of languages, with two tiers offering basic and advanced features with differences in price and functionality.

---

### 10.2.5. Speech‐to‐Text

Google's Speech-to-Text service converts recorded or streaming audio into text, often used for creating subtitles and translating video content for multiple languages.

---

### 10.2.6. Text‐to‐Speech

The Text‐to‐Speech service provides realistic speech with humanlike intonation using state‐of‐the‐art expertise from DeepMind, supporting 220+ voices across 40+ languages.

---

## 10.3. AutoML

AutoML (Automated Machine Learning) is a process that automates model training tasks, allowing users to input data and configure settings, with the rest of the training done automatically for common machine learning problems.

---

### 10.3.1. AutoML for Tables or Structured Data

Structured data can be trained using BigQuery ML or Vertex AI Tables, with options including serverless training and prediction through SQL-based queries or Python/Java/Node.js APIs, and various AutoML algorithms such as classification, regression, and forecasting models.

---

### 10.3.2. AutoML for Images and Video

Vertex AI AutoML provides pre-built models for various image and video data types, including single-label classification, multiclass classification, object detection, image segmentation, video classification, action recognition, and object tracking, with optimized edge models available for deployment on iPhones, Android phones, and Edge TPU devices.

---

### 10.3.3. AutoML for Text

Machine learning models for text can be easily built using Vertex AI AutoML, solving problems like text classification, sentiment analysis, entity extraction, and text-to-text translation.

---

### 10.3.4. Recommendations AI/Retail AI

Google Cloud's Retail AI solution uses AutoML to provide a customizable search engine, Vision API for image-based searches, and Recommendations AI to analyze customer behavior and offer relevant product recommendations based on usage data, optimization objectives, and user context.

---

### 10.3.5. Document AI

Document AI extracts text and layout information from scanned or handwritten documents, such as forms and government papers, to identify key details for storage in databases.

---

### 10.3.6. Dialogflow and Contact Center AI

Dialogflow is a conversational AI offering from Google Cloud that provides chatbots and voicebots. This is integrated into a telephony service and other services to provide you with a Contact Center AI (CCAI) solution. Let us look at the services provided by this solution.

---

## 10.4. Custom Training

Training deep learning models on custom hardware with GPUs can significantly accelerate the process, reducing training times by an order of magnitude and making it faster than using a single CPU alone.

---

### 10.4.1. How a CPU Works

A CPU is a flexible processor designed to run a wide range of operations, but its serial computation architecture makes it inefficient for tasks requiring trillions of calculations.

---

### 10.4.2. GPU

GPUs can significantly speed up large matrix multiplications and differential operations by utilizing thousands of arithmetic logic units in parallel, allowing for a speedup of an order of magnitude in time.

---

### 10.4.3. TPU

TPUs are specialized hardware accelerators designed by Google for machine learning workloads, featuring multiple matrix multiply units with thousands of directly connected multiply-accumulators that enable massive matrix processing operations.

---

## 10.5. Provisioning for Predictions

The prediction phase of workload provisioning involves deploying models on servers or processing large batches of data to generate predictions, with continuous scaling being crucial due to varying demand.

---

### 10.5.1. Scaling Behavior

Vertex AI automatically scales up to more prediction nodes when CPU usage on existing nodes exceeds a threshold.

---

### 10.5.2. Finding the Ideal Machine Type

To determine the ideal Compute Engine instance for a custom prediction container, benchmark its performance by calling prediction calls until CPU utilization reaches 90+ percent and consider factors such as model type, resource utilization, latency, throughput requirements, and price to select an instance type that balances cost and performance.

---

### 10.5.3. Edge TPU

The Google-designed Edge TPU coprocessor accelerates machine learning (ML) inference on IoT devices by performing 4 trillion operations per second while consuming only 2 watts of power.

---

### 10.5.4. Deploy to Android or iOS Device

ML Kit provides an easy-to-use package for mobile developers to integrate Google's machine learning expertise into their iOS and Android apps, enabling real-time predictions with low latency and minimal bandwidth usage.

---

## 10.6. Summary

You learned about pretrained models, AutoML models, and different hardware options for training and prediction workloads on Google Cloud, including hardware accelerators and edge device deployment.

---

## 10.7. Exam Essentials

Choose between pretrained models, AutoML, or custom models based on solution readiness, flexibility, and approach, and provision optimized hardware for training and prediction according to the specific requirements of scalability, CPU, memory constraints, and deployment needs.

---

## 10.8. Review Questions

Here are the summaries: 1. To identify objects in photos, use AutoML with Vision AI and customize it for best results, or combine AutoML and a custom model. 2. For translating thousands of pages of legal documents to Spanish and French, use Google's Translate service with human-in-the-loop (HITL) to fix translations, or create a new translation model using AutoML. 3. To create subtitles in English and French from 1,000 hours of video recordings, use AutoML to train a subtitle job model, or create a custom model using the data and run it on GPUs or TPUs. 4. For building a mobile app to classify different kinds of insects, use AutoML to train a classification model, deploy it using AutoML Edge as the method, or use an Android app with ML Kit and deploy it to the edge device. 5. To speed up

---

training of a deep learning object detection model on a TPU, try various hardware options such as Deep Learning VMs, e2-highCPU-16 machines, 8 GPUs, or 1 TPU. 6. For building an image segmentation model using AutoML and Edge, use AutoML to train the image segmentation model with AutoML Edge, or create an Android app with ML Kit and deploy it to the edge device. 7. When training a deep learning model for object detection, try speeding up training by moving to the cloud on Google Cloud using better hardware such as Deep Learning VMs, e2-highCPU-16 machines, 8 GPUs, or 1 TPU. 8. For displaying recommendations on the home page of a website, use Recommendations AI's "Others you may like" model. 9. To increase revenue by showing personalized recommendations while customers check out, use

---

Recommendations AI's "Recommended for you" model with revenue per order as an optimization objective. 10. For engaging customers based on their browsing history, use Recommendations AI's "Recommended for you" model. 11. When translating customer's browsing history to engage them more without any browsing events data, use Recommendations AI's "Others you may like" model. 12. To increase cart size by showing details using Recommendations AI, use the "Others you may like" model with click-through rate as an objective. 13. For building a custom deep learning neural network model in Keras to summarize large documents into 50-word summaries, try creating multiple AutoML jobs, using Cloud Composer to automate multiple jobs, or running multiple jobs on the AI platform and comparing results. 14. To

---

analyze customer calls sentiment without requiring high accuracy, use the pretrained Natural Language API to predict sentiment. 15. For speeding up training of a large deep learning object tracking model in videos using custom TensorFlow operations written in C++ for CPU or GPU, try using TPU-v4 with default settings or recompile the custom operations for TPU. 16. To train models that require 50 GB of memory on GPUs, use options such as n1-standard-64 with 8 NVIDIA_TESLA_P100, e2-standard-32 with 4 NVIDIA_TESLA_P100, or n2d-standard-32 with 4 NVIDIA_TESLA_P100. 17. To deploy a real-time voice translation model to an end device, push the model to Edge TPU devices, recompile it before deployment, or use ML Kit to reduce the size of the model for deployment on Android devices. 18. When using

---

cloud TPUs, valid options include A single TPU VM, HPC cluster of instances with TPU, TPU Pod or slice, or instance with both TPU and GPU. 19. For training large deep learning models (more than 100 GB) that have most values as zero in the matrix, use a TPU because there are no custom TensorFlow operations, try using a TPU Pod due to the size of the model, use an appropriately sized TPUv4 slice, or use GPUs. 20. When building a model to estimate energy usage based on photos and other factors using a large deep learning model with the dataset being very large, consider using a TPU because there are no custom TensorFlow operations, try using a TPU Pod due to the size of the model, use GPUs, or use an appropriately sized TPUv4 slice.

---
# 11. Chapter 5Architecting ML Solutions

---

## 11.1. Designing Reliable, Scalable, and Highly Available ML Solutions

Google Cloud services are used to design, automate, and deploy highly reliable, scalable, and available machine learning (ML) solutions by providing a range of tools for data collection, transformation, training, tuning, experimentation, deployment, monitoring, orchestration, and explanation.

---

## 11.2. Choosing an Appropriate ML Service

Google Cloud ML services are divided into three layers, with AI solutions (Document AI, Contact Center AI, Enterprise Translation Hub) at the top, managed SaaS offerings built on Vertex AI in the middle, and infrastructure such as compute instances and containers at the bottom, requiring manual management for scalability and reliability.

---

## 11.3. Data Collection and Data Management

Google Cloud provides several data stores to handle your combination of latency, load, throughput, and size requirements for features: Google Cloud Storage BigQuery Vertex AI's datasets to manage training and annotation sets Vertex AI Feature Store NoSQL data store

---

### 11.3.1. Google Cloud Storage (GCS)

Google Cloud Storage enables storage of various data types, including image, video, audio, and unstructured data, in large files of at least 100 MB, which can be split into shards to improve read and write throughput.

---

### 11.3.2. BigQuery

Store tabular data in BigQuery, utilizing tools like the Google Cloud console, bq command-line tool, REST API, Vertex AI, and Jupyter Notebooks to facilitate faster processing for training data.

---

### 11.3.3. Vertex AI Managed Datasets

Google Cloud recommends using Vertex AI managed datasets for training custom models instead of ingesting data from storage directly, as they offer advantages such as centralized management, integrated data labeling, and easy tracking of lineage.

---

### 11.3.4. Vertex AI Feature Store

Vertex AI Feature Store is a fully managed repository that organizes, stores, and serves ML features, allowing for fast online predictions, batch exports, and real-time retrieval with benefits including automated computation of feature values and detection of data drifts.

---

### 11.3.5. NoSQL Data Store

Static reference features are stored in optimized NoSQL databases like Memorystore, Datastore, or Bigtable for low-latency singleton read operations with submillisecond latency capabilities suitable for applications requiring real-time user-feature lookup, media processing, and Fintech/Adtech use cases.

---

## 11.4. Automation and Orchestration

Machine learning workflows typically involve data collection, training, evaluation, and deployment phases, which can be automated and orchestrated using tools like Kubeflow Pipelines or Google's Vertex AI Pipelines.

---

### 11.4.1. Use Vertex AI Pipelines to Orchestrate the ML Workflow

Vertex AI Pipelines is a managed service that automates, monitors, and governs your ML systems by orchestrating your workflow in a serverless manner using Kubeflow Pipelines SDK or TensorFlow Extended.

---

### 11.4.2. Use Kubeflow Pipelines for Flexible Pipeline Construction

Kubeflow Pipelines is an open-source framework that enables users to compose, orchestrate, and automate machine learning systems using simple code and integrates with various cloud services like Google Cloud.

---

### 11.4.3. Use TensorFlow Extended SDK to Leverage Pre‐built Components for Common Steps

TensorFlow and TensorFlow Extended SDK are two tools used for building machine learning workflows, with TFX being a more comprehensive framework suitable for production-ready models.

---

### 11.4.4. When to Use Which Pipeline

Vertex AI Pipelines supports Kubeflow Pipelines SDK v1.8.9 or higher and TensorFlow Extended v0.30.0 or higher, providing a more streamlined experience with built-in support for common ML operations and lineage tracking.

---

## 11.5. Serving

After you train, evaluate, and tune a machine learning (ML) model, the model is deployed to production for predictions. An ML model can provide predictions in two ways: offline prediction and online prediction.

---

### 11.5.1. Offline or Batch Prediction

You can use Vertex AI batch prediction to run a batch prediction job processing large batches of data stored in BigQuery or Google Cloud Storage for applications such as recommendations and demand forecasting.

---

### 11.5.2. Online Prediction

The prediction model provides real-time responses in near real-time, with two approaches: synchronous (caller waits for response) and asynchronous (model notifies or polls user), optimizing latency by minimizing model complexity and serving time.

---

## 11.6. Summary

This chapter covers best practices for designing reliable and scalable ML solutions on Google Cloud Platform, including data management, automation, orchestration, and serving models in both batch and real-time modes.

---

## 11.7. Exam Essentials

Design reliable, scalable, and highly available machine learning (ML) solutions by selecting an appropriate Google Cloud AI/ML service, understanding the AI/ML stack, implementing data collection and management, automation, orchestration, and model deployment best practices.

---

## 11.8. Review Questions

Here are the concise summaries: 1. Configure prediction pipeline: Embed client on website, deploy gateway on App Engine, and deploy model on Vertex AI platform. 2. Improve input/output execution performance: Load data into Cloud Bigtable, read data from Bigtable. 3. Configure pipeline for product recommendation system: Pub/Sub ‐> Preprocess(1) ‐> ML training/serving(2) ‐> Storage(3)‐> Data studio/Looker studio for visualization 1 = Dataflow, 2 = Vertex AI platform, 3 = Cloud BigQuery. 4. Port models to Google Cloud with minimal code refactoring: Use Vertex AI platform for distributed training, Create a cluster on Dataproc for training, or Use Kubeflow Pipelines to train on a GKE cluster. 5. Digitize documents with least infrastructure efforts: Use Document AI solution or Vision AI OCR,

---

and store documents in Google Cloud Storage. 6. Configure end-to-end architecture of predictive model: Configure Kubeflow Pipelines, use Vertex AI Platform Training, write accuracy metrics to BigQuery, or use Vertex AI Workbench Notebooks. 7. Design deep neural network for customer purchases: Create multiple models using AutoML Tables, automate training runs using Cloud Composer, run multiple training jobs on the Vertex AI platform, or create an experiment in Kubeflow Pipelines. 8. Architect CI/CD workflow for data refresh: Configure pipeline with Dataflow, start training job on GKE cluster after file is saved in Google Cloud Storage, use App Engine to poll storage bucket, or use Pub/Sub–triggered Cloud Function to start training job. 9. Track and report experiments using API: Use Kubeflow

---

Pipelines API, write accuracy metrics to BigQuery, or use Vertex AI Platform Training with Monitoring API. 10. Use batch prediction functionality on aggregated data: Use the batch prediction functionality of the Vertex AI platform, or deploy model on Compute Engine for prediction, or create a serving pipeline in Cloud Functions. 11. Store TensorFlow storage data from block storage to BigQuery: Use tf.data.dataset reader for BigQuery, BigQuery Python Client library, or BigQuery I/O Connector. 12. Modify code to access BigQuery data using TensorFlow and Keras: Use BigQuery I/O Connector, BigQuery Omni, or tf.data.dataset reader for BigQuery. 13. Organize data in Vertex AI for NLP models: Use Vertex AI–managed datasets, or use BigQuery as a data store, or create a feature store with CSV

---

files. 14. Avoid types of storage in managed GCP environment: Avoid Block storage and File storage in Vertex AI, use Google Cloud Storage instead.

---
# 12. Chapter 6Building Secure ML Pipelines

---

## 12.1. Building Secure ML Systems

Google Cloud provides built-in security features, including encryption that protects data both while it is stored (at rest) and as it is transmitted (in transit).

---

### 12.1.1. Encryption at Rest

Google encrypts machine learning model data stored in Cloud Storage or BigQuery tables, either with customer-managed keys or automatically managed by default, to protect it from corruption and unauthorized access.

---

### 12.1.2. Encryption in Transit

To protect your data as it travels over the Internet during read and write operations, Google Cloud uses Transport Layer Security (TLS).

---

### 12.1.3. Encryption in Use

Confidential Computing protects your data in memory from compromise by encrypting it while it is being processed using tools like Confidential VMs and Confidential GKE Nodes.

---

## 12.2. Identity and Access Management

Vertex AI uses Identity and Access Management (IAM) to manage access to resources at both project and resource levels, providing customization options for permissions such as granting read or write access to specific feature stores or entity types.

---

### 12.2.1. IAM Permissions for Vertex AI Workbench

Vertex AI Workbench is a Google Cloud Platform data science service that uses JupyterLab to explore and access data, offering customizable notebook options and encryption at rest and in transit.

---

### 12.2.2. Securing a Network with Vertex AI

Google Cloud's shared responsibility model places cloud provider responsibility for security threats, while end users must protect their own data and assets in the cloud, with shared fate aiming to foster a collaborative security partnership between providers and customers through secure infrastructure as code deployments, assured workloads, and risk protection.

---

## 12.3. Privacy Implications of Data Usage and Collection

Google Cloud recommends strategies for handling personally identifiable information (PII) and protected health information (PHI), including sensitive data protection measures, with reference to federal regulations like HIPAA's Privacy Rule.

---

### 12.3.1. Google Cloud Data Loss Prevention

The Google Cloud Data Loss Prevention (DLP) API uses techniques such as masking, tokenization, and encryption to de-identify sensitive data in text content, including PII, and provides features like risk analysis, inspection jobs, and templates for configuring de-identification workflows.

---

### 12.3.2. Google Cloud Healthcare API for PHI Identification

The Google Cloud Healthcare API provides a de-identification operation that removes sensitive information, such as 18 protected health identifiers listed in the HIPAA Privacy Rule, from various types of healthcare data using configurable transformations like masking or deletion.

---

### 12.3.3. Best Practices for Removing Sensitive Data

To remove sensitive data from datasets, strategies include creating views, using Cloud DLP, masking out-of-box patterns in images and unstructured content with NLP APIs, applying PCA or dimension-reducing techniques, and coarsening fields such as IP addresses, numeric quantities, zip codes, and locations.

---

## 12.4. Summary

This chapter covers Google Cloud security best practices for machine learning, including encryption, IAM access management, secure ML development techniques, and data protection using Cloud DLP and Cloud Healthcare APIs.

---

## 12.5. Exam Essentials

To build secure machine learning systems in Google Cloud, you need to understand encryption at rest and transit, IAM roles for managing Vertex AI Workbench, network security, differential privacy, federated learning, tokenization, data collection implications, DLP API for PII data masking, Healthcare API for PHI data masking, and best practices for removing sensitive data.

---

## 12.6. Review Questions

Here are the concise summaries: 1. To build an ML fingerprint authentication system securely, recommend Federated learning. 2. Manage Vertex AI resources with restrictive IAM permissions, separate projects for each data scientist, use labels, and set up BigQuery sinks. 3. Use K‐anonymity or Replacement Masking to replace sensitive PII data in a dataset. 4. To access Vertex AI Python library on Google Colab Jupyter Notebook, choose service account key, environment variable, or Vertex AI user role. 5. For managing PII data in GCP, use Cloud DLP or VPC security control. 6. To manage Vertex AI instances for 20 data scientists with minimal effort, use Vertex AI–managed notebooks. 7. Use Cloud DLP to remove PHI from FHIR data for text classification. 8. Stream files to Google Cloud and conduct

---

periodic bulk scans using the DLP API to scan for PII.

---
# 13. Chapter 7Model Building

---

## 13.1. Choice of Framework and Model Parallelism

Trained on large datasets, sophisticated modern deep learning models require multi-node training using either data parallelism or model parallelism to achieve feasible training times.

---

### 13.1.1. Data Parallelism

Data parallelism involves splitting a dataset into parts, assigning them to multiple GPUs or nodes, and using the same parameters for forward propagation, with gradients computed locally and sent back to the main node through synchronous or asynchronous methods.

---

### 13.1.2. Model Parallelism

Model parallelism is achieved by partitioning a deep learning model into multiple GPUs, allowing it to train larger models that fit in individual GPU memories and overcoming memory limitations of single-GPU training.

---

## 13.2. Modeling Techniques

Let's go over some basic terminology in neural networks that you might see in exam questions.

---

### 13.2.1. Artificial Neural Network

Artificial neural networks (ANNs) with one hidden layer, like feedforward neural networks, are primarily used for supervised learning of numerical and structured data in regression problems.

---

### 13.2.2. Deep Neural Network (DNN)

Deep neural networks are artificial neural networks that feature multiple hidden layers between the input and output layers, typically qualifying as such with at least two layers.

---

### 13.2.3. Convolutional Neural Network

Convolutional neural networks (CNNs) are a type of DNN network designed for image input. CNNs are most well‐suited to image classification tasks, although they can be used on a wide array of tasks that take images as input.

---

### 13.2.4. Recurrent Neural Network

Recurrent neural networks (RNNs) are designed for processing sequences of data, offering effective solutions for natural language processing and time-series forecasting tasks with the most popular type being long short-term memory (LSTM) networks that use stochastic gradient descent and a chosen loss function to minimize prediction errors.

---

### 13.2.5. What Loss Function to Use

The choice of loss function in a neural network is directly related to the activation function used in the output layer, with common pairings including mean squared error for regression problems and binary cross-entropy or categorical hinge loss for binary classification.

---

### 13.2.6. Gradient Descent

The gradient descent algorithm calculates the negative of the gradient of the loss curve at each point, then moves in that direction by one step, minimizing loss and reducing it as quickly as possible.

---

### 13.2.7. Learning Rate

Gradient descent algorithms adjust their position by multiplying the current direction (gradient) by a scalar learning rate to determine the next point's displacement.

---

### 13.2.8. Batch

A batch in gradient descent is the total number of examples used to calculate the gradient in a single iteration, which can greatly impact computation speed if too large.

---

### 13.2.9. Batch Size

Batch size is the number of examples in a batch. For example, the batch size of SGD is 1, while the batch size of a mini‐batch is usually between 10 and 1,000. Batch size is usually fixed during training and inference; however, TensorFlow does permit dynamic batch sizes.

---

### 13.2.10. Epoch

An epoch means an iteration for training the neural network with all the training data. In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass. An epoch is made up of one or more batches.

---

### 13.2.11. Hyperparameters

Hyperparameters such as loss, learning rate, batch size, and epoch can be adjusted while training an ML model, with tuning the learning rate being particularly crucial for efficient training times.

---

## 13.3. Transfer Learning

Transfer learning is a machine learning technique that allows a neural network model trained on one problem to be adapted and reused for a related but different problem, saving time and improving performance by leveraging pre-trained knowledge from a similar task.

---

## 13.4. Semi‐supervised Learning

Semi-supervised learning is a type of machine learning that uses a small number of labeled examples and a large number of unlabeled examples to train models, falling between unsupervised and supervised learning.

---

### 13.4.1. When You Need Semi‐supervised Learning

Semi-supervised machine learning techniques can be used to increase the size of training data by labeling known instances and applying a semi-supervised algorithm, but this method may result in less trustworthy outcomes due to the potential for inaccurate labels.

---

### 13.4.2. Limitations of SSL

With a minimal amount of labeled data and plenty of unlabeled data, semi‐supervised learning shows promising results in classification tasks. But it doesn't mean that semi‐supervised learning is applicable to all tasks. If the portion of labeled data isn't representative of the entire distribution, the approach may fall short.

---

## 13.5. Data Augmentation

Machine learning models with many parameters require large amounts of data examples for good performance, leading to the use of data augmentation techniques like flips, translations, or rotations to synthetically generate new data.

---

### 13.5.1. Offline Augmentation

Offline augmentation involves performing transformations on a pre-existing dataset before use, which increases its size by the number of applied transformations.

---

### 13.5.2. Online Augmentation

Online augmentation involves performing data augmentation transformations on mini-batches just before feeding them to a machine learning model, enabling efficient enhancement of large datasets.

---

## 13.6. Model Generalization and Strategies to Handle Overfitting and Underfitting

A model's performance is characterized by its bias (overemphasis on training data) or variance (overfitting to unseen data), both of which are trade-offs: high bias means low variance but poor generalization, while high variance implies low bias but poor performance on test data.

---

### 13.6.1. Bias Variance Trade‐Off

A model's balance between bias and variance is crucial, with underfitting requiring increased capacity and overfitting necessitating specialized techniques to mitigate.

---

### 13.6.2. Underfitting

An underfit model fails to learn a problem due to high bias and low variance, resulting in poor performance on both training and testing datasets, and can be mitigated by increasing model complexity, feature engineering, removing noise, or increasing training epochs.

---

### 13.6.3. Overfitting

An overfitted model has low bias but high variance, requiring techniques such as regularization, dropout, and early stopping to balance training data.

---

### 13.6.4. Regularization

Regularization techniques such as L1 and L2 regularization are used to prevent overfitting by shrinking coefficients towards zero, reducing model size while maintaining generalization in linear models, and preventing features from becoming obsolete or redundant.

---

## 13.7. Summary

This chapter covers key techniques for training neural networks in TensorFlow, including model and data parallelism, loss functions, hyperparameter optimization, transfer learning, semi-supervised learning, data augmentation, and regularization methods to address common issues like underfitting and overfitting.

---

## 13.8. Exam Essentials

Train large neural networks using data parallel or model parallel multinode strategies with distributed training, tune hyperparameters for optimal loss minimization, incorporate transfer learning, semi-supervised learning, data augmentation, and handle overfitting/underfitting through bias-variance trade-off and regularization techniques such as L1/L2 regularization.

---

## 13.9. Review Questions

Here are the concise summaries: 1. Model performing poorly due to input data changes: Retrain with L1 regularization and fewer features. 2. Model overfitting in validation data: Optimize for L1 regularization, dropout parameters. 3. Out of Memory error during training: Reduce batch size, image shape. 4. Loss function for multi-class classification: Categorical cross-entropy. 5. Oscillation in loss during training: Decrease learning rate, increase batch size. 6. Comparing model performance over time: Use Continuous Evaluation feature. 7. Training LSTM-based model efficiently: Modify batch size parameter. 8. Loss function for binary classification with categorical features: One-hot encoding. 9. Problem with custom DNN model: Overfitting, need to regularize using L2. 10. Bias and accuracy in

---

customer satisfaction model: Consider variance. 11. Techniques not suitable for text classification: K-means, Recurrent neural network. 12. Generating avatars with limited data: Data augmentation, Feedforward neural network. 13. Encoding categorical features: One-hot encoding, Embeddings. 14. Problem in ML learning rate and hyperparameter tuning: Hyperparameter tuning. 15. Choosing GPU distribution strategy for cost-effectiveness: MirroredStrategy, ParameterServerStrategy.

---
# 14. Chapter 8Model Training and Hyperparameter Tuning

---

## 14.1. Ingestion of Various File Types into Training

Data for machine learning training can be structured, semi-structured, unstructured, batch or real-time streaming, ranging from a few megabytes to petabyte scale.

---

### 14.1.1. Collect

Google offers Pub/Sub for real-time streaming and messaging, Pub/Sub Lite for cost-optimized streaming, Datastream for change data capture and replication, and the BigQuery Data Transfer Service to load data from various sources such as databases, cloud storage, and SaaS apps into BigQuery.

---

### 14.1.2. Process

Once you have collected the data from various sources, you need tools to process or transform the data before it is ready for ML training. The following sections cover some of the tools that can help.

---

### 14.1.3. Store and Analyze

Google Cloud Storage is recommended for storing various types of machine learning data, such as tabular data (BigQuery, Vertex Data Labeling) and unstructured data (Vertex AI Feature Store), with optimal storage strategies including large container formats, sharded files, and the use of TensorFlow I/O for efficient dataset management.

---

## 14.2. Developing Models in Vertex AI Workbench by Using Common Frameworks

You can create a Jupyter Notebook environment using Vertex AI Workbench to train, tune, and deploy machine learning models with automated shutdown, integration with Cloud Storage and BigQuery, and customizable containers, frameworks, network, and security options.

---

### 14.2.1. Creating a Managed Notebook

To create and manage a Vertex AI notebook, go to the Workbench tab, select New Notebook, enable the Open JupyterLab button after creation, and consider upgrading by manually switching to a new boot disk while preserving data on the secondary data disk.

---

### 14.2.2. Exploring Managed JupyterLab Features

You open a JupyterLab notebook and access various frameworks, including Serverless Spark and PySpark, for building and training models in a managed environment.

---

### 14.2.3. Data Integration

Click the Browse GCS icon on the left navigation bar ( Figure 8.7 ) to browse and load data from cloud storage folders. FIGURE 8.7 Data integration with Google Cloud Storage within a managed notebook

---

### 14.2.4. BigQuery Integration

Click the BigQuery icon on the left as shown in Figure 8.8 to get data from your BigQuery tables. The interface also has an Open SQL editor option to query these tables without leaving the JupyterLab interface. FIGURE 8.8 Data Integration with BigQuery within a managed notebook

---

### 14.2.5. Ability to Scale the Compute Up or Down

Click n1‐standard‐4 (see Figure 8.9 ). You will get the option to modify the hardware of the Jupyter environment. You can also attach a GPU to this instance without leaving the environment.

---

### 14.2.6. Git Integration for Team Collaboration

You can integrate an existing git repository or clone one using the left navigation branch icon, terminal command "git clone <your-repository-name>", or by cloning your own repository manually.

---

### 14.2.7. Schedule or Execute a Notebook Code

You can execute a Python cell manually by clicking the triangle black arrow or automatically by clicking Execute as the notebook is submitted to an Executor, which allows scheduling of Vertex AI training jobs or deployment of Prediction endpoints from within Jupyter.

---

### 14.2.8. Creating a User‐Managed Notebook

You can create a TensorFlow notebook with advanced options for networking and access to Git integration and terminal access without needing large hardware or compute instances, as user-managed notebooks provide a more flexible environment for training and prediction tasks.

---

## 14.3. Training a Model as a Job in Different Environments

Vertex AI offers two types of training: AutoML, which automates model creation and training with minimal technical effort, and custom training, which allows full control over training application functionality.

---

### 14.3.1. Training Workflow with Vertex AI

You can create training jobs or resources for AutoML or custom models using Vertex AI's training pipelines, custom jobs, and hyperparameter tuning jobs, which orchestrate custom training with worker pools, machine types, and Python training application settings.

---

### 14.3.2. Training Dataset Options in Vertex AI

You can store datasets in Google Cloud Storage or BigQuery, or use a managed dataset that offers centralized management, automated data splitting, label creation, and governance features.

---

### 14.3.3. Pre‐built Containers

You can set up a Vertex AI training pipeline with pre-built containers by uploading your local Python code to a Cloud Storage bucket and specifying it in the container registry, then using the gcloud ai custom-jobs create command to build a Docker image based on the pre-built container and push it to Container Registry.

---

### 14.3.4. Custom Containers

You can build and run a custom container to train with any ML framework, giving you extended support for distributed training and the ability to use the newest version of an ML framework.

---

### 14.3.5. Distributed Training

You can specify multiple worker pools in a distributed training job on Vertex AI by defining different machine types, replica counts, and accelerator configurations.

---

## 14.4. Hyperparameter Tuning

Hyperparameters are adjustable settings of the training algorithm that are not learned during training, such as the learning rate in gradient descent.

---

### 14.4.1. Why Hyperparameters Are Important

Bayesian optimization and random search algorithms from Vertex AI are used to efficiently search for optimal hyperparameters in neural networks by considering past evaluations and exploring randomized combinations of parameters to maximize predictive accuracy.

---

### 14.4.2. Techniques to Speed Up Hyperparameter Optimization

Using techniques like parallel training, pre-computing, and grid search reduction can increase hyperparameter optimization speed by a factor of ~k (validation set), ~n (distributed training with n machines), or up to a large speedup via reduced number of hyperparameter values considered.

---

### 14.4.3. How Vertex AI Hyperparameter Tuning Works

Hyperparameter tuning is performed by running multiple trials with specified hyperparameters within limits, using a training application and communicating with Vertex AI for optimization of numeric target variables called hyperparameter metrics.

---

### 14.4.4. Vertex AI Vizier

Vertex AI Vizier is a black-box optimization service that helps tune hyperparameters in complex ML models, optimizing tasks such as learning rate, batch size, and model parameters.

---

## 14.5. Tracking Metrics During Training

In the following sections, we will cover how you can track and debug machine learning model metrics by using tools such as an interactive shell, the TensorFlow Profiler, and the What‐If Tool.

---

### 14.5.1. Interactive Shell

You can access and debug your Vertex AI container by using an interactive shell, which allows browsing file systems, running debugging utilities, and analyzing GPU usage while the job is in the RUNNING state, but loses access once the job completes.

---

### 14.5.2. TensorFlow Profiler

Vertex AI TensorBoard Profiler is a feature that helps monitor and optimize model training performance by analyzing resource consumption to pinpoint and fix performance bottlenecks.

---

### 14.5.3. What‐If Tool

You can use the What-If Tool to visualize and interact with AI Platform Prediction models through an interactive dashboard by installing the witwidget library and configuring a WitConfigBuilder, then passing it to WitWidget.

---

## 14.6. Retraining/Redeployment Evaluation

Machine learning models slowly degrade in performance over time due to changes in user behavior and training data, primarily caused by data drift and concept drift.

---

### 14.6.1. Data Drift

Data drift occurs when the statistical distribution of production data changes from the baseline data used to train a model, often due to changes in input data such as unit conversions.

---

### 14.6.2. Concept Drift

Concept drift occurs when the statistical properties of a target variable change over time, necessitating continuous monitoring of deployed models to detect shifts, such as changes in sentiment patterns.

---

### 14.6.3. When Should a Model Be Retrained?

The optimal retraining strategy depends on the chosen interval (periodic), threshold-based trigger (performance), data changes, or manual retraining on demand.

---

## 14.7. Unit Testing for Model Training and Serving

Testing machine learning models involves testing both the model code and data through unit tests, explicit checks for expected behaviors, and various types of numerical checks including output shape, range, gradient descent progress, dataset assertions, and label leakage detection.

---

### 14.7.1. Testing for Updates in API Calls

You can test updates to an API call by retraining your model, but that would be resource intensive. Rather, you can write a unit test to generate random input data and run a single step of gradient descent to complete without runtime errors.

---

### 14.7.2. Testing for Algorithmic Correctness

To verify algorithmic correctness, train your model multiple times with decreasing loss, run specific subcomputations through exhaustive testing, and ensure complex models don't memorize training data.

---

## 14.8. Summary

Google Cloud Platform provides various services for storing and processing file types, including Pub/Sub, BigQuery Data Transfer Service, and Vertex AI training with scikit-learn, TensorFlow, PyTorch, XGBoost frameworks.

---

## 14.9. Exam Essentials

Understand how to ingest various file types into Google Cloud Platform for AI/ML workloads, including data transformation using services like Cloud Dataflow and Cloud Data Fusion, model training with frameworks like scikit-learn and TensorFlow, and hyperparameter tuning with search algorithms like grid and random search.

---

## 14.10. Review Questions

Here are the concise summaries of each scenario in one sentence: 1. Use Vertex AI custom jobs for training to minimize code refactoring and infrastructure overhead. 2. Translate the normalization algorithm into SQL for use with BigQuery or normalize the data with Apache Spark using the Dataproc connector. 3. Create multiple models using AutoML Tables or run multiple training jobs on the Vertex AI platform with an interactive shell enabled. 4. Decrease the number of parallel trials, change the search algorithm from grid search to random search, or decrease the range of floating-point values in hyperparameter tuning jobs. 5. Use Data Fusion's GUI or convert PySpark commands into Spark SQL queries to transform data and run pipelines on Dataproc. 6. Configure Kubeflow to run on Google

---

Kubernetes Engine or create containerized images on Compute Engine using GKE. 7. Load the dataset into Cloud Bigtable or store the data in Cloud Storage as TFRecords to improve input/output execution performance. 8. Use Pub/Sub with Cloud Dataflow streaming pipeline or Apache Kafka with Cloud Dataproc to ingest player interaction data and perform ML. 9. Store results for analytics and visualization in Data Studio using Vertex AI AutoML, BigQuery, or Cloud Storage. 10. Ingest data into BigQuery from Cloud Storage and use BigQuery SQL queries to transform the data and write transformations to a new table. 11. Use pandas read_csv to ingest the CSV file as a pandas DataFrame or export the table as a CSV file from BigQuery to Google Drive. 12. Configure the pipeline using Pub/Sub, Dataflow, and

---

BigQuery for anomaly detection in real-time sensor data. 13. Ingest data into Cloud SQL or use Vertex AI AutoML and Cloud Storage to build a support requests classifier. 14. Use TensorBoard or TensorFlow Profiler with Jupyter Notebooks to track metrics of the model during training. 15. Use Cloud Dataproc to manage input while training TensorFlow models on Google Cloud. 16. Create multiple models using AutoML Tables or run multiple training jobs on the Vertex AI platform with an interactive shell enabled to demonstrate classification metric and inference.

---
# 15. Chapter 9Model Explainability on Vertex AI

---

## 15.1. Model Explainability on Vertex AI

The level of explanation required from machine learning (ML) model developers increases with the impact of predictions on business outcomes, and in critical applications such as lending or medicine, model developers are responsible for justifying predictions to users.

---

### 15.1.1. Explainable AI

Explainability in machine learning refers to the ability to interpret and understand the internal workings of an ML or deep learning system, enabling trust, transparency, and improved model performance through the explanation of individual predictions and overall model behavior.

---

### 15.1.2. Interpretability and Explainability

Interpretability focuses on understanding the causal relationships between inputs and outputs, while explainability targets the explanation of the decisions made by complex machine learning models, particularly those with hidden layers in deep neural networks.

---

### 15.1.3. Feature Importance

Feature importance explains the relative value of each attribute in predicting an outcome, allowing for easy removal of unimportant variables to reduce compute and infrastructure costs and prevent data leakage.

---

### 15.1.4. Vertex Explainable AI

Vertex Explainable AI integrates feature attributions into Vertex AI, providing insights into how each feature in the data contributed to predicted results for classification and regression tasks.

---

### 15.1.5. Data Bias and Fairness

Bias and unfairness exist in machine learning models when data is not collected or represented accurately, leading to skewed outcomes that can result in discriminatory treatment of individuals based on characteristics such as race or gender.

---

### 15.1.6. ML Solution Readiness

Google uses Responsible AI principles, fairness best practices, and tools like Explainable AI, Model cards, and TensorFlow to ensure model governance, including human oversight, responsibility assignment matrices, and data evaluation.

---

### 15.1.7. How to Set Up Explanations in the Vertex AI

For custom-trained models (TensorFlow, Scikit, or XGBoost), configure explanations to support Vertex Explainable AI by sending synchronous requests (online explanations) or optional batch explanations using `projects.locations.endpoints.explain` and setting `generateExplanation` to true in batch prediction jobs.

---

## 15.2. Summary

Explainable AI involves understanding why machine learning models make predictions by analyzing feature importance, addressing data bias and fairness, and using techniques like Shapley, XRAI, and integrated gradients to attribute model outputs to specific input features on platforms like Vertex AI.

---

## 15.3. Exam Essentials

Explainability in Vertex AI refers to techniques such as Sampled Shapley algorithm, integrated gradients, and XRAI that help understand model decision-making processes by attributing feature importance, enabling data bias and fairness analysis and supporting Responsible AI and ML governance best practices.

---

## 15.4. Review Questions

Here are the concise summaries: 1. Remove non-informative features by applying principal component analysis (PCA) or using techniques like L1 regularization, Shapley values, and iterative dropout. 2. Use Sampled Shapley Integrated gradients PCA, What-if Tool, or analysis for demonstrating inner workings of a TensorFlow deep neural network model on Google Cloud. 3. Select AutoML Tables, Custom DNN models, and Decision trees as model types supported by Vertex Explainable AI. 4. Utilize XRAI, Sampled Shapley, and Minimum likelihood Interpretability for feature attribution techniques with image and tabular datasets in Vertex Explainable AI. 5. Employ Integrated gradients and XRAI to debug or explain a TensorFlow model with graph operations on Vertex AI. 6. Use AutoML images' integrated

---

gradients and XRAI DNN for feature attribution of images on Vertex AI. 7. Set up local explanations using the Explainable AI SDK in user-managed notebooks or configure custom TensorFlow models with explanations. 8. Apply the Sampled Shapley technique to provide explanations for poor-quality image datasets used in model identification tasks like plane detection.

---
# 16. Chapter 10Scaling Models in Production

---

## 16.1. Scaling Prediction Service

A trained TensorFlow model can be deployed after training by saving it as a directory containing the model's parameters and computation graph in a format called protocol buffer, which does not require the original model building code.

---

### 16.1.1. TensorFlow Serving

TensorFlow Serving allows hosting trained models as API endpoints through model servers, providing REST and gRPC APIs for serving models with version management and source loading capabilities.

---

## 16.2. Serving (Online, Batch, and Caching)

In Chapter 5 , “Architecting ML Solutions,” we covered two types of serving options in ML systems, batch prediction (or offline serving) and online prediction, and their recommended architectures. In this chapter, we will cover some best practices for your serving and caching strategy.

---

### 16.2.1. Real‐Time Static and Dynamic Reference Features

Real-time input features are computed on the fly in an event-stream processing pipeline and include customer ID, movie ID, and aggregated values for a specific window, used for applications such as predicting engine failure or recommending news articles based on user behavior.

---

### 16.2.2. Pre‐computing and Caching Prediction

A pre-computed predictions architecture stores batch-scoring results in low-latency data stores for clients to fetch predictions by referencing unique keys, reducing online latency and handling high-cardinality entities through hybrid approaches like precomputing top-N entities for direct online use.

---

## 16.3. Google Cloud Serving Options

In Google Cloud, you can deploy your models for either online predictions or batch predictions. You can perform batch and online predictions for both AutoML and custom models. In the following sections, we will cover how to set up online and batch jobs using Vertex AI.

---

### 16.3.1. Online Predictions

You can deploy and create a real-time prediction endpoint using either models trained in Vertex AI or importing existing models from on-premise, another cloud, or local device, with specific requirements for model artifacts and container image creation.

---

### 16.3.2. Batch Predictions

You can create batch predictions by running a job on Google Cloud Storage using a model trained in Vertex AI, specifying input data in formats such as JSON Lines, TFRecord, CSV, file list, or BigQuery table.

---

## 16.4. Hosting Third‐Party Pipelines (MLflow) on Google Cloud

MLflow is an open-source platform for managing the end-to-end machine learning life cycle, offering components such as tracking experiments, packaging models, and managing model registries, which can be utilized individually or in combination with Google Cloud's scalability and high-performance computing resources.

---

## 16.5. Testing for Target Performance

Testing a model in production requires checking training-serving skew, monitoring model age and stability, and writing tests to ensure numerical stability of weights and layer outputs, such as detecting NaN values or zero outputs.

---

## 16.6. Configuring Triggers and Pipeline Schedules

Cloud Scheduler can be used to set up a cron job schedule, while Cloud Build is ideal for retraining models or building Dockerfiles and pushing them for custom training, with options including Cloud Functions, Cloud Pub/Sub, Cloud Workflows, and Vertex AI pipelines.

---

## 16.7. Summary

TF Serving is used to scale prediction services by utilizing static and dynamic reference architectures, caching, and pre-computing, while also offering deployment options for online and batch mode with tools like Vertex AI Model Monitoring.

---

## 16.8. Exam Essentials

Here is a one-sentence summary of the content:

TensorFlow Serving is a system for deploying and serving trained machine learning models, offering options for scaling prediction services, caching, and integrating with Google Cloud services like Vertex AI for model monitoring, scheduling, and automation.

---

## 16.9. Review Questions

Here are one-sentence summaries of each content: 1. Implement a prediction pipeline by embedding the client on the website, deploying the gateway on App Engine, caching predictions in a data store using batch prediction jobs, and deploying the model using Vertex AI Prediction. 2. Use a batch prediction job on Vertex AI pointing to input data as a BigQuery table for minimizing computational overhead with saved models and Dataflow integration. 3. Serve model predictions by creating a Pub/Sub topic for each user, deploying a Cloud Function or App Engine application that sends notifications when account balance is predicted to drop below threshold. 4. Write the correct predict request using json.dumps({'signature_name': 'serving_default', 'instances': [[...]]}). 5. Use a Cloud Function

---

triggered by Pub/Sub messages or a Dataflow job to execute preprocessing logic before submitting predictions to Vertex AI platform. 6. Validate model accuracy and create a new real-time model on Vertex AI platform for online prediction of digitized scanned customer forms in aggregated data. 7. Schedule batch jobs using schedule function in Vertex AI managed notebooks, serving pipeline, Cloud Functions, or Cloud Workflow to run Jupyter Notebook at the end of each day. 8. Configure the prediction pipeline by embedding client on website, deploying gateway on App Engine, caching predictions in data store, and deploying model using Vertex AI Prediction for online news article recommendation.

---
# 17. Chapter 11Designing ML Training Pipelines

---

## 17.1. Orchestration Frameworks

An orchestrator is used to manage the various steps in an ML pipeline, such as data cleaning and training a model, by automatically moving from one step to the next based on predefined conditions, simplifying manual execution and automation of pipeline execution during development and production phases.

---

### 17.1.1. Kubeflow Pipelines

Kubeflow Pipelines is a platform for building, deploying, and managing multistep machine learning workflows using Docker containers, comprising a user interface, scheduling engine, SDK, notebooks, and orchestration tools.

---

### 17.1.2. Vertex AI Pipelines

You can run Kubeflow or TFX pipelines serverless on Vertex AI Pipelines, which automatically provisions and manages underlying infrastructure, provides data lineage, and automates machine learning workflows with portable and scalable components based on containers.

---

### 17.1.3. Apache Airflow

Apache Airflow is an open-source workflow management platform that lets users build, run, and manage data engineering pipelines as directed acyclic graphs (DAGs) of tasks with dependencies and data flows.

---

### 17.1.4. Cloud Composer

Cloud Composer is a fully managed workflow orchestration service built on Apache Airflow that automates and simplifies the creation and execution of data-driven workflows with no installation or management overhead.

---

### 17.1.5. Comparison of Tools

Kubeflow Pipelines offers managed orchestration of ML workflows in any supported framework, with optional infrastructure setup and integration with Apache Airflow for ETL/ELT pipelines, using Kubernetes, while also enabling failure handling through its native support and compatibility with Vertex AI pipelines.

---

## 17.2. Identification of Components, Parameters, Triggers, and Compute Needs

Automate ML production pipelines to retrain models on demand, schedules, or data availability by triggering continuous training (CT) pipelines that deploy new prediction services without redeploying components or new pipelines.

---

### 17.2.1. Schedule the Workflows with Kubeflow Pipelines

Kubeflow Pipelines utilize a Python SDK to programmatically operate pipelines using services such as Cloud Scheduler, Pub/Sub, and Cloud Functions for scheduling and event-driven invocations.

---

### 17.2.2. Schedule Vertex AI Pipelines

You can schedule Vertex AI pipeline execution with either Cloud Scheduler by triggering an HTTP or Pub/Sub event, or using Cloud Functions with a Pub/Sub trigger, allowing for precompiled pipelines to run automatically at set times or in response to events.

---

## 17.3. System Design with Kubeflow/TFX

In the following sections, we will discuss system design with the Kubeflow DSL, and then we will cover system design with TFX.

---

### 17.3.1. System Design with Kubeflow DSL

Kubeflow Pipelines uses Argo Workflows to execute pipelines created with a custom Python domain-specific language (DSL) that allows for automated creation of containers and operations without explicit container configuration.

---

### 17.3.2. System Design with TFX

A TFX pipeline is a sequence of components, including ExampleGen, Trainer, Evaluator, ModelValidator, and Pusher, designed for scalable and high-performance machine learning tasks using TensorFlow and orchestrateable via Apache Airflow or Kubeflow Pipelines.

---

## 17.4. Hybrid or Multicloud Strategies

BigQuery Omni and other GCP tools enable multicloud operations by allowing data transfer between public cloud providers (AWS, Azure) and Google Cloud regions for analytics, machine learning, and training purposes, while Anthos provides a hybrid cloud platform for private on-premises environments to integrate with public clouds for AI development.

---

## 17.5. Summary

Orchestration for machine learning pipelines is covered using tools such as Kubeflow, Vertex AI Pipelines, Apache Airflow, and Cloud Composer, with each tool offering managed serverless options, scheduling methods, and integration with other services like TensorFlow and BigQuery.

---

## 17.6. Exam Essentials

Understand the different orchestration frameworks including Kubeflow Pipelines, Vertex AI Pipelines, Apache Airflow, and Cloud Composer, to automate ML workflows, comparing components, parameters, triggers, compute needs, and scheduling methods for each framework.

---

## 17.7. Review Questions

Here are the concise summaries: 1. To validate models trained with 100+ features, use k-fold cross-validation with L1 regularization and track AUC ROC as the main metric, then deploy model upon production readiness. 2. Design CI/CD workflow using Cloud Storage triggers, Pub/Sub topics, and Cloud Scheduler to automatically refresh Kubeflow Pipelines job on GKE. 3. Configure Kubeflow Pipelines with Dataflow and use App Engine for lightweight polling of Cloud Storage for new files. 4. Set up pipeline on Google Cloud with the least effort by using Vertex AI Pipelines or Kubeflow Pipelines on GKE. 5. Automate unit tests execution with custom libraries by setting up automated triggers in Cloud Source Repositories using Cloud Build and Cloud Function. 6. To scale a training pipeline on-premises,

---

use Anthos to set up Kubeflow Pipelines or Vertex AI Training jobs on GKE.

---
# 18. Chapter 12Model Monitoring, Tracking, and Auditing Metadata

---

## 18.1. Model Monitoring

After deployment, machine learning models can become less accurate over time due to changes in their environment, a phenomenon known as drift, which requires techniques to detect and recover from concept drift and data drift.

---

### 18.1.1. Concept Drift

Concept drift occurs when the relationship between input variables and predicted variables changes over time due to shifting underlying assumptions or patterns in the data, often resulting from attempts by adversaries to evade detection.

---

### 18.1.2. Data Drift

Machine learning models can deteriorate over time due to changes in the input data distribution, schema, or column meaning, requiring continuous monitoring and re-evaluation of the model using the original training metrics.

---

## 18.2. Model Monitoring on Vertex AI

Vertex AI offers model monitoring features to detect training-serving skew and drift in categorical and numerical feature types, enabling the detection of performance impacts due to changes in input distributions over time.

---

### 18.2.1. Drift and Skew Calculation

Vertex AI calculates baseline distributions for skew and drift detection using statistical analysis of training and production data, comparing them to identify anomalies via L-infinity distance and Jensen-Shannon divergence.

---

### 18.2.2. Input Schemas

You can configure a schema when monitoring model predictions in Vertex AI to help parse the input values, and this parsing is automatically done for AutoML models but requires manual configuration for custom-trained models.

---

## 18.3. Logging Strategy

You can log requests for deployed AutoML models in Vertex AI for future auditing and updating training data, enabling it as part of model deployment or endpoint creation.

---

### 18.3.1. Types of Prediction Logs

You can enable three kinds of logs to get information from the prediction nodes. These three types of logs are independent of each other and so can be enabled or disabled independently.

---

### 18.3.2. Log Settings

You can update log settings for an endpoint when deploying or redeploying a model, and should consider the costs of logging due to high QPS rates.

---

### 18.3.3. Model Monitoring and Logging

You cannot enable both model monitoring and request-response logging on the same endpoint due to shared backend infrastructure restrictions.

---

## 18.4. Model and Dataset Lineage

The metadata of an ML experiment, including parameters, artifacts, and metrics, is essential for recording and tracking the lineage of experiments, comparing hyperparameters, detecting model degradation, and auditing downstream usage.

---

### 18.4.1. Vertex ML Metadata

Vertex ML Metadata utilizes a graph-like data model, consisting of metadata store, artifacts (datasets, models, etc.), context (grouped executions with parameters), execution (step in workflow), and events (connecting artifacts and executions), to analyze, debug, or audit machine learning workflows using the open source ML Metadata library.

---

## 18.5. Vertex AI Experiments

Vertex AI Experiments helps develop ML models by tracking various trials, analyzing different variations, and providing a single pane of glass to understand what works and choose the next direction.

---

## 18.6. Vertex AI Debugging

You can debug issues in your model training by connecting to the container where training is running, requiring permissions check, web access enablement, and utilization of profiling tools like py-spy for Python execution analysis and perf for CPU and GPU performance evaluation.

---

## 18.7. Summary

The chapter covers steps beyond deploying a model, including monitoring for performance degradation, logging strategies, and tracking model lineage using Vertex ML Metadata and Vertex AI Experiments.

---

## 18.8. Exam Essentials

Monitor the performance of deployed machine learning models for degradation caused by data drift and concept drift, use logging to track deployment and create new training data in Vertex AI, and utilize ML Metadata to store and access model lineage on GCP.

---

## 18.9. Review Questions

1. Data drift occurs when there's a difference in input feature distribution between training and production data. 2. You need to enable monitoring to establish if the input data has drifted from the train/test data. Monitoring is necessary no matter how well it might have performed on test data. 3. Concept drift happens when the underlying concept or pattern in the data changes over time. 4. Distribution drift occurs when the statistical distribution of input features in production data changes over time. 5. Baseline statistical distribution of input features in training data and baseline statistical distribution of input features in production data are needed to detect prediction drift. 6. L‐infinity distance is used for categorical features in Vertex AI. 7. Periodically switching off

---

monitoring to save money is a valid approach. 8. Sampling rate: Configure a prediction request sampling rate, Monitoring frequency: Rate at which model's inputs are monitored and Choose different distance metrics are features of Vertex AI model monitoring. 9. Custom model with automatic schema parsing without values in key/value pairs is not a correct combination of model building and schema parsing in Vertex AI model monitoring. 10. Number is not a valid data type in the model monitoring schema. 11. Container logging is not a valid logging type in Vertex AI. 12. You can get a log of a sample of the prediction requests and responses by using Input logging. 13. To compare the effectiveness of different sets of hyperparameters, track lineage or find the right proportion of train and test

---

data are valid reasons for using a metadata store. 14. An artifact is any piece of information in the metadata store that can be created by or consumed by an ML workflow. 15. Workflow step is not part of the data model in a Vertex ML metadata store.

---
# 19. Chapter 13Maintaining ML Solutions

---

## 19.1. MLOps Maturity

Machine learning organizations progress through three phases of maturity in MLOps, starting with manual training and model evaluation, to strategic automation and full CI/CD automation.

---

### 19.1.1. MLOps Level 0: Manual/Tactical Phase

Organizations in the experimentation phase of ML development focus on building proof-of-concepts, testing AI/ML use cases, and training individual models that are later deployed for prediction serving through a model registry.

---

### 19.1.2. MLOps Level 1: Strategic Automation Phase

MLOps Level 1 or strategic phase involves implementing automated continuous training and delivery of machine learning models, using tools such as pipelines, feature stores, and metadata management to ensure data and model validation.

---

### 19.1.3. MLOps Level 2: CI/CD Automation, Transformational Phase

In the transformational phase of organizational maturity, ML experts are integrated into product teams and business units, with datasets shared across silos and automated CI/CD pipelines ensuring seamless updates through machine learning pipeline automation.

---

## 19.2. Retraining and Versioning Models

You can monitor your deployed model's performance over time using Vertex AI Model Monitoring, enabling drift detection and data collection to inform decisions on when to retrain the model with new training datasets at predetermined frequencies.

---

### 19.2.1. Triggers for Retraining

A retraining policy that balances model performance with training costs can be implemented using a schedule-based approach where the model is retrained at fixed intervals, incorporating new data on a regular basis to maintain optimal performance while minimizing delays and expenses.

---

### 19.2.2. Versioning Models

Model versioning enables deploying multiple versions of a model alongside each other, allowing end-users to select the desired version through a version ID, ensuring backward compatibility and facilitating easy comparison and management of model updates.

---

## 19.3. Feature Store

Feature engineering is a crucial aspect of building good ML models, but its time-consuming nature leads to duplication of effort, resulting in problems such as non-reusable features, data governance issues, division among teams, training and serving differences, and inability to productize these features.

---

### 19.3.1. Solution

A central feature store allows data engineers and ML engineers to share and manage features and metadata, applying software engineering principles like versioning, documentation, and access control, while offering fast processing and low latency for real-time and batch predictions.

---

### 19.3.2. Data Model

The Vertex AI Feature Store uses a time-series data model with a hierarchical structure consisting of a featurestore containing entity types, which store similar features, organized by mapping unique column headers to entities and feature values.

---

### 19.3.3. Ingestion and Serving

Vertex AI Feature Store supports both batch and streaming ingestion, allowing users to retrieve features in batch mode during model training and online mode for real-time inference.

---

## 19.4. Vertex AI Permissions Model

To manage access to resources and perform operations in Vertex AI, follow GCP's Identity and Access Management (IAM) best practices, including least privilege, service account management, auditing, and policy management.

---

### 19.4.1. Custom Service Account

When running a Vertex AI training job, it's recommended to create custom service accounts with only necessary permissions instead of using automatically created accounts with excessive permissions.

---

### 19.4.2. Access Transparency in Vertex AI

Cloud Audit logs track user activity within your project, while Access Transparency logs capture actions by Google personnel, offering two types of access for compliant logging.

---

## 19.5. Common Training and Serving Errors

Common errors during training and serving can be identified by understanding the specific types of errors and using TensorFlow as the framework for analysis and debugging.

---

### 19.5.1. Training Time Errors

During the training phase, the most relevant errors are seen when you run Model.fit() .Errors happen when the following scenarios occur: Input data is not transformed or not encoded. Tensor shape is mismatched. Out of memory errors occur because of instance size (CPU and GPU).

---

### 19.5.2. Serving Time Errors

The serving time errors are seen only during deployment and the nature of the errors is also different. The typical errors are as follows: Input data is not transformed or not encoded. Signature mismatch has occurred. Refer to this URL for a full list of TensorFlow errors: www.tensorflow.org/api_docs/python/tf/errors

---

### 19.5.3. TensorFlow Data Validation

To prevent and reduce these errors, you can use TensorFlow Data Validation (TFDV). TFDV can analyze training and serving data as follows: To compute statistics To infer schema To detect anomalies Refer here for full documentation: https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell

---

### 19.5.4. Vertex AI Debugging Shell

Vertex AI provides an interactive shell to debug training containers, allowing users to run tracing and profiling tools, analyze GPU utilization, and validate IAM permissions.

---

## 19.6. Summary

MLOps incorporates continuous integration and deployment (CI/CD) principles to automate training, deployment, and monitoring of machine learning models while balancing model quality and cost with shared feature management through the use of feature stores.

---

## 19.7. Exam Essentials

MLOps maturity ranges from experimental phase to fully mature CI/CD-inspired architecture, involving automation of model versioning, retraining triggers, and feature sharing through services like Vertex AI Feature Store or Feast.

---

## 19.8. Review Questions

Here are the concise summaries: 1. MLOps workflow: Model training, testing, and validation is not a major step. 2. MLOps level implementation for small retail organization: MLOps level 0. 3. MLOps level implementation for fashion retail store: MLOps level 1. 4. MLOps level implementation for image processing team: MLOps level 2. 5. Problems solved by MLOps level 0: Ad hoc building of models, automation of model training, and deployment. 6. MLOps level implementation for large organization with custom algorithms: No MLOps or ad hoc is needed; a higher level is required to scale. 7. Handoff in MLOps level 1: The container containing the model. 8. Handoff in MLOps level 0: The pipeline to train a model. 9. Trigger for building new models in MLOps level 2: Feature store, Random trigger,

---

Performance degradation from monitoring ML, and Metadata Store. 10. Factors to consider when setting retraining triggers: Algorithm, frequency of triggering retrains, Cost of retraining, Time to access data. 11. Policies for triggering retraining from model monitoring data: Model performance degradation below a threshold and Security breach. 12. Deployment of new versions vs models: Whenever the model has similar inputs and outputs and is used for the same purpose. 13. Good reasons to use a feature store: Many features, many engineered features not shared between teams, and differences in training/serving due to data unavailability. 14. Services that Feast does not use: BigQuery and Redis. 15. Highest level in Vertex AI Feature Store hierarchy: Entity. 16. Options for implementing a

---

feature store with structured data: Use existing services (e.g., BigQuery), download open-source Feast, or create a custom Feature Store using BigQuery, Redis, and Apache Beam. 17. False statements about Vertex AI Feature Store: It cannot ingest from Google Cloud Storage and serves features with low latency.

---
# 20. Chapter 14BigQuery ML

---

## 20.1. BigQuery – Data Access

You can access BigQuery data using the web console to execute SQL queries directly, running queries in a Jupyter Notebook via the magic command %%bigquery, or using a Python API with libraries such as Pandas.

---

## 20.2. BigQuery ML Algorithms

BigQuery ML allows users to create and deploy machine learning models using standard SQL queries without writing any Python code, leveraging a fully serverless architecture.

---

### 20.2.1. Model Training

You create a BigQuery ML model using the CREATE MODEL statement, specifying the model type (e.g. linear_reg), input label columns, and selecting data from a table with the SELECT * FROM query.

---

### 20.2.2. Model Evaluation

To evaluate the performance of a supervised machine learning model, use `ML.EVALUATE` with a separate unseen dataset as shown in SELECT * FROM ML.EVALUATE(MODEL projectid.test.creditcard_model1, ( SELECT * FROM test.creditcardtable)).

---

### 20.2.3. Prediction

You can use the ML.PREDICT function to generate a table with predicted values and probabilities by specifying a model and passing it an entire table or selecting specific columns.

---

## 20.3. Explainability in BigQuery ML

You can get global feature importance values at the model level in BigQuery using SQL functions, such as `ML.GLOBAL_EXPLAIN(MODEL 'model1')`, which returns a table containing input features and their corresponding floating-point numbers representing importance.

---

## 20.4. BigQuery ML vs. Vertex AI Tables

BigQuery and Vertex AI cater to different types of users, with BigQuery designed for SQL experts who focus on data analysis and visualization, and Vertex AI tailored for machine learning engineers familiar with Java, Python, and Jupyter Notebooks.

---

## 20.5. Interoperability with Vertex AI

Although Vertex AI and BigQuery ML are very distinct products, they have been designed to interoperate at every point in the machine learning pipeline. There are at least six integration points that make it easy to use both products together seamlessly.

---

### 20.5.1. Access BigQuery Public Dataset

BigQuery provides over 200 publicly available datasets through the Google Cloud Public Datasets Program, allowing users to access and query them without incurring storage costs, only paying for query results.

---

### 20.5.2. Import BigQuery Data into Vertex AI

You can create a Vertex AI dataset directly from a BigQuery URL, allowing seamless connection to data without exporting and importing.

---

### 20.5.3. Access BigQuery Data from Vertex AI Workbench Notebooks

You can directly access and interact with your BigQuery dataset from a Jupyter Notebook in Vertex AI Workbench, allowing for efficient data exploration and manipulation.

---

### 20.5.4. Analyze Test Prediction Data in BigQuery

You can export model predictions to BigQuery during training by providing a train and test dataset, allowing for post-hoc analysis of test predictions via SQL methods.

---

### 20.5.5. Export Vertex AI Batch Prediction Results

You can use BigQuery tables directly as input for batch predictions in Vertex AI and store the output results back in BigQuery.

---

### 20.5.6. Export BigQuery Models into Vertex AI

You can export BigQuery ML models to Vertex AI directly through the Model Registry, eliminating the need to store them in GCS, except for certain model types like ARIMA_PLUS and XGBoost models that are currently unsupported.

---

## 20.6. BigQuery Design Patterns

BigQuery ML offers novel solutions to recurring data science and machine learning problems, collectively referred to as design patterns.

---

### 20.6.1. Hashed Feature

This solution addresses the problems of incomplete vocabulary, high cardinality, and cold start issues with categorical variables by transforming them into low-cardinality domains using hashing functions like FarmHash.

---

### 20.6.2. Transforms

BigQuery ML's TRANSFORM clause allows users to apply transformations to input data before feeding it into a model, which must be applied in production, and automatically applies these transformations during prediction.

---

## 20.7. Summary

BigQuery ML provides a powerful platform for democratizing machine learning by offering an integrated SQL-based solution for building and training ML models directly within the Google Cloud platform.

---

## 20.8. Exam Essentials

BigQuery ML provides a SQL-based interface for analysts to train, predict, and explain models, while Vertex AI is designed for ML engineers with integration points to seamlessly work together.

---

## 20.9. Review Questions

Here are the concise summaries: 1. Forecast sales per country using Vertex AI AutoML Tables. 2. Predict time and distance for bicycle usage using TensorFlow model on Vertex AI tables or advanced path prediction algorithm in Google Maps. 3. Provide recommendations per user using BigQuery classification model_type, a custom collaborative filtering model with Vertex AI AutoML, matrix factorization model in BigQuery ML, or Vertex AI AutoML. 4. Run predictions quickly without setting up instances or pipelines by using BigQuery ML and TensorFlow SavedModel or Kubeflow Pipelines. 5. Create an initial machine learning model for loan classification using Kubeflow Pipelines with Vertex AI AutoML Tables, explanations, or BigQuery ML. 6. Build a new model with data in BigQuery using Vertex AI

---

pipeline components to download the dataset to GCS bucket and then run Vertex AI AutoML Tables or train BigQuery ML models. 7. Send predictions of new test data for bias and fairness testing by exporting from GCS, transferring to BigQuery, or adding to AutoML Tables test set with export functionality. 8. Use accurate models built on BigQuery ML in pipeline by running predictions in BigQuery, exporting to GCS, or retraining Vertex AI tables models with same data and hyperparameters. 9. Fix categorical feature suffering model accuracy using BigQuery's hashing function (ABS(MOD(FARM_FINGERPRINT(zipcode),buckets))) or removing the input feature. 10. Perform simple feature engineering using BigQuery TRANSFORM clause during CREATE_MODEL for hashing or bucketizing, Data Fusion for feature

---

engineering, or AutoML Tables with automatic problem-solving capabilities. Incorrect statements: - BigQuery ML models can run SQL queries through Vertex AI - BigQuery public datasets cannot be used in AutoML Tables - You cannot use SQL in BigQuery directly - Bringing TensorFlow models into BigQuery ML is a good idea - Using TRANSFORM functionality in BigQuery ML is correct Incorrect comparison statement: - BigQuery ML does not provide explanations with each prediction.

---
# 21. AppendixAnswers to Review Questions

---

## 21.1. Chapter 1: Framing ML Problems

Here are concise summaries of each point: A. Understand the problem context before finding an algorithm. B. Hyperparameters cannot be learned with supervised learning; use hyperparameter optimization instead. C. This is a video object tracking problem, where frames need to identify objects across multiple videos. D. Topic modeling clusters documents into groups using unsupervised machine learning. A. RMSE is the best metric for reducing extreme errors in regression problems. B. Accuracy is incorrect as a metric due to imbalanced datasets; precision or recall are more suitable. C. Unsupervised learning can be applied to purely unlabeled data with no labeled feedback. D. Use supervised, unsupervised, or semi-supervised learning based on the availability of labeled data. A. Supervised

---

learning is industry-standard for handling hyperparameter bias. B. Hyperparameter tuning is common in practice but not as crucial as claimed. C. Model interpretability and explanations are key to addressing biases in sensitive customer data. D. Deep learning models can handle Android devices, but deployment metrics are unknown. B. Private data includes not just images but also scans, requiring constant privacy concerns. A. Customer data is often used creatively but always raises privacy concerns at checkout. C. Changes in user behavior or product catalogs require retraining for more accurate results. D. Important product information can be used to sell more with complementary products and similar items.

---
## 21.2. Chapter 2: Exploring Data and Building Data Pipelines

The solution involves transforming the data before splitting to test and train, downsamplening majority class data using unweighting, removing features with missing values, and retraining models to minimize data skew and label leakage issues in imbalanced datasets.

---

## 21.3. Chapter 3: Feature Engineering

Here are concise summaries of each point:

A. One-hot encoding converts categorical features to numeric features.
B. Normalizing data range and convergence help models converge.
C. AUC PR minimizes false positives in imbalanced datasets.
D. Cross-validation prevents data leakage in model performance.
E. Prefetching and interleaving improve TensorFlow data processing time.
F. Cloud Data Fusion is the UI-based tool for ETL.
G. TensorFlow Transform is the most scalable data transformation method.
H. tf.Transform pipeline addresses training-serving skew in models.

---

## 21.4. Chapter 4: Choosing the Right ML Infrastructure

Here are the concise summaries: 1. Always start with a pre-trained model and use AutoML or custom models as a last resort. 2. Google Translate's "Glossary" feature can translate specific words/phrases, allowing for more accurate translations. 3. The correct approach for classification problems is using the AutoML Edge model type on edge devices. 4. To deploy models quickly, use an Android app with ML Kit instead of Coral.ai or other options. 5. Using n1-standard-2 instances may not provide enough quota for GPUs, and using 1 TPU is a better option. 6. The correct recommendation services are "Recommended for you" for home pages, "Similar items" based on product information, and "Others you may like" based on browsing history. 7. To engage customers more, use "Frequently bought together" at

---

checkout, while "Recommended for you" is suitable for home pages and "Similar items" helps with choosing between products. 8. When creating recommendations without user events, focus on project catalog information only using the model that doesn't require user data. 9. Click-through rate measures the number of clicks based on links, while revenue per order captures effectiveness at checkout. 10. Custom models can provide more accurate results but may take time, and using a Vertex AI custom job is often preferred. 11. Natural Language API does not accept voice input and should be handled with custom models or other options. 12. GPUs are the best option for TPU support in machine learning tasks. 13. Only A2 and N1 machine series support GPUs, while Option C is incorrect due to instance size

---

limitations. 14. Devices without Edge TPU support may experience significant slowdowns when deploying large models, making Edge TPU installation a better choice. 15. It's not possible to have both TPU and GPU in a single instance, and using cluster VMs for TPUs is not recommended. 16. For sparse matrices, GPUs are more efficient than TPUs, while TPUs are better suited for high-precision predictions. 17. Checking code and root cause identification (e.g., instance size) is more effective than increasing instance size or ignoring single-threaded code.

---
## 21.5. Chapter 5: Architecting ML Solutions

Here are the summaries: App Engine + Vertex AI Prediction can handle a 300ms@p99 latency requirement. Bigtable is designed for low-latency reads of large datasets. Use Cloud BigQuery, Dataflow, and Vertex AI platform for preprocessing, training, and serving in a recommendation use case. Vertex AI platform minimizes infrastructure overhead for distributed training. Cloud Storage is recommended as the document data lake solution with Document AI. Kubeflow Pipelines automates retraining workflows and provides experiment tracking features. A Cloud Storage trigger can send messages to Pub/Sub topics to trigger GKE training jobs. Use Kubeflow experiments for training and executing experiments, and batch prediction for aggregated daily data. TensorFlow's BigQueryClient efficiently reads data

---

directly from BigQuery storage using the Storage API. All three connectors (tf.data.datasetreader, TensorFlow's BigQueryClient, and others) can connect to framework datasets in BigQuery. Use Vertex AI-managed datasets to organize and manage data for training and prediction.

---
## 21.6. Chapter 6: Building Secure ML Pipelines

Federated learning is used for deploying ML models on devices with stored data, masking sensitive information with characters like # and *, requires service account key authentication, and utilizes services like Cloud DLP and Vertex AI for data security and redaction.

---

## 21.7. Chapter 7: Model Building

Here are the concise summaries: A model is performing poorly due to future changes in data distribution, requiring monitoring and retraining when necessary. The model is memorizing training data, and doubling neurons will worsen performance; a 20% dropout can help generalize without increasing training time. Out of memory error resolved by changing batch size, not image size or regularization methods. Sparse categorical cross-entropy used for multiclass classification problems. High learning rate causes oscillating loss curves, requiring adjusting hyperparameters to prevent overcorrection. Comparing validation data loss performance is the correct approach for deep learning model selection and training optimization. Modifying batch size reduces training time without impacting accuracy in

---

deep learning models. Categorical cross-entropy used with one-hot encoded data. Regularizing L2 helps resolve convergence issues indicating overfitting, while bias-variance trade-off considers both parameters during training. L1 feature selection used for clustering with k-means algorithm, not regularization for model optimization. Data augmentation techniques applied when limited data is available for machine learning models. Sigmoid activation function used for binary classification problems.

---
## 21.8. Chapter 8: Model Training and Hyperparameter Tuning

You can train TensorFlow code in the cloud with minimal manual intervention using BigQuery SQL, enabling an interactive shell, and configuring Vertex AI hyperparameter tuning to reduce computation time and costs.

---

## 21.9. Chapter 9: Model Explainability on Vertex AI

Sampled Shapley method is used to calculate feature attributions for nondifferentiable models, such as those using decoding and rounding operations.

---

## 21.10. Chapter 10: Scaling Models in Production

To achieve low-latency serving of dynamic features with minimal setup effort, consider using Bigtable for data storage, Cloud Bigtable as a datastore for writing and reading user navigation context, embedding the client on the website, deploying the gateway on App Engine, and then deploying the model using Vertex AI Prediction.

---

## 21.11. Chapter 11: Designing ML Training Pipelines

You can create performance benchmarks for models in production using TFX Evaluator or ModelValidator component and schedule pipelines on Vertex AI or Kubeflow using event-based Cloud Storage triggers.

---

## 21.12. Chapter 12: Model Monitoring, Tracking, and Auditing Metadata

Here are concise summaries for each section: 1. Data Drift vs Concept Drift: Data drift occurs when input data distribution changes, while concept drift happens when relationship between input and predicted value changes. 2. Training-Serving Skew: Requires a statistical distribution of training input features for reference to compare with production inputs. 3. Prediction Drift: Needs a baseline statistical distribution of production input features for reference to compare with changing production inputs over time. 4. Distance Metrics in Vertex AI: L-infinity distance or Chebyshev distance is the greatest distance between two vectors, while other options are incorrect. 5. Custom Model Schema: Supports string and number data types, but not a category type. 6. Input Logging: Not applicable

---

to Vertex AI, instead using container logging, access logging, and request-response logging. 7. Metadata Store Usage: Options A, B, and D are valid for metadata store usage, while option C is nonsensical. 8. Data Model Elements: Artifact refers to a specific piece of data, context provides information about the experiment, and execution tracks the status of the run.

---
## 21.13. Chapter 13: Maintaining ML Solutions

The organization is at an advanced stage of machine learning usage and should use MLOps level 2, where the data scientist experiments with the model, creates a pipeline, submits the code repository for orchestration, and triggers retraining based on performance degradation or monitoring alerts.

---

## 21.14. Chapter 14: BigQuery ML

Option D correctly uses ARIMA_PLUS to leverage COVID-19 public datasets in BigQuery.

---

# 22. Index

---

## 22.1. A

This summary focuses on machine learning and artificial intelligence best practices, including optimization techniques (AdaGrad, Adam), model explanations, fairness, interpretability, and security, as well as various AI/ML tools and technologies like ANNs, Apache Airflow, AutoML, and Vertex AI.

---

## 22.2. B

BigQuery is a fully managed cloud-based analytics platform that integrates with other Google Cloud services such as Dataproc, Pub/Sub, and Vertex AI for machine learning tasks.

---

## 22.3. C

Caching architecture leverages techniques like mapping embedding, feature hashing, and hybrid approaches to optimize data storage efficiency.

---

## 22.4. D

The provided text appears to be a technical document about machine learning models and data processing on Google Cloud Platform, summarizing concepts such as DAGs, model training, data augmentation, data transformation, data visualization, and data management.

---

## 22.5. E

BigQuery provides various encryption and analytics features including at-rest encryption with FPE, server-side encryption, tokenization, edge inference, Edge TPU, embedding, and explainability tools like Vertex Explainable AI.

---

## 22.6. F

The model achieved high accuracy with AUC-ROC of 40-41, performed well in data preprocessing and transformation using Cloud Data Fusion and TFX, and was successfully deployed to Android and iOS devices, while incorporating features like Federated Learning and Edge TPU.

---

## 22.7. G

This content lists various technical terms related to Google Cloud Platform's AI and healthcare APIs, including Generative Adversarial Networks, Google Cloud Storage, GitHub integration, and others.

---

## 22.8. H

The content discusses various strategies for hyperparameter tuning and machine learning pipeline management on Google Cloud, including Bayesian search, grid search, importance, optimization speed, parameter comparison, random search, and the integration of MLFlow with other tools such as Vertex AI.

---

## 22.9. I

IAM (identity and access management), 104 FPE (Format‐Preserving Encryption), 113 project‐level roles, 105 resource‐level roles, 105 Vertex AI and, 106 federated learning, 112 Vertex AI Workbench permissions, 106 –108 infrastructure, 86 review question answers, 302 –304 inside model data transformation, 41 integer encoding, 43 interactive shells, 175 –176

---

## 22.10. J

JupyterLab features, 154 –155

---

## 22.11. K

k‐NN (k‐nearest neighbors) algorithm, missing data, 32 Kubeflow DSL, system design, 232 –233 pipeline components, 233 Kubeflow Pipelines, 92 –93, 224 –225, 229 workflow scheduling, 230 –232 Kubernetes Engine, 87

---

## 22.12. L

The content appears to be related to machine learning model development and data preprocessing, focusing on techniques such as label encoding, latency, online prediction, data visualization, logging, and evaluation metrics like LOCF, log scaling, and loss functions.

---

## 22.13. M

Here is a concise summary of the content:

Google Cloud Machine Learning offers various machine types (75 QPS), supported by Vertex AI Workbench with BigQuery integration and managed notebooks, featuring data integration, scaling up, and scheduling code execution. It includes tools like Media Translation API, Memorystore, and ML Kit for model building and deployment, as well as TensorFlow and AutoML for custom models.

---

## 22.14. N

Naive Bayes and various machine learning models, including neural networks and deep neural networks, struggle with missing data issues like NaN errors, while also requiring normalization and data augmentation to effectively process numeric data.

---

## 22.15. O

This content appears to be related to machine learning and data processing, specifically focusing on various deployment methods (offline and online prediction), data augmentation techniques, model optimization, and orchestration tools such as Apache Airflow, Kubeflow Pipelines, and Cloud Composer.

---

## 22.16. P

Google Cloud Healthcare API uses Data Loss Prevention to protect PHI while enabling secure data transfer between devices through precomputing predictions with Edge TPU, caching architecture, and deployment on Android.

---

## 22.17. Q

quality of data, 24 –27

---

## 22.18. R

Random Forest algorithm can handle missing data and is used for regression tasks such as predicting continuous values with metrics like MAE, RMSE, and RMSLE, often deployed in Retail AI to detect concept drift and require retraining via periodic or performance-based triggers.

---

## 22.19. S

This text is a technical summary of software concepts related to machine learning, data analysis, and artificial intelligence, including topics such as SaaS, scaling, prediction, z-score, semi-supervised learning, sensitive data removal, Seq2seq+, server-side encryption, and more.

---

## 22.20. T

The provided text lists various technical terms and concepts related to TensorFlow and data science, including t-SNE, Temporal Fusion Transformer, training strategies, TFX, and APIs.

---

## 22.21. U

univariate analysis, data visualization, 20 unstructured data, model training and, 145 unsupervised learning, 6 topic modeling, 6 –7 user‐managed notebook, Vertex AI Workbench, 151 –153, 159 –161

---

## 22.22. V

Vertex AI provides various features including AutoML, Feature Store, Model Monitoring, and Workbench with features like differential privacy, explainability, and data bias/fairness analysis.

---

## 22.23. W

WIT (What‐If Tool), 177 –178 Workflow, 216 workpool tasks, distributed training, 168 –169

---

## 22.24. Z

z‐score, 26 zero correlation, 24

---

# 23. Online Test Bank

---

## 23.1. Register and Access the Online Test Bank

Register your book for online access by visiting www.wiley.com/go/sybextestprep, selecting your book from a list, completing required registration information, and entering the provided pin code to activate your account.

---

# 24. WILEY END USER LICENSE AGREEMENT

---

