# Estructura del Libro
## my_epub

---

## Índice de Contenidos

### Cover

### Table of Contents

#### List of Tables

#### List of Illustrations

#### Guide

#### Pages

### Official Google Cloud CertifiedProfessional Machine Learning EngineerStudy Guide

### F02

### F03

### Acknowledgments

### About the Author

### About the Technical Editors

#### About the Technical Proofreader

#### Google Technical Reviewer

### Introduction

#### Google Cloud Professional Machine Learning Engineer Certification

##### Why Become Professional ML Engineer (PMLE) Certified?

##### How to Become Certified

#### Who Should Buy This Book

#### How This Book Is Organized

##### Chapter Features

#### Bonus Digital Contents

##### Interactive Online Learning Environment and Test Bank

#### Conventions Used in This Book

#### Google Cloud Professional ML Engineer Objective Map

#### How to Contact the Publisher

#### Assessment Test

#### Answers to Assessment Test

### Chapter 1Framing ML Problems

#### Translating Business Use Cases

#### Machine Learning Approaches

##### Supervised, Unsupervised, and Semi‐supervised Learning

##### Classification, Regression, Forecasting, and Clustering

#### ML Success Metrics

##### Area Under the Curve Receiver Operating Characteristic (AUC ROC)

##### The Area Under the Precision‐Recall (AUC PR) Curve

#### Regression

#### Responsible AI Practices

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 2Exploring Data and Building Data Pipelines

#### Visualization

##### Box Plot

##### Line Plot

##### Bar Plot

##### Scatterplot

#### Statistics Fundamentals

##### Mean

##### Median

##### Mode

##### Outlier Detection

##### Standard Deviation

##### Correlation

#### Data Quality and Reliability

##### Data Skew

##### Data Cleaning

##### Scaling

##### Log Scaling

##### Z‐score

##### Clipping

##### Handling Outliers

#### Establishing Data Constraints

##### Exploration and Validation at Big‐Data Scale

#### Running TFDV on Google Cloud Platform

#### Organizing and Optimizing Training Datasets

##### Imbalanced Data

##### Data Splitting

##### Data Splitting Strategy for Online Systems

#### Handling Missing Data

#### Data Leakage

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 3Feature Engineering

#### Consistent Data Preprocessing

#### Encoding Structured Data Types

##### Why Transform Categorical Data?

##### Mapping Numeric Values

###### Normalizing

###### Bucketing

##### Mapping Categorical Values

###### Label Encoding or Integer Encoding

###### One‐Hot Encoding

###### Out of Vocab (OOV)

###### Feature Hashing

###### Hybrid of Hashing and Vocabulary

###### Embedding

##### Feature Selection

#### Class Imbalance

##### Classification Threshold with Precision and Recall

##### Area under the Curve (AUC)

###### AUC ROC

###### AUC PR

#### Feature Crosses

#### TensorFlow Transform

##### TensorFlow Data API (tf.data)

##### TensorFlow Transform

#### GCP Data and ETL Tools

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 4Choosing the Right ML Infrastructure

#### Pretrained vs. AutoML vs. Custom Models

#### Pretrained Models

##### Vision AI

##### Video AI

##### Natural Language AI

##### Translation AI

##### Speech‐to‐Text

##### Text‐to‐Speech

#### AutoML

##### AutoML for Tables or Structured Data

##### AutoML for Images and Video

##### AutoML for Text

##### Recommendations AI/Retail AI

##### Document AI

##### Dialogflow and Contact Center AI

###### Virtual Agent (Dialogflow)

###### Agent Assist

###### Insights

###### CCAI

#### Custom Training

##### How a CPU Works

##### GPU

##### TPU

###### How to Use TPUs

###### Advantages of TPUs

###### When to Use CPUs, GPUs, and TPUs

###### Cloud TPU Programming Model

#### Provisioning for Predictions

##### Scaling Behavior

##### Finding the Ideal Machine Type

##### Edge TPU

##### Deploy to Android or iOS Device

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 5Architecting ML Solutions

#### Designing Reliable, Scalable, and Highly Available ML Solutions

#### Choosing an Appropriate ML Service

#### Data Collection and Data Management

##### Google Cloud Storage (GCS)

##### BigQuery

##### Vertex AI Managed Datasets

##### Vertex AI Feature Store

##### NoSQL Data Store

#### Automation and Orchestration

##### Use Vertex AI Pipelines to Orchestrate the ML Workflow

##### Use Kubeflow Pipelines for Flexible Pipeline Construction

##### Use TensorFlow Extended SDK to Leverage Pre‐built Components for Common Steps

##### When to Use Which Pipeline

#### Serving

##### Offline or Batch Prediction

##### Online Prediction

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 6Building Secure ML Pipelines

#### Building Secure ML Systems

##### Encryption at Rest

##### Encryption in Transit

##### Encryption in Use

#### Identity and Access Management

##### IAM Permissions for Vertex AI Workbench

##### Securing a Network with Vertex AI

###### Securing Vertex AI Workbench

###### Securing Vertex AI Endpoints

###### Securing Vertex AI Training Jobs

###### Federated Learning

###### Differential Privacy

###### Format‐Preserving Encryption and Tokenization

#### Privacy Implications of Data Usage and Collection

##### Google Cloud Data Loss Prevention

##### Google Cloud Healthcare API for PHI Identification

##### Best Practices for Removing Sensitive Data

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 7Model Building

#### Choice of Framework and Model Parallelism

##### Data Parallelism

###### Synchronous Training

###### Asynchronous Training

##### Model Parallelism

#### Modeling Techniques

##### Artificial Neural Network

##### Deep Neural Network (DNN)

##### Convolutional Neural Network

##### Recurrent Neural Network

##### What Loss Function to Use

##### Gradient Descent

##### Learning Rate

##### Batch

##### Batch Size

##### Epoch

##### Hyperparameters

###### Tuning Batch Size

###### Tuning Learning Rate

#### Transfer Learning

#### Semi‐supervised Learning

##### When You Need Semi‐supervised Learning

##### Limitations of SSL

#### Data Augmentation

##### Offline Augmentation

##### Online Augmentation

#### Model Generalization and Strategies to Handle Overfitting and Underfitting

##### Bias Variance Trade‐Off

##### Underfitting

##### Overfitting

##### Regularization

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 8Model Training and Hyperparameter Tuning

#### Ingestion of Various File Types into Training

##### Collect

##### Process

###### Cloud Dataflow

###### Cloud Data Fusion

###### Cloud Dataproc

###### Cloud Composer

###### Cloud Dataprep

###### Summary of Processing Tools

##### Store and Analyze

#### Developing Models in Vertex AI Workbench by Using Common Frameworks

##### Creating a Managed Notebook

##### Exploring Managed JupyterLab Features

##### Data Integration

##### BigQuery Integration

##### Ability to Scale the Compute Up or Down

##### Git Integration for Team Collaboration

##### Schedule or Execute a Notebook Code

##### Creating a User‐Managed Notebook

#### Training a Model as a Job in Different Environments

##### Training Workflow with Vertex AI

##### Training Dataset Options in Vertex AI

##### Pre‐built Containers

##### Custom Containers

##### Distributed Training

#### Hyperparameter Tuning

##### Why Hyperparameters Are Important

##### Techniques to Speed Up Hyperparameter Optimization

##### How Vertex AI Hyperparameter Tuning Works

##### Vertex AI Vizier

###### How Vertex AI Vizier Differs from Custom Training

#### Tracking Metrics During Training

##### Interactive Shell

##### TensorFlow Profiler

##### What‐If Tool

#### Retraining/Redeployment Evaluation

##### Data Drift

##### Concept Drift

##### When Should a Model Be Retrained?

#### Unit Testing for Model Training and Serving

##### Testing for Updates in API Calls

##### Testing for Algorithmic Correctness

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 9Model Explainability on Vertex AI

#### Model Explainability on Vertex AI

##### Explainable AI

##### Interpretability and Explainability

##### Feature Importance

##### Vertex Explainable AI

###### Feature Attribution

###### Vertex AI Example–Based Explanations

##### Data Bias and Fairness

##### ML Solution Readiness

##### How to Set Up Explanations in the Vertex AI

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 10Scaling Models in Production

#### Scaling Prediction Service

##### TensorFlow Serving

###### Serving a Saved Model with TensorFlow Serving

#### Serving (Online, Batch, and Caching)

##### Real‐Time Static and Dynamic Reference Features

##### Pre‐computing and Caching Prediction

#### Google Cloud Serving Options

##### Online Predictions

###### Deploying the Model

###### Deploying a Model Using an API

###### Deploying a Model Using the Google Cloud Console

###### Make Predictions

###### A/B Testing of Different Versions of a Model

###### Undeploy Endpoints

###### Send an Online Explanation Request

##### Batch Predictions

#### Hosting Third‐Party Pipelines (MLflow) on Google Cloud

#### Testing for Target Performance

#### Configuring Triggers and Pipeline Schedules

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 11Designing ML Training Pipelines

#### Orchestration Frameworks

##### Kubeflow Pipelines

##### Vertex AI Pipelines

##### Apache Airflow

##### Cloud Composer

##### Comparison of Tools

#### Identification of Components, Parameters, Triggers, and Compute Needs

##### Schedule the Workflows with Kubeflow Pipelines

##### Schedule Vertex AI Pipelines

#### System Design with Kubeflow/TFX

##### System Design with Kubeflow DSL

###### Kubeflow Pipelines Components

##### System Design with TFX

#### Hybrid or Multicloud Strategies

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 12Model Monitoring, Tracking, and Auditing Metadata

#### Model Monitoring

##### Concept Drift

##### Data Drift

#### Model Monitoring on Vertex AI

##### Drift and Skew Calculation

###### Practical Considerations of Enabling Monitoring

##### Input Schemas

###### Automatic Schema Parsing

###### Custom Schema

#### Logging Strategy

##### Types of Prediction Logs

###### Container Logging

###### Access Logging

###### Request‐Response Logging

##### Log Settings

##### Model Monitoring and Logging

#### Model and Dataset Lineage

##### Vertex ML Metadata

###### Manage ML Metadataschemas

###### Vertex AI Pipelines

#### Vertex AI Experiments

#### Vertex AI Debugging

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 13Maintaining ML Solutions

#### MLOps Maturity

##### MLOps Level 0: Manual/Tactical Phase

###### Key Features of Level 0

###### Challenges

##### MLOps Level 1: Strategic Automation Phase

###### Key Features of MLOps Level 1

###### Challenges

##### MLOps Level 2: CI/CD Automation, Transformational Phase

###### Key Features of Level 2

#### Retraining and Versioning Models

##### Triggers for Retraining

##### Versioning Models

#### Feature Store

##### Solution

##### Data Model

##### Ingestion and Serving

#### Vertex AI Permissions Model

##### Custom Service Account

##### Access Transparency in Vertex AI

#### Common Training and Serving Errors

##### Training Time Errors

##### Serving Time Errors

##### TensorFlow Data Validation

##### Vertex AI Debugging Shell

#### Summary

#### Exam Essentials

#### Review Questions

### Chapter 14BigQuery ML

#### BigQuery – Data Access

#### BigQuery ML Algorithms

##### Model Training

##### Model Evaluation

##### Prediction

#### Explainability in BigQuery ML

#### BigQuery ML vs. Vertex AI Tables

#### Interoperability with Vertex AI

##### Access BigQuery Public Dataset

##### Import BigQuery Data into Vertex AI

##### Access BigQuery Data from Vertex AI Workbench Notebooks

##### Analyze Test Prediction Data in BigQuery

##### Export Vertex AI Batch Prediction Results

##### Export BigQuery Models into Vertex AI

#### BigQuery Design Patterns

##### Hashed Feature

##### Transforms

#### Summary

#### Exam Essentials

#### Review Questions

### AppendixAnswers to Review Questions

#### Chapter 1: Framing ML Problems

#### Chapter 2: Exploring Data and Building Data Pipelines

#### Chapter 3: Feature Engineering

#### Chapter 4: Choosing the Right ML Infrastructure

#### Chapter 5: Architecting ML Solutions

#### Chapter 6: Building Secure ML Pipelines

#### Chapter 7: Model Building

#### Chapter 8: Model Training and Hyperparameter Tuning

#### Chapter 9: Model Explainability on Vertex AI

#### Chapter 10: Scaling Models in Production

#### Chapter 11: Designing ML Training Pipelines

#### Chapter 12: Model Monitoring, Tracking, and Auditing Metadata

#### Chapter 13: Maintaining ML Solutions

#### Chapter 14: BigQuery ML

### Index

#### A

#### B

#### C

#### D

#### E

#### F

#### G

#### H

#### I

#### J

#### K

#### L

#### M

#### N

#### O

#### P

#### Q

#### R

#### S

#### T

#### U

#### V

#### W

#### Z

### Online Test Bank

#### Register and Access the Online Test Bank

### WILEY END USER LICENSE AGREEMENT

---

## Resumen

- **Total de capítulos encontrados:** 433
- **Archivo EPUB:** `my_epub.epub`
