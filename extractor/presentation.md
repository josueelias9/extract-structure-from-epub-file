---
marp: true
theme: default
paginate: true
style: |
  section {
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/5/51/Google_Cloud_logo.svg');
    background-size: 250px;
    background-position: 95% 90%; /* esquina inferior derecha */
    background-repeat: no-repeat;
    opacity: 1;
  }
---

# 13 Chapter 4: Choosing the Right ML Infrastructure

---

## 13.1 Pretrained vs. AutoML vs. Custom Models

Choosing the right machine learning infrastructure depends on control, scalability, and data availability.

---

## 13.2 Pretrained Models

| Service | Description |
|----------|-------------|
| **13.2.1 Vision AI** | Detects objects, landmarks, and faces in images. |
| **13.2.2 Video AI** | Analyzes videos to recognize activities, scenes, and objects. |
| **13.2.3 Natural Language AI** | Performs entity recognition, sentiment, and content classification. |
| **13.2.4 Translation AI** | Translates text across multiple languages using neural translation. |
| **13.2.5 Speech-to-Text** | Converts human speech into written text. |
| **13.2.6 Text-to-Speech** | Generates realistic speech from text input. |

---

## 13.3 AutoML

AutoML automates the model training process for common ML tasks, requiring only user data and minimal configuration to build models through a web console or SDK.

### 13.3.1 AutoML for Tables or Structured Data
Automatically trains models for tabular datasets.

### 13.3.2 AutoML for Images and Video
Builds image and video classifiers without manual feature engineering.

---

### 13.3.3 AutoML for Text
Creates NLP models for text classification and sentiment analysis.


### 13.3.4 Recommendations AI / Retail AI
Provides personalized product recommendations for e-commerce.

### 13.3.5 Document AI
Extracts structured data from scanned or digital documents.

---

### 13.3.6 Dialogflow and Contact Center AI

| Component | Description |
|------------|-------------|
| **13.3.6.1 Virtual Agent (Dialogflow)** | Builds conversational interfaces using natural language. |
| **13.3.6.2 Agent Assist** | Suggests real-time responses to customer service agents. |
| **13.3.6.3 Insights** | Analyzes conversations to find common issues and trends. |
| **13.3.6.4 CCAI** | Integrates all Dialogflow services into a complete contact center solution. |

---

## 13.4 Custom Training

Custom training gives full control over model design, tuning, and optimization — ideal for specific data or advanced architectures.

### 13.4.1 How a CPU Works
Executes instructions sequentially; good for inference and lightweight ML tasks.

### 13.4.2 GPU
Performs parallel operations, speeding up deep learning training.

---

### 13.4.3 TPU
Google’s hardware accelerator optimized for large-scale tensor operations.

| Topic | Description |
|--------|-------------|
| **13.4.3.1 How to Use TPUs** | Accessible through Google Cloud or TensorFlow APIs for distributed training. |
| **13.4.3.2 Advantages of TPUs** | High throughput, energy efficiency, and reduced training time. |
| **13.4.3.3 When to Use CPUs, GPUs, and TPUs** | CPUs for light tasks, GPUs for flexibility, TPUs for heavy workloads. |
| **13.4.3.4 Cloud TPU Programming Model** | Scalable training across multiple TPU cores using TensorFlow or JAX. |

---

## 13.5 Provisioning for Predictions
Prediction workloads differ from training because they must handle continuous demand and scale dynamically, especially for online predictions. There are two main types—online (real-time responses) and batch (cost-efficient large-scale jobs)—and the key factors are scaling behavior and choosing the right machine type.

---

| Topic | Description |
|--------|-------------|
| **13.5.1 Scaling Behavior** | Ensures prediction services scale based on demand. |
| **13.5.2 Finding the Ideal Machine Type** | Balances cost, latency, and throughput for production workloads. |
| **13.5.3 Edge TPU** | Specialized hardware for on-device inference at the network edge. |
| **13.5.4 Deploy to Android or iOS Device** | Enables real-time mobile predictions with optimized performance. |

---

## 13.6 Summary

- **Pretrained Models (13.2)** → Quick deployment, no training required.  
- **AutoML (13.3)** → Automated model building with minimal coding.  
- **Custom Training (13.4)** → Maximum flexibility for specialized data.  

Choosing the right option depends on the problem scale, data ownership, and required customization.

---

## 13.7 Exam Essentials

- Recognize when to use Pretrained, AutoML, or Custom models.  
- Understand CPU, GPU, and TPU differences.  
- Know the purpose of Edge TPUs and prediction provisioning.  
- Be familiar with Dialogflow and Contact Center AI components.

---

## 13.8 Review Questions

1. When should you choose AutoML over Pretrained Models?  
2. What are the main advantages of TPUs?  
3. What services are part of Contact Center AI?  
4. How do you optimize resources for prediction services?
