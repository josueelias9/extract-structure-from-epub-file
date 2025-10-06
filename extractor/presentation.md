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
    opacity: 1; /* el logo es translúcido, así que puedes mantenerlo visible */
  }

---

# Chapter 4: Choosing the Right ML Infrastructure

---

## Pretrained vs. AutoML vs. Custom Models

Choosing the right machine learning infrastructure depends on control, scalability, and data availability.

---

## Pretrained Models

| Service | Description |
|----------|-------------|
| **Vision AI** | Detects objects, landmarks, and faces in images. |
| **Video AI** | Analyzes videos to recognize activities, scenes, and objects. |
| **Natural Language AI** | Performs entity recognition, sentiment, and content classification. |
| **Translation AI** | Translates text across multiple languages using neural translation. |
| **Speech-to-Text** | Converts human speech into written text. |
| **Text-to-Speech** | Generates realistic speech from text input. |

---

## AutoML

### AutoML for Tables or Structured Data
Automatically trains models for tabular datasets.

### AutoML for Images and Video
Builds image and video classifiers without manual feature engineering.

### AutoML for Text
Creates NLP models for text classification and sentiment analysis.

---

### Recommendations AI / Retail AI
Provides personalized product recommendations for e-commerce.

### Document AI
Extracts structured data from scanned or digital documents.


---

### Dialogflow and Contact Center AI

| Component | Description |
|------------|-------------|
| **Virtual Agent (Dialogflow)** | Builds conversational interfaces using natural language. |
| **Agent Assist** | Suggests real-time responses to customer service agents. |
| **Insights** | Analyzes conversations to find common issues and trends. |
| **CCAI** | Integrates all Dialogflow services into a complete contact center solution. |

---

## Custom Training

Custom training gives full control over model design, tuning, and optimization — ideal for specific data or advanced architectures.


### How a CPU Works
Executes instructions sequentially; good for inference and lightweight ML tasks.

### GPU
Performs parallel operations, speeding up deep learning training.


---

### TPU
Google’s hardware accelerator optimized for large-scale tensor operations.



| Topic | Description |
|--------|-------------|
| **How to Use TPUs** | Accessible through Google Cloud or TensorFlow APIs for distributed training. |
| **Advantages of TPUs** | High throughput, energy efficiency, and reduced training time. |
| **When to Use CPUs, GPUs, and TPUs** | CPUs for light tasks, GPUs for flexibility, TPUs for heavy workloads. |
| **Cloud TPU Programming Model** | Allows scalable training across multiple TPU cores using frameworks like TensorFlow or JAX. |

---

## Provisioning for Predictions

| Topic | Description |
|--------|-------------|
| **Scaling Behavior** | Ensures that prediction services scale based on demand. |
| **Finding the Ideal Machine Type** | Balances cost, latency, and throughput for production workloads. |
| **Edge TPU** | Specialized hardware for on-device inference at the network edge. |
| **Deploy to Android or iOS Device** | Enables real-time mobile predictions with optimized performance. |

---

## Summary

- **Pretrained Models** → Quick deployment, no training required.  
- **AutoML** → Automated model building with minimal coding.  
- **Custom Training** → Maximum flexibility, suitable for specialized data.  

Choosing the right option depends on the problem scale, data ownership, and required customization.

---

## Exam Essentials

- Recognize when to use Pretrained, AutoML, or Custom models.  
- Understand CPU, GPU, and TPU differences.  
- Know the purpose of Edge TPUs and prediction provisioning.  
- Be familiar with Dialogflow and Contact Center AI components.

---

## Review Questions

1. When should you choose AutoML over Pretrained Models?  
2. What are the main advantages of TPUs?  
3. What services are part of Contact Center AI?  
4. How do you optimize resources for prediction services?

