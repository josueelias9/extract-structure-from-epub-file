---
marp: true
theme: default
paginate: true
title: "Chapter 13 Summary – Pretrained, AutoML, and Custom Models"
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

# 13.1. Pretrained vs. AutoML vs. Custom Models

---

- **Pretrained models:**  
  Already trained and deployed by Google.  
  Used via APIs (serverless, pay-per-request).  
  Ideal for developers without ML expertise.  

- **AutoML:**  
  “Just add data” approach.  
  Automates model selection and training.  
  Requires provisioning compute resources.  
  Suitable when pretrained models don’t fit your use case.  


- **Custom models:**  
  Full flexibility to design and train unique ML models.  
  Requires expert knowledge and hardware management.  
  Used for highly specific or novel problems.  

---

# 13.2. Pretrained Models

Pretrained models are large-scale models trained by Google and available through APIs or SDKs.  
They offer **quick integration**, **high performance**, and **no infrastructure setup**.

---

## 13.2.1. Vision AI

- Image processing: object & face detection, OCR, color analysis, SafeSearch.
- Serverless API for analyzing images via URL or upload.


## 13.2.2. Video AI

- Recognizes 20,000+ objects, places, and actions in video.  
- Works on **stored** or **streaming** video.  
- Used for tagging, metadata generation, and analytics.

---

## 13.2.3. Natural Language AI

- Extracts entities, analyzes sentiment and syntax, classifies text.  
- Enriches entities with external data (e.g., Wikipedia).  
- Useful for text mining, sentiment monitoring, and categorization.


## 13.2.4. Translation AI

- Detects and translates 100+ languages.  
- **Basic vs Advanced** versions: glossaries and document translation.  
- Supports text and real-time audio translation (Media Translation API).

---

## 13.2.5. Speech-to-Text

- Converts audio to text for recordings or streams.  
- Used in transcription, captioning, and multilingual subtitle generation.


## 13.2.6. Text-to-Speech

- Generates realistic human-like voices (220+ voices, 40+ languages).  
- Useful for branding, voice assistants, and accessibility.

---

# 13.3. AutoML

Automated machine learning for **well-known problem types**.  
You bring the data; Vertex AI handles algorithm selection and training.


---

## 13.3.1. AutoML for Tables or Structured Data

- Works with tabular datasets.  
- Define **budget** (max node hours) and optionally enable **early stopping**.  
- Hardware and cost depend on job type.


## 13.3.2. AutoML for Images and Video

- Simplifies image/video ML model creation.  
- **AutoML Edge models**: optimized for mobile and IoT (low-latency, low-memory).  
- Supports trade-offs between accuracy and latency.


---

## 13.3.3. AutoML for Text

- Solves text-related problems like classification and sentiment analysis.  
- Easy model building via console or SDKs.

---

## 13.3.4. Recommendations AI / Retail AI

- Provides retail-focused ML solutions:  
  - **Retail Search** (context-aware search)  
  - **Product Search** (image-based search)  
  - **Recommendations AI** (personalized product suggestions)  
- Serverless and continuously fine-tuned.

Common recommendation types:

- Others you may like → predicts next likely purchase.
- Frequently bought together → suggests cart expansion items.
- Recommended for you → personalizes homepage suggestions.
-Similar items → shows alternative products with similar attributes.
---

## 13.3.5. Document AI

- Extracts data from documents (printed or handwritten).  
- Key components:  
  - **Processors** (general, specialized, or custom)  
  - **Document AI Warehouse** for storage and metadata search.

---

## 13.3.6. Dialogflow & Contact Center AI

Provides conversational AI tools for chatbots and call centers.

| Component | Description |
|------------|-------------|
| **Virtual Agent** | Automates customer interactions. |
| **Agent Assist** | Helps human agents with real-time suggestions. |
| **Insights** | Analyzes calls and sentiment. |
| **CCAI Platform** | Multichannel, cloud-native contact center solution. |

---

# 13.4. Custom Training

Gives full control over hardware and model architecture.

---

## 13.4.1. CPU

- General-purpose, serial computation.  
- Good for basic workloads but inefficient for large-scale ML.


## 13.4.2. GPU

- Thousands of parallel ALUs for fast matrix operations.  
- Greatly accelerates deep learning training.  
- Specify GPU type and count via `machineSpec.acceleratorType` and `acceleratorCount`.

---

## 13.4.3. TPU

- Google-designed chips specialized for ML matrix operations.  
- Composed of **MXUs (Matrix Multiply Units)**.  
- Enable training speedups of **10x or more** over GPUs.


### 13.4.3.1. How to Use TPUs

- Available in **Pods** for large-scale distributed workloads.  

### 13.4.3.2. Advantages

- Massive acceleration for deep learning tasks.  
- Ideal for models with heavy matrix computations.

---

### 13.4.3.3. When to Use Each

| Hardware | Best For | Notes |
|-----------|-----------|-------|
| **CPU** | Simple workloads | Flexible but slow |
| **GPU** | General deep learning | Good parallelism |
| **TPU** | Tensor-heavy neural nets | Extreme acceleration |

### 13.4.3.4. Programming Model

- Keep the **training loop entirely on TPU** to avoid PCIe bottlenecks.

---

# 13.5. Provisioning for Predictions

Prediction = **model inference** (after training).  
Two types: **online** and **batch** predictions.


---

## 13.5.1. Scaling Behavior

- Use **autoscaling** to handle varying prediction loads.  
- Monitor CPU, memory, and GPU usage for GPU nodes.


## 13.5.2. Finding the Ideal Machine Type

- Benchmark prediction containers on Compute Engine first.  
- Consider:  
  - Model type  
  - Web server concurrency  
  - CPU/memory/GPU utilization  
  - Latency and throughput requirements  
  - Cost per query  

---

## 13.5.3. Edge TPU

- Runs inference directly on edge devices.  
- 4 TOPS on 2 watts.  
- Sold under **Coral.ai** for IoT and prototyping.


## 13.5.4. Deploy to Android or iOS

- Use **ML Kit** to deploy AutoML or custom models.  
- In-device inference = low latency + offline capability.

---

# 13.6. Summary

- Start with **Pretrained Models** for simplicity.  
- Move to **AutoML** if customization is needed.  
- Use **Custom Models** for full control and flexibility.  
- Choose hardware (CPU, GPU, TPU) based on computation type.  
- Understand training vs. prediction workloads.  
- Google Cloud supports **edge deployment** for on-device inference.
