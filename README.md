# 🧠 FailureNet — Failure-Aware Deep Learning System

# img 

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-00c97a?style=for-the-badge"/>
</p>

<p align="center">
  <strong>A deep learning system that predicts when its own predictions might fail.</strong>
</p>

<p align="center">
  Traditional ML models only output a prediction.<br/>
  FailureNet goes one step further — it estimates uncertainty and <em>rejects</em> unreliable predictions.
</p>

---

## 🚀 What Makes This Different

Most ML projects stop here:

```
Train model → Evaluate accuracy → Done
```

FailureNet focuses on **AI reliability**. It demonstrates:

- 📊 Uncertainty estimation
- 🔮 Failure prediction
- ✅ Selective prediction (Accept / Reject)
- ⚖️ Risk-aware decision making

These topics are actively researched in **AI safety** and **trustworthy machine learning**.

---

## ❗ The Problem

Most ML models always output a prediction — even when they're wrong.

```
Image   →  Model predicts "Cat"
Confidence  →  0.95
Reality     →  The image was actually a dog
```

The model is **confident but incorrect**.

This becomes dangerous in real-world systems such as:

| Domain | Risk |
|---|---|
| 🏥 Medical Diagnosis | Wrong diagnosis with high confidence |
| 🚗 Autonomous Driving | Misclassifying obstacles at speed |
| 💳 Financial Fraud Detection | Approving fraudulent transactions |
| 🏭 Industrial Automation | Robots acting on incorrect data |

We want the model to say **"I'm not sure"** instead of making a wrong decision.

---

## ✅ The Solution

FailureNet outputs not just a class, but a full reliability report:

**When confident:**
```
Prediction:         Cat
Confidence:         0.91
Failure Probability: 0.08
Decision:           ✅ ACCEPT
```

**When uncertain:**
```
Prediction:          Dog
Confidence:          0.54
Failure Probability:  0.72
Decision:            ❌ REJECT
```

---

## 🔑 Key Concepts

### Confidence
The highest probability assigned to any single class.

```
[cat: 0.82, dog: 0.12, bird: 0.06]  →  confidence = 0.82
```

> ⚠️ Neural networks are often **overconfident**, so confidence alone is not reliable.

---

### Entropy
Measures how spread out prediction probabilities are.

| Entropy | Meaning |
|---|---|
| Low | Model is confident → safe to trust |
| High | Model is confused → risk of failure |

```
[cat: 0.90, dog: 0.05, bird: 0.05]  →  Entropy: LOW  ✅
[cat: 0.34, dog: 0.33, bird: 0.33]  →  Entropy: HIGH ❌
```

---

### Monte Carlo Dropout
Instead of running the model once, we run it **multiple times** with dropout active.

```
Run 1 → Cat ✅
Run 2 → Cat ✅
Run 3 → Dog ❌  ← disagreement detected
Run 4 → Cat ✅
Run 5 → Cat ✅
```

High variance across runs = **uncertain model**.

This gives us additional signals: prediction variance, entropy, and confidence stability.

---

## 🔄 Full Pipeline

```
Input Image
     │
     ▼
CNN Classifier
     │
     ▼
Monte Carlo Dropout  (20 forward passes)
     │
     ▼
Uncertainty Signals
  ├── Confidence
  ├── Entropy
  ├── Variance
  └── Margin
     │
     ▼
Failure Prediction Head  (secondary neural network)
     │
     ▼
Failure Probability
     │
     ▼
 ACCEPT  or  REJECT
```

---

## 🤖 Failure Prediction Model

A **second neural network** is trained specifically to predict whether the primary classifier will fail.

**Input features:**
```
confidence  |  entropy  |  variance  |  margin
```

**Output:**
```
failure_probability = 0.78
→ 78% chance this prediction is wrong
```

---

## 📂 Project Structure

```
FailureNet/
│
├── Data/                    # Dataset files
├── Fail/                    # Virtual environment
│
├── model/
│   ├── best_model.pth       # Trained CNN classifier
│   └── failure_head.pth     # Trained failure predictor
│
├── Notebook/                # Research & experiments
│
├── app.py                   # Streamlit UI
├── failurenet_pipeline.py   # Inference pipeline
├── models_def.py            # Model architectures
│
└── README.md
```

---

## ⚙️ How to Run

**1. Activate the virtual environment**
```bash
Fail\Scripts\activate
```

**2. Install dependencies**
```bash
pip install torch torchvision streamlit pillow plotly
```

**3. Launch the app**
```bash
streamlit run app.py
```

**4. Upload any image to get:**
- Predicted class
- Confidence score
- Entropy
- Failure probability
- Final ACCEPT / REJECT decision

---

## ⚠️ Known Limitation

The classifier was trained on **CIFAR-10** (32×32 images).

Real-world images contain far more detail than 32×32 pixels, so resizing them to this resolution loses significant information. As a result, failure detection may not perfectly reject all out-of-distribution images.

> The goal of this project is to **demonstrate failure-aware AI concepts** — not to maximise classification accuracy.

---

## 🌍 Real-World Applications

| Domain | How FailureNet Helps |
|---|---|
| 🏥 Healthcare | Flags uncertain diagnoses for human review |
| 🚗 Autonomous Vehicles | Triggers safe-stop when object detection is uncertain |
| 💳 Financial Systems | Escalates uncertain fraud cases to analysts |
| 🏭 Industrial Automation | Halts operations when sensor readings are unreliable |

---

## 🔭 Future Improvements

- [ ] Train with larger, higher-resolution datasets
- [ ] Add out-of-distribution (OOD) detection
- [ ] Improve failure prediction calibration
- [ ] Integrate real-time video stream support
- [ ] Export reliability reports as PDF

---

## 🛠️ Built With

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

---

<p align="center">
  Made with ❤️ by <strong>Dhruv Devaliya</strong> · <a href="https://github.com/Bit-Bard">Bit-Bard</a>
</p>
