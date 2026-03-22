# IGZO/MgO Synaptic Transistor — Complete ML Project

### Neuromorphic Digit Recognition System

---

## 📁 Project Structure

```
igzo_project/
├── data/
│   ├── IDVD_IDBG_IGZO_MgO.csv              # Experimental transfer curve data
│   └── filterchar_IGZO_MgO_new.csv         # EPSC frequency response data
│
├── models/                                 # Trained models are stored here
│   ├── device_params.pkl                   # Extracted device parameters (Vth, ION/IOFF, SS)
│   ├── poly_model.pkl                      # Polynomial IDS predictor
│   ├── nn_model.pkl                        # Neural Network IDS predictor
│   ├── gain_model.pkl                      # EPSC Gain model
│   ├── igzo_digit_model.pkl                # Standard digit recognition model
│   └── igzo_TRUE_digit_model.pkl           # TRUE IGZO-based digit model (BEST)
│
├── 1_train_model.py                        # STEP 1: Train device-level models
├── 2_digit_recognition.py                  # STEP 2: Standard digit recognition
├── 2b_digit_recognition_IGZO_TRUE.py       # STEP 2b: TRUE IGZO learning (Recommended)
├── 3_app.py                                # STEP 3: GUI application
├── requirements.txt
└── README.md
```

---

## 🧠 Model Description

| File                      | IGZO-Based? | Description                                                          |
| ------------------------- | ----------- | -------------------------------------------------------------------- |
| poly_model.pkl            | Yes         | Predicts IDS from VGS using polynomial regression                    |
| nn_model.pkl              | Yes         | Predicts IDS from VGS using neural network                           |
| gain_model.pkl            | Yes         | Predicts EPSC gain from input frequency                              |
| igzo_digit_model.pkl      | Partial     | Uses IGZO-based weight quantization                                  |
| igzo_TRUE_digit_model.pkl | Yes (Full)  | Learns using actual IGZO potentiation/depression curves (Best Model) |

---

## ⚙️ Setup (One-Time)

1. Install **Python 3.10 or above**
   https://python.org/downloads
   ✔ Make sure to check **"Add to PATH"**

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Execution Flow

Follow this sequence:

### STEP 1 — Train Device Models

```
python 1_train_model.py
```

* Extracts device parameters
* Trains IDS prediction models

---

### STEP 2b — TRUE IGZO Learning (Recommended)

```
python 2b_digit_recognition_IGZO_TRUE.py
```

* Uses real IGZO potentiation/depression curves
* Weight updates follow actual device physics

---

### STEP 2 — Standard Learning (Optional)

```
python 2_digit_recognition.py
```

* Uses traditional optimization (gradient descent)
* Higher accuracy but less hardware realism

---

### STEP 3 — Launch GUI Application

```
python 3_app.py
```

* Draw digits
* Model predicts output in real time

---

## 🔬 TRUE IGZO Learning Concept

Traditional Learning:

```
W = W - learning_rate × gradient
```

IGZO-Based Learning:

```
W = igzo.potentiate(W)   # Based on experimental dG/dVGS
W = igzo.depress(W)      # Based on experimental dG/dVGS
```

* 402 conductance states derived from experimental dataset
* Weight updates directly reflect device physics

---

## ⚠️ Troubleshooting

**ModuleNotFoundError**

```
pip install -r requirements.txt
```

**FileNotFoundError (models missing)**

```
python 1_train_model.py
```

**GUI not opening**

```
pip install pillow --upgrade
```

**Slow MNIST download**

* Requires internet on first run (~11MB)

---

## 📄 Reference

Based on research paper:
**Low-Temperature Solution-Processed In₂O₃ Synaptic Transistors**
IEEE Transactions on Electron Devices, Vol. 73, No. 1, Jan 2026

---

## 🚀 Key Highlights

* Hardware-aware machine learning using IGZO/MgO synaptic transistor data
* Integration of experimental device physics into neural training
* Realistic neuromorphic system implementation
* GUI-based digit recognition demo

---
