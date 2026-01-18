# IBM AI Engineering Professional Certificate Projects

![IBM AI Engineering](https://img.shields.io/badge/IBM-AI%20Engineering-blue?style=for-the-badge&logo=ibm) ![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow) ![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

This repository contains the hands-on projects and capstone assignments completed as part of the **IBM AI Engineering Professional Certificate**. Each project is contained within a Jupyter Notebook, demonstrating key concepts in Machine Learning, Deep Learning, and Neural Networks.

> **Note:** All notebooks were developed and executed in **Google Colab**

## 📂 Project Index

| Project Name | Notebook File | Key Concepts |
| :--- | :--- | :--- |
| **Waste Classification** | [`WasteProductsClassificationUsingTransferLearning.ipynb`](./WasteProductsClassificationUsingTransferLearning.ipynb) | Transfer Learning (VGG16), Computer Vision, Image Augmentation |
| **Aircraft Damage Detection** | [`ClassificationandCaptioningAircraftDamageUsingPretrainedModels.ipynb`](./ClassificationandCaptioningAircraftDamageUsingPretrainedModels.ipynb) | Multi-label Classification, VGG16, Image Captioning (BLIP), PyTorch & TensorFlow |
| **Fruit Classification** | [`FruitClassificationUsingTransferLearning.ipynb`](./FruitClassificationUsingTransferLearning.ipynb) | Deep Learning, VGG16, Fine-tuning, MobileNetV2, Data Augmentation |
| **Rainfall Prediction** | [`RainfallPredictionClassifier.ipynb`](./RainfallPredictionClassifier.ipynb) | Binary Classification, Random Forest, Logistic Regression, Scikit-Learn Pipelines |
| **Titanic Survival** | [`TitanicSurvivalPrediction.ipynb`](./TitanicSurvivalPrediction.ipynb) | Data Cleaning, Feature Engineering, KNN, Random Forest, Grid Search |
| **Breast Cancer Classification** | [`BreastCancerClassification.ipynb`](./BreastCancerClassification.ipynb) | Deep Learning (PyTorch), Binary Classification, Data Balancing, UCI Dataset |
| **Iris Flower Classification** | [`IrisClassification.ipynb`](./IrisClassification.ipynb) | Deep Learning (PyTorch), Multi-class Classification, Scikit-Learn Pipelines |
| **LoL Match Predictor** | [`LeagueofLegendsMatchPredictor.ipynb`](./LeagueofLegendsMatchPredictor.ipynb) | PyTorch, Logistic Regression, L2 Regularization, ROC & Confusion Matrix |
| **Anime Face Classification** | [`AnimeFaceClassification.ipynb`](./AnimeFaceClassification.ipynb) | Deep Learning (PyTorch), Computer Vision, Custom Dataset Loading (KaggleHub) |
| **Fashion Mnist Classification** | [`FashionMnistClassification.ipynb`](./FashionMnistClassification.ipynb) | Deep Learning (PyTorch), CNN, Batch Normalization, SGD Optimizer |


## 🚀 Featured Project Highlights

### 1. Waste Classification using Transfer Learning
* **Objective:** Automate waste sorting by classifying images into 'Organic' or 'Recyclable'.
* **Methodology:** Leveraged a pre-trained **VGG16** model (Transfer Learning) to extract features, followed by fine-tuning dense layers for specific waste classification.
* **Tech Stack:** TensorFlow, Keras, Matplotlib.
* **Outcome:** Achieved high accuracy in distinguishing waste types, suitable for deployment in smart recycling bins.

### 2. Aircraft Damage Classification & Captioning
* **Objective:** Detect specific types of aircraft damage (dents, cracks) and generate descriptive captions.
* **Methodology:** Used **VGG16** for multi-label classification to identify defects. Explored **BLIP** (Bootstrapping Language-Image Pre-training) from Hugging Face for generating natural language descriptions of the images.
* **Tech Stack:** PyTorch, Transformers, TensorFlow, Roboflow API.
* **Outcome:** A multi-modal solution capable of both categorizing damage severity and describing it textually.

### 3. Fruit Classification (121 Classes)
* **Objective:** Classify images of fruits into 121 distinct categories.
* **Methodology:** implemented a **VGG16** base model with custom top layers. Utilized `ImageDataGenerator` for robust data augmentation and applied **fine-tuning** by unfreezing the top 5 layers of the base model to improve feature extraction specific to fruits.
* **Tech Stack:** TensorFlow, Keras, Scikit-learn.
* **Outcome:** High-precision multi-class classification model capable of distinguishing between subtle fruit varieties.

## 🛠️ Setup & Usage

To run these notebooks locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. You can install the common requirements:
    ```bash
    pip install tensorflow numpy pandas matplotlib scikit-learn seaborn torch transformers
    ```

3.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```

## 📜 Certification Details
* **Course:** [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer)
* **Provider:** Coursera & IBM
* **Completion Date:** 2026 ~ *Predicted*

---
*Disclaimer: These projects are part of the educational coursework provided by IBM and are for demonstration purposes.*