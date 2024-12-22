# E-Commerce Product Categorization

## **Project Overview**
This project focuses on **E-Commerce Product Categorization**, where products are classified into predefined categories based on their textual descriptions. The dataset consists of product descriptions and corresponding categories. Multiple machine learning and deep learning models have been implemented to solve this multi-class classification problem.

---

## **Table of Contents**
1. [Dataset Loading](#dataset-loading)
2. [EDA](#project-EDA)
3. [Data Preprocessing](#data-preprocessing)
4. [Models Implemented](#models-implemented)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results and Comparison](#results-and-comparison)

---

## **Dataset Description**
The dataset contains the following features:
- **Product Description**: Textual information about the product.
- **Product Category**: Target labels representing the category of each product.

The dataset is preprocessed to clean text by removing punctuation, stopwords, and performing lowercasing. The cleaned text is then converted into numerical features for modeling.

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Cleaning text (lowercase, punctuation, and stopwords removal).
   - Feature extraction using **CountVectorizer** and **TF-IDF Vectorizer**.

2. **Model Training**:
   - Traditional ML models: Logistic Regression, Random Forest, and Multinomial Naive Bayes.
   - Deep learning model: LSTM (Long Short-Term Memory).

3. **Model Evaluation**:
   - Metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

4. **Hyperparameter Tuning**:
   - Grid Search and Random Search for optimal parameters.

---

## **Models Implemented**
### **1. Machine Learning Models**
- **Logistic Regression**:
  - Suitable for linear classification tasks.
  - Used with CountVectorizer and TF-IDF Vectorizer features.
- **Random Forest Classifier**:
  - Ensemble learning method based on decision trees.
  - Effective for non-linear data patterns.
- **Multinomial Naive Bayes**:
  - Probabilistic model well-suited for text classification.
  - Works effectively with frequency-based vectorization.

### **2. Deep Learning Model**
- **LSTM (Long Short-Term Memory)**:
  - Handles sequential data effectively.
  - Utilized embedding layers, dropout, and LSTM layers for learning.
  - Tokenized and padded sequences were used as input.

---

## **Evaluation Metrics**
- **Accuracy**: Measures overall correctness of predictions.
- **Precision**: Focuses on correctly predicted positive cases.
- **Recall**: Captures how well the model identifies true positives.
- **F1-Score**: Balances Precision and Recall.
- **Confusion Matrix**: Visualizes prediction performance for each class.

