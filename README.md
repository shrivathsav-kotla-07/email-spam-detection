# email-spam-
# Spam Email Detection System

### 📋 Overview

This project implements a **Spam Email Detection System** using **Naive Bayes Classifier**. The system is designed to classify emails as either **Spam** or **Ham (Not Spam)** based on their content. It includes data preprocessing, visualization, model training, evaluation, and real-time spam detection.

## 🚀 Features

- Data Preprocessing and Cleaning
- Spam vs Ham Distribution Visualization
- Word Cloud of Most Frequent Spam Words
- Model Training using **Multinomial Naive Bayes**
- Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Real-time Spam Detection for Custom Email Input

## 📊 Dataset

The dataset used is a collection of labeled messages with the following columns:

- **v1**: Label (ham/spam)
- **v2**: Message text

### Sample Data:

```
v1,v2
ham,"Go until jurong point, crazy.."
spam,"Free entry in 2 a wkly comp to win FA Cup"
```

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/spam-email-detection.git
cd spam-email-detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the code:**

```bash
python spam_detection.py
```

## 🧪 Libraries Used

- **NumPy**
- **Pandas**
- **Matplotlib & Seaborn** (for data visualization)
- **Scikit-learn** (for machine learning models and evaluation)
- **WordCloud** (for visual representation of spam words)

## ⚡ Usage Example

```python
# Example Usage of Spam Detection
sample_email = 'Free Tickets for IPL'
result = detect_spam(sample_email)
print(result)
```

**Output:**

```
This is a Spam Email!
```

## 📈 Model Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

The model also displays:

- **Confusion Matrix** (for both training and testing data)
- **ROC Curve** to visualize model performance

## 📊 Visualization

- **Pie Chart** for Spam vs Ham Distribution
- **Word Cloud** for Most Used Words in Spam Messages

## 📂 Project Structure

```
spam-email-detection/
├── spam_detection.py
├── spam.csv
├── requirements.txt
└── README.md
```

## 📝 Contributing

Contributions are welcome! Feel free to fork this repository, make changes, and create pull requests.

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- Dataset sourced from public repositories.
- Inspired by real-world email spam filtering systems.



