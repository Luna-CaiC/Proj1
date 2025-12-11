# Sentiment Analysis on Movie Reviews

## Project Overview
This project focuses on performing binary sentiment analysis on the IMDB Movie Reviews dataset. The goal is to classify movie reviews as either **positive** or **negative** using various machine learning and deep learning techniques. The project covers the entire pipeline from data exploration and preprocessing to model training and evaluation.

## Dataset
- **Source:** IMDB Dataset
- **Size:** 50,000 movie reviews
- **Class Balance:** Balanced (25,000 positive, 25,000 negative)
- **Features:** Raw textual reviews
- **Target:** Sentiment (Binary: 0 for Negative, 1 for Positive)

## Technologies Used
- **Language:** Python 3.x
- **Libraries:**
  - **Data Manipulation:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn, WordCloud
  - **Natural Language Processing:** NLTK (Tokenization, Stopwords, Stemming)
  - **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression, SVM, KNN, MLP)
  - **Deep Learning:** TensorFlow/Keras (CNN, Embedding, Sequence Padding)

## Methodology

### 1. Data Preprocessing
- **Text Cleaning:** Removal of HTML tags, punctuation, special characters, and digits.
- **Normalization:** Conversion to lowercase.
- **NLP Techniques:** Tokenization, stopword removal, and Porter Stemming.
- **Vectorization:**
  - **TF-IDF:** Used for traditional ML models (max 5000 features).
  - **Sequence Padding:** Used for the CNN model (max length 200).
- **Data Split:** 75% Training, 25% Testing.

### 2. Models Implemented
The project implements and compares the following classification models:
- **Logistic Regression**: A baseline linear model suitable for sparse high-dimensional text data.
- **Linear SVC (Support Vector Classifier)**: Optimized for high-dimensional spaces.
- **K-Nearest Neighbors (KNN)**: A non-parametric method based on feature similarity.
- **Multi-Layer Perceptron (MLP)**: Feedforward neural networks with various hidden layer architectures (1, 2, and 3 layers).
- **Convolutional Neural Network (CNN)**: A deep learning model utilizing 1D convolutions to capture local patterns in text sequences.
  - Architecture: Embedding (128 dim) -> Conv1D (128 filters) -> GlobalMaxPooling -> Dense (64) -> Output (Sigmoid).

## Results
The models were evaluated based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

- **Best Performers:** Logistic Regression, Linear SVC, and CNN achieved the highest accuracies, reaching approximately **88%**.
- **CNN Performance:** The CNN model demonstrated robust performance, effectively learning semantic features from the text sequences.
- **KNN Performance:** Lower accuracy (~77%) compared to linear models and neural networks, highlighting the effectiveness of dimensionality-aware models for text.

## How to Run
1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow wordcloud
   ```
2. Download the `IMDB Dataset.csv` and place it in the project root.
3. Run the Jupyter Notebook `SentimentAnalysis_MovieReview.ipynb` to execute the analysis and training pipeline.

## Conclusion
This project demonstrates that both traditional linear models (like SVM and Logistic Regression) and deep learning models (CNN) are highly effective for sentiment analysis on this dataset, with CNNs offering the potential for further scalability on larger, more complex corpora.
