# 🎓 Educational Sentiment Analysis with Transformer Models
A comprehensive sentiment analysis pipeline for educational conversations using state-of-the-art transformer models. This project includes extensive exploratory data analysis (EDA), advanced text preprocessing, model fine-tuning, and detailed evaluation with visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Visualizations](#visualizations)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project performs sentiment analysis on educational conversations, classifying text into three categories:
- **Positive** 😊
- **Neutral** 😐
- **Negative** 😞

The pipeline leverages **DistilBERT**, a distilled version of BERT with ~66M parameters, providing an excellent balance between performance and efficiency.

## ✨ Features

### 🔍 Comprehensive EDA
- Text length and word count statistics
- Sentiment distribution analysis
- WordCloud generation per sentiment
- Top words and bigram analysis
- Word frequency heatmaps

### 🧹 Advanced Text Preprocessing
- Contraction expansion ("can't" → "cannot")
- URL, email, and HTML tag removal
- Accent normalization
- Repeated character reduction
- Intelligent stopword removal (preserves negations)
- Lemmatization and stemming options
- Sentiment word normalization
- Rare word filtering

### 🤖 Model Training
- Fine-tuning pre-trained transformer models
- Configurable hyperparameters
- Mixed precision training (FP16)
- Automatic best model selection
- Stratified train/validation/test split

### 📊 Evaluation & Visualization
- Confusion matrices (raw and normalized)
- Per-class metrics (Precision, Recall, F1-Score)
- Multi-class ROC curves with AUC scores
- Classification reports
- Class distribution analysis
- Performance summary dashboards

## 📊 Dataset

**Source**: Educational Conversations with Sentiment Dataset

**Format**: CSV file with columns:
- `text`: Educational conversation text
- `sentiment`: Sentiment label (Positive/Neutral/Negative)

**Sample Statistics**:
- Total samples: [Varies by dataset]
- Average text length: ~[X] characters
- Average word count: ~[Y] words

### Data Split
```
Training:   68% (stratified)
Validation: 12% (stratified)
Test:       20% (stratified)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Clone Repository
```bash
git clone https://github.com/samratabduljalil/FineTune4Sentiment-End-to-End-Sentiment-Analysis-with-Transformers.git
cd FineTune4Sentiment-End-to-End-Sentiment-Analysis-with-Transformers
```

### Install Dependencies
```bash
pip install -r requirements.txt --upgrade
```


### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```


## 💻 Usage

### 1. Exploratory Data Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/Educational_Conversations_with_Sentiment.csv')

# Analyze sentiment distribution
print(df['sentiment'].value_counts())

# Visualize
sns.countplot(x='sentiment', data=df)
plt.show()
```

### 2. Text Preprocessing
```python
from src.preprocessing import AdvancedTextPreprocessor

# Initialize preprocessor
preprocessor = AdvancedTextPreprocessor(
    remove_stopwords=True,
    apply_lemmatization=True,
    expand_contractions=True
)

# Preprocess text
df['preprocessed_text'] = df['text'].apply(preprocessor.preprocess)
```

### 3. Model Training
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3
)

# Train model (see notebook for complete training setup)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()
```

### 4. Model Inference
```python
from transformers import pipeline

# Load trained model
classifier = pipeline(
    "sentiment-analysis",
    model="./Educational_Conversations_sentiment_model",
    tokenizer="./Educational_Conversations_sentiment_model"
)

# Predict sentiment
text = "The lecture was incredibly helpful and engaging!"
result = classifier(text)
print(result)
# Output: [{'label': 'Positive', 'score': 0.98}]
```

## 🔬 Methodology

### 1. Data Exploration
- Statistical analysis of text features
- Sentiment distribution visualization
- Word frequency analysis (unigrams and bigrams)
- WordCloud generation
- Cross-sentiment word comparison

### 2. Text Preprocessing Pipeline
```
Raw Text
   ↓
Contraction Expansion
   ↓
Lowercasing & Cleaning
   ↓
URL/Email/HTML Removal
   ↓
Tokenization
   ↓
Stopword Removal (Smart)
   ↓
Lemmatization
   ↓
Normalization
   ↓
Cleaned Text
```

### 3. Model Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Parameters**: ~66 million
- **Classification Head**: 3-class output layer
- **Optimization**: AdamW optimizer
- **Loss Function**: Cross-entropy loss

### 4. Training Strategy
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Warmup Steps**: 30
- **Weight Decay**: 0.01
- **Mixed Precision**: FP16 (if GPU available)

### 5. Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- AUC-ROC (per class and macro)
- Confusion Matrix

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | XX.XX% |
| **Macro F1-Score** | XX.XX% |
| **Weighted F1-Score** | XX.XX% |
| **Macro AUC-ROC** | XX.XX |

### Per-Class Performance

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **Negative** | XX.XX% | XX.XX% | XX.XX% | XXX |
| **Neutral** | XX.XX% | XX.XX% | XX.XX% | XXX |
| **Positive** | XX.XX% | XX.XX% | XX.XX% | XXX |

### Confusion Matrix
```
                Predicted
              Neg  Neu  Pos
Actual  Neg  [XX] [XX] [XX]
        Neu  [XX] [XX] [XX]
        Pos  [XX] [XX] [XX]
```

## 📊 Visualizations

### Sentiment Distribution
![Sentiment Distribution](results/sentiment_distribution.png)

### Word Clouds
![WordClouds](results/wordclouds.png)

### Text Length Analysis
![Text Length](results/text_length_boxplot.png)

### Evaluation Summary
![Evaluation](results/evaluation_summary.png)

### ROC Curves
![ROC Curves](results/roc_curves.png)

## 🎓 Key Insights

1. **Preprocessing Impact**: Text preprocessing reduces word count by ~X%, improving model efficiency
2. **Class Balance**: [Observation about class distribution]
3. **Word Patterns**: [Key words/phrases associated with each sentiment]
4. **Model Performance**: DistilBERT achieves strong performance with efficient inference
5. **Text Length**: [Insights about text length vs sentiment relationship]

## 🛠️ Advanced Configuration

### Custom Preprocessing Options
```python
preprocessor = AdvancedTextPreprocessor(
    remove_stopwords=True,          # Remove common words
    apply_lemmatization=True,        # Use lemmatization
    apply_stemming=False,            # Alternative to lemmatization
    expand_contractions=True,        # Expand contractions
    remove_rare_words=True,          # Filter rare words
    rare_word_threshold=2            # Minimum word frequency
)
```

### Training Arguments Customization
```python
training_args = TrainingArguments(
    output_dir='./custom_results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update documentation as needed
- Add tests for new features
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **DistilBERT** authors for the efficient model architecture
- **NLTK** team for natural language processing tools
- **Kaggle** community for dataset and resources
- Contributors and maintainers of open-source libraries

## 🚀 Future Improvements

- Add support for multi-label classification.
- Experiment with advanced transformers (e.g., RoBERTa, DeBERTa, or T5).
- Integrate hyperparameter tuning.
- Deploy the model using FastAPI or Streamlit.

---

## 👨‍💻 Author

**Samrat Abdul Jalil**  
AI/ML Engineer | NLP & Computer Vision Engineer  
[LinkedIn](https://www.linkedin.com/in/samratabduljalil) | [GitHub](https://github.com/samratabduljalil)

---


**Made with ❤️ for the NLP community**

*Last Updated: October 2025*
