# Complete Markdown Documentation for Educational Sentiment Analysis Notebook

## Cell 1: Package Installation
```markdown
## 📦 Package Installation

**Purpose**: Install and upgrade required Python packages for the project

**Packages**:
- `transformers`: Hugging Face library for transformer models
- `huggingface_hub`: Interface for Hugging Face model hub
- `wandb`: Weights & Biases for experiment tracking

**Action**: Run this cell first to ensure all dependencies are installed
```

---

## Cell 2: Import Libraries
```markdown
## 📚 Library Imports

**Purpose**: Import all necessary libraries for data processing, visualization, and model training

**Categories**:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud
- **NLP**: nltk, text preprocessing tools
- **Machine Learning**: sklearn metrics and utilities
- **Deep Learning**: torch, transformers
- **Text Analysis**: CountVectorizer, tokenizers

**Note**: Warnings are suppressed for cleaner output
```

---

## Cell 3: NLTK Data Download
```markdown
## 🔽 Download NLTK Resources

**Purpose**: Download required Natural Language Toolkit (NLTK) datasets

**Resources**:
- `punkt`: Tokenizer for splitting text into sentences and words
- `stopwords`: Common words to filter out (e.g., "the", "is", "and")
- `wordnet`: Lexical database for lemmatization
- `averaged_perceptron_tagger`: Part-of-speech tagger
- `punkt_tab`: Enhanced tokenization tables

**Usage**: These resources enable text preprocessing and tokenization
```

---

## Cell 4: Load Dataset
```markdown
## 📂 Dataset Loading

**Purpose**: Load the educational conversations sentiment dataset from CSV

**Dataset**: Educational_Conversations_with_Sentiment.csv

**Expected Columns**:
- `text`: Conversation text/messages
- `sentiment`: Sentiment labels (Positive/Neutral/Negative)

**Output**: Confirmation message upon successful loading
```

---

## Cell 5: EDA Section Header
```markdown
# 📊 Exploratory Data Analysis (EDA)

This section performs comprehensive analysis of the dataset to understand:
- Text characteristics and statistics
- Sentiment distribution
- Word patterns and frequencies
- Data quality and structure
```

---

## Cell 6: Text Statistics
```markdown
## 📈 Text Length and Word Count Analysis

**Purpose**: Calculate and display text statistics

**Metrics Calculated**:
- Character length per text
- Word count per text
- Average text length across dataset
- Average word count by sentiment category

**Insights**: Helps understand text complexity and variation across sentiments
```

---

## Cell 7: Basic Dataset Information
```markdown
## 🔍 Dataset Overview

**Purpose**: Display comprehensive dataset information

**Information Displayed**:
- First 5 rows (sample data)
- Dataset structure (columns, dtypes, memory usage)
- Missing values count
- Class distribution (sentiment counts)

**Importance**: Identifies data quality issues and class imbalance
```

---

## Cell 8: Sentiment Distribution Visualization
```markdown
## 📊 Sentiment Distribution Plot

**Purpose**: Visualize the distribution of sentiment classes

**Chart Type**: Count plot (bar chart)

**Features**:
- X-axis: Sentiment categories
- Y-axis: Number of samples
- Sorted by frequency
- Color palette: Set2

**Insight**: Shows class balance/imbalance in the dataset
```

---

## Cell 9: Text Length by Sentiment
```markdown
## 📏 Text Length Distribution Analysis

**Purpose**: Compare text lengths across different sentiment categories

**Visualization**: Box plot showing word count distribution

**Features**:
- Grouped by sentiment (Negative, Neutral, Positive)
- Shows median, quartiles, and outliers
- Identifies length patterns per sentiment

**Insight**: Reveals if certain sentiments tend to have longer/shorter texts
```

---

## Cell 10: Average Text Length Summary
```markdown
## 📊 Average Text Length by Sentiment

**Purpose**: Calculate and display mean text length for each sentiment

**Output**: Sorted list of average word counts per sentiment

**Use Case**: Quick numerical comparison of text verbosity across sentiments
```

---

## Cell 11: WordCloud Generation
```markdown
## ☁️ WordCloud Visualization

**Purpose**: Generate visual word clouds for each sentiment category

**Features**:
- Three separate word clouds (one per sentiment)
- Size indicates word frequency
- White background for clarity
- Side-by-side comparison

**Insight**: Visual representation of most common words per sentiment
```

---

## Cell 12: Top Words Analysis
```markdown
## 🔤 Most Frequent Words per Sentiment

**Purpose**: Identify and visualize top N words for each sentiment

**Method**:
- Uses CountVectorizer with English stopwords removed
- Displays top 10 words per sentiment
- Bar chart showing frequency

**Parameters**:
- `n`: Number of top words to display (default: 10)
- Stopwords: Automatically filtered

**Output**: Three bar charts (one per sentiment category)
```

---

## Cell 13: Bigram Analysis
```markdown
## 🔗 Bigram (Two-Word Phrase) Analysis

**Purpose**: Identify most common two-word combinations per sentiment

**Method**:
- Uses CountVectorizer with ngram_range=(2,2)
- Removes stopwords
- Displays top 10 bigrams per sentiment

**Use Case**: Reveals common phrases and word associations

**Output**: Bar charts showing bigram frequencies for each sentiment
```

---

## Cell 14: Word Frequency Heatmap
```markdown
## 🔥 Word Frequency Heatmap Across Sentiments

**Purpose**: Compare word usage patterns across all sentiment categories

**Method**:
- Extracts top 30 most frequent words overall
- Calculates average frequency per sentiment
- Creates heatmap visualization

**Features**:
- Rows: Words
- Columns: Sentiments
- Color intensity: Frequency (coolwarm palette)
- Annotated values

**Insight**: Identifies words strongly associated with specific sentiments
```

---

## Cell 15: Preprocessing Section Header
```markdown
# 🧹 Text Preprocessing

This section implements advanced text preprocessing techniques to:
- Clean and normalize text
- Remove noise and irrelevant information
- Standardize text format
- Prepare data for model training
```

---

## Cell 16: Preprocessing Class Header
```markdown
## 🛠️ Advanced Text Preprocessing Class

**Purpose**: Define a comprehensive text preprocessing class with multiple techniques

**Next Cell Contains**: Complete AdvancedTextPreprocessor class implementation
```

---

## Cell 17: Text Preprocessor Class
```markdown
## 🔧 AdvancedTextPreprocessor Class Implementation

**Purpose**: Comprehensive text preprocessing with configurable options

**Features**:
1. **Contraction Expansion**: "can't" → "cannot"
2. **Accent Removal**: "café" → "cafe"
3. **URL/Email Removal**: Cleans web links and emails
4. **HTML Tag Removal**: Removes markup
5. **Repeated Character Reduction**: "sooooo" → "soo"
6. **Stopword Removal**: Keeps negations and intensifiers
7. **Lemmatization**: Reduces words to base form
8. **Stemming**: Alternative to lemmatization
9. **Rare Word Filtering**: Removes infrequent words
10. **Sentiment Normalization**: Standardizes sentiment words

**Parameters**:
- `remove_stopwords`: Remove common words (default: True)
- `apply_lemmatization`: Use lemmatization (default: True)
- `apply_stemming`: Use stemming (default: False)
- `expand_contractions`: Expand contractions (default: True)
- `remove_rare_words`: Filter rare words (default: False)
- `rare_word_threshold`: Minimum word frequency (default: 1)

**Methods**:
- `preprocess()`: Main preprocessing pipeline
- `clean_text()`: Text cleaning and normalization
- `tokenize_text()`: Word tokenization
- `lemmatize_tokens()`: Lemmatization
- `stem_tokens()`: Stemming
- And more helper methods...

**Usage**: Initialize with desired parameters and call `preprocess(text)`
```

---

## Cell 18: Preprocessing Application Header
```markdown
## ⚙️ Apply Preprocessing Configuration

**Purpose**: Configure preprocessing settings based on your dataset needs

**Recommendation**: Adjust parameters based on your specific requirements
```

---

## Cell 19: Apply Preprocessing
```markdown
## 🎯 Execute Text Preprocessing

**Purpose**: Apply preprocessing pipeline to entire dataset

**Configuration**:
- ✓ Text cleaning (URLs, emails, HTML)
- ✓ Lowercasing
- ✓ Accent removal
- ✓ Contraction expansion
- ✓ Special character removal
- ✓ Repeated character reduction
- ✓ Tokenization
- ✓ Short word removal (< 2 chars)
- ✓ Stopword removal (preserving negations)
- ✓ Lemmatization
- ✓ Sentiment word normalization

**Steps**:
1. Initialize preprocessor with chosen parameters
2. Backup original text
3. Build vocabulary (if rare word filtering enabled)
4. Apply preprocessing to all texts
5. Clean sentiment labels
6. Remove empty texts
7. Remove duplicates

**Output**:
- New column: `preprocessed_text`
- Statistics on removed duplicates
- Total samples after preprocessing
```

---

## Cell 20: Preprocessing Analysis
```markdown
## 📊 Preprocessing Impact Analysis

**Purpose**: Analyze and visualize the effects of preprocessing

**Analysis Includes**:
1. **Statistics**:
   - Average character count (before/after)
   - Average word count (before/after)
   - Median word count
   - Word count reduction percentage

2. **Visualizations**:
   - Histogram of preprocessed word counts
   - Before vs After comparison histogram
   - Boxplot comparison
   - Sentiment distribution bar chart
   - Word count reduction pie chart

**Insight**: Quantifies preprocessing effectiveness and data transformation
```

---

## Cell 21: Preprocessing Examples
```markdown
## 📝 Preprocessing Examples

**Purpose**: Display side-by-side comparison of original and preprocessed texts

**Output**: Shows 10 sample examples with:
- Original text
- Preprocessed text
- Sentiment label

**Use Case**: Manual verification of preprocessing quality
```

---

## Cell 22: Label Encoding and Data Split
```markdown
## 🏷️ Label Encoding and Dataset Splitting

**Purpose**: Prepare data for model training

**Label Mapping**:
- Positive → 2
- Neutral → 1
- Negative → 0

**Data Split**:
- Training: 68% (80% of 85%)
- Validation: 12% (15% of 85%)
- Test: 20%

**Stratification**: Maintains class distribution across splits

**Output**: Train, validation, and test dataframes with sample counts
```

---

## Cell 23: Dataset Class Definition
```markdown
## 🗂️ Custom PyTorch Dataset Class

**Purpose**: Create a custom dataset class for PyTorch DataLoader

**Class**: `SentimentDataset`

**Features**:
- Tokenizes text on-the-fly
- Pads/truncates to max_length (default: 128)
- Returns input_ids, attention_mask, and labels
- Compatible with Hugging Face transformers

**Parameters**:
- `texts`: Text data
- `labels`: Sentiment labels
- `tokenizer`: Hugging Face tokenizer
- `max_length`: Maximum sequence length

**Usage**: Wraps data for efficient batch processing during training
```

---

## Cell 24: Model Setup
```markdown
## 🤖 Model Configuration and Loading

**Purpose**: Load pre-trained transformer model for fine-tuning

**Model**: DistilBERT (distilbert-base-uncased)
- Size: ~66M parameters
- Efficient distilled version of BERT
- Good balance of speed and performance

**Configuration**:
- Number of labels: 3 (Negative, Neutral, Positive)
- Label mappings configured
- Tokenizer loaded

**Output**:
- Model statistics (total and trainable parameters)
- Model size in millions (M)

**Note**: Can be replaced with other models like RoBERTa, ALBERT, etc.
```

---

## Cell 25: Create Dataset Objects
```markdown
## 📦 Prepare Dataset Objects

**Purpose**: Create PyTorch Dataset objects for training, validation, and testing

**Creates**:
- `train_dataset`: Training data
- `val_dataset`: Validation data
- `test_dataset`: Test data

**Features**:
- Uses preprocessed text
- Encoded labels
- Tokenization applied
- Ready for DataLoader

**Next Step**: Configure training arguments
```

---

## Cell 26: Training Configuration
```markdown
## ⚙️ Training Arguments Configuration

**Purpose**: Define hyperparameters and training settings

**Key Parameters**:
- **Epochs**: 3
- **Batch Size**: 8 (train and eval)
- **Learning Rate**: 5e-5
- **Warmup Steps**: 30
- **Weight Decay**: 0.01
- **FP16**: Enabled if GPU available

**Evaluation & Saving**:
- Evaluation: Every epoch
- Saving: Every epoch (keep best 2)
- Metric: Accuracy
- Load best model at end

**Logging**:
- Strategy: Every 5 steps
- Directory: ./logs

**Output Directory**: ./results

**Optimization**: Mixed precision training for faster computation
```

---

## Cell 27: Metrics Function
```markdown
## 📊 Evaluation Metrics Function

**Purpose**: Define metrics for model evaluation

**Metrics Computed**:
1. **Accuracy**: Overall correctness
2. **F1-Score**: Weighted harmonic mean of precision and recall
3. **Precision**: Weighted precision across classes
4. **Recall**: Weighted recall across classes

**Weighting**: Uses 'weighted' average to account for class imbalance

**Usage**: Called automatically by Trainer during evaluation
```

---

## Cell 28: Model Training
```markdown
## 🚀 Model Training

**Purpose**: Train the sentiment classification model

**Process**:
1. Initialize Trainer with model and arguments
2. Execute training loop
3. Validate on validation set after each epoch
4. Save best model based on accuracy
5. Log training metrics

**Duration**: Varies based on:
- Dataset size
- Hardware (CPU/GPU)
- Number of epochs
- Batch size

**Output**: Training logs with loss and accuracy metrics

**Note**: Best model automatically loaded at end
```

---

## Cell 29: Comprehensive Evaluation
```markdown
## 📈 Comprehensive Model Evaluation & Visualization

**Purpose**: Evaluate model performance on test set with detailed visualizations

**Evaluation Components**:

1. **Test Set Evaluation**: Overall metrics on unseen data

2. **Classification Report**: Per-class precision, recall, F1-score

3. **Confusion Matrices**: 
   - Raw counts
   - Normalized percentages

4. **ROC Curves**: Multi-class ROC analysis with AUC scores

5. **Visualizations** (2x3 grid):
   - Confusion Matrix
   - Normalized Confusion Matrix
   - Per-Class Metrics Bar Chart
   - Multi-Class ROC Curves
   - Class Distribution (True vs Predicted)
   - Performance Summary Box

**Metrics Displayed**:
- Accuracy, Precision, Recall, F1-Score
- Macro and Weighted averages
- AUC-ROC scores per class
- Macro AUC

**Output**: 
- Printed classification report
- Saved visualization (evaluation_summary.png)
- High-resolution plot (300 DPI)

**Use Case**: Complete performance analysis for model assessment
```

---

## Cell 30: Save Model
```markdown
## 💾 Save Trained Model

**Purpose**: Save the fine-tuned model and tokenizer for deployment

**Saved Components**:
- Model weights and configuration
- Tokenizer and vocabulary
- Label mappings

**Save Location**: ./canteen_sentiment_model_100m

**Output**:
- Confirmation message with path
- Model size on disk (MB)

**Usage**: Load later for inference using:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)
```

**Deployment Ready**: Model can be uploaded to Hugging Face Hub or deployed locally
```

---

## 📌 Complete Notebook Structure Summary

```
1. Setup & Installation (Cells 1-3)
2. Data Loading (Cell 4)
3. Exploratory Data Analysis (Cells 5-14)
4. Text Preprocessing (Cells 15-21)
5. Data Preparation (Cells 22-25)
6. Model Training (Cells 26-28)
7. Evaluation & Saving (Cells 29-30)
```

## 🎯 Best Practices

- Run cells sequentially
- Adjust preprocessing parameters based on your dataset
- Monitor training metrics for overfitting
- Experiment with different models and hyperparameters
- Save multiple model checkpoints
- Document any modifications

## 📚 Additional Notes

- GPU recommended for faster training
- Adjust batch size based on available memory
- Consider data augmentation for small datasets
- Use cross-validation for robust evaluation
- Track experiments with wandb (optional)
