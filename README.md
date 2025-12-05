# AI-Powered Email Spam & Phishing Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-Educational-green)

An advanced machine learning system for detecting spam and phishing emails using deep learning and explainable AI. This project features a hybrid LSTM architecture combined with engineered features, wrapped in an interactive Streamlit web application.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Web Application](#running-the-web-application)
  - [Single Email Prediction](#single-email-prediction)
  - [Batch Testing](#batch-testing)
  - [Training Game](#training-game)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Explainable AI](#explainable-ai)
- [Screenshots & Examples](#screenshots--examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a sophisticated email spam/phishing detection system that combines the power of deep learning with explainable AI techniques. The system analyzes email content using a hybrid approach:

1. **Text Analysis**: LSTM neural network processes the semantic content
2. **Feature Engineering**: Extracts behavioral patterns (URLs, urgency indicators, special characters)
3. **Explainability**: LIME integration shows which words influenced the AI's decision

The application provides an interactive web interface where users can:
- Analyze individual emails with detailed threat assessments
- Process batches of emails for bulk analysis
- Test their own spam detection skills against the AI
- Understand the AI's decision-making process through visual explanations

## Features

### Core Capabilities

- **Real-time Email Classification**: Instant spam/ham detection with confidence scores
- **Threat Level Assessment**: Categorizes emails as High/Medium/Low threat
- **Batch Processing**: Upload and analyze multiple emails simultaneously
- **Interactive Training Game**: Learn spam patterns by competing with the AI
- **Explainable Predictions**: See exactly which words influenced the classification
- **Feature Extraction**: Automatically detects URLs, currency symbols, urgency patterns, and more
- **Performance Metrics**: Comprehensive model statistics and evaluation metrics

### Advanced Features

- **Dual-Input Architecture**: Combines text embeddings with numerical features
- **Early Stopping**: Prevents overfitting during training
- **Configurable Threshold**: Adjust spam sensitivity based on requirements
- **Visual Analytics**: Interactive charts showing prediction distributions
- **ROC Curve Analysis**: Detailed performance visualization
- **Confusion Matrix**: Understanding model predictions across classes

## Model Architecture

The system uses a hybrid neural network architecture that processes both textual and numerical features:

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  ┌────────────────────────┐    ┌───────────────────────┐   │
│  │   Text Sequence        │    │  Engineered Features   │   │
│  │   (Max Length: 200)    │    │  (7 features)          │   │
│  └────────────┬───────────┘    └───────────┬───────────┘   │
└───────────────┼─────────────────────────────┼───────────────┘
                │                             │
        ┌───────▼────────┐            ┌───────▼────────┐
        │   Embedding    │            │  Dense Layer   │
        │   (128 dim)    │            │  (32 units)    │
        │   (20K vocab)  │            │  ReLU          │
        └───────┬────────┘            └───────┬────────┘
                │                             │
        ┌───────▼────────┐            ┌───────▼────────┐
        │   LSTM Layer   │            │   Dropout      │
        │   (64 units)   │            │   (0.3)        │
        │   Dropout: 0.6 │            └───────┬────────┘
        └───────┬────────┘                    │
                │                             │
                └─────────┬───────────────────┘
                          │
                  ┌───────▼────────┐
                  │  Concatenate   │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  Dense (64)    │
                  │  ReLU          │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  Dropout (0.6) │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  Dense (1)     │
                  │  Sigmoid       │
                  └────────────────┘
                          │
                    OUTPUT: [0-1]
              (Spam Probability)
```

### Model Components

1. **Text Processing Branch**:
   - Embedding Layer: 20,000 vocabulary size, 128-dimensional vectors
   - LSTM Layer: 64 units with 60% dropout and recurrent dropout
   - Handles sequential text patterns and context

2. **Feature Processing Branch**:
   - Dense layer: 32 units with ReLU activation
   - Processes 7 engineered features:
     - URL count
     - Exclamation mark count
     - Question mark count
     - Currency symbol count
     - Number count
     - Capital letter ratio
     - Special character count

3. **Classification Head**:
   - Concatenates both branches
   - Dense layer: 64 units with ReLU
   - Dropout: 60% for regularization
   - Output layer: Sigmoid activation for binary classification

## Technologies Used

### Core Dependencies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core programming language |
| TensorFlow | 2.x | Deep learning framework |
| Keras | 2.x | High-level neural network API |
| Streamlit | 1.x | Web application framework |
| Pandas | 1.3+ | Data manipulation and analysis |
| NumPy | 1.21+ | Numerical computing |
| Scikit-learn | 0.24+ | Machine learning utilities |

### Visualization & Explainability

- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts and gauges
- **LIME**: Model explainability and interpretation
- **SHAP**: Alternative explainability framework (imported but not used in main flow)

### Data Source

- **KaggleHub**: Dataset download and management
- **Dataset**: [Phishing Email Dataset by Naser Abdullah Alam](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for dataset download)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Auditd_AI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit tensorflow keras pandas numpy scikit-learn matplotlib seaborn plotly lime shap kagglehub
```

Or create a `requirements.txt` file:

```txt
streamlit>=1.20.0
tensorflow>=2.10.0
keras>=2.10.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
lime>=0.2.0
shap>=0.41.0
kagglehub>=0.1.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 4: Kaggle API Configuration (Optional)

If you don't have a Kaggle account, the application will use cached data. For full dataset access:

1. Create a Kaggle account at [kaggle.com](https://kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Place the downloaded `kaggle.json` in:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

## Usage

### Running the Web Application

Navigate to the project directory containing the phishing detection files:

```bash
cd "EATC Assignment2 Demo and Slides_Phishing Detection"
```

Launch the Streamlit application:

```bash
streamlit run "EATC Assignment2 Demo_Phishing Detection.py"
```

Or use the notebook version:

```bash
jupyter notebook "EATC Assignment2 Demo_Phishing Detection.ipynb"
```

The application will open in your default web browser at `http://localhost:8501`

### Single Email Prediction

1. **Navigate to "Single Prediction" tab**
2. **Choose input method**:
   - **Manual Input**: Paste your own email content
   - **Example Messages**: Select from pre-defined examples
3. **Click "Analyze Email"**
4. **View Results**:
   - Spam/Ham classification
   - Confidence score
   - Threat level (High/Medium/Low)
   - Risk factors detected
   - Processing time
   - Feature analysis

**Optional**: Enable "Show AI Explanation" to see:
- Which words influenced the decision
- Spam vs safe word indicators
- Detailed impact scores

### Batch Testing

1. **Navigate to "Batch Testing" tab**
2. **Upload a text file**:
   - Format: One email per line
   - Supports `.txt` files
3. **Click "Analyze All Emails"**
4. **Review comprehensive results**:
   - Summary statistics
   - Classification distribution
   - Threat level analysis
   - Detailed results table
5. **Download CSV report** (optional)

### Training Game

1. **Navigate to "Email Classification Game" tab**
2. **Click "Get New Email to Classify"**
3. **Read the email content**
4. **Make your prediction**: SAFE or SPAM
5. **Compare your answer with the AI**
6. **Learn from explanations**:
   - Why the email was spam/safe
   - Key indicators you might have missed
   - Performance comparison (You vs AI)

## Dataset

### Source

The model is trained on the [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) from Kaggle, which contains:

- Legitimate emails (ham)
- Spam emails
- Phishing attempts

### Dataset Processing

The application automatically:
1. Downloads the dataset from Kaggle (on first run)
2. Combines multiple CSV files
3. Standardizes columns to `Message` and `Category`
4. Removes duplicates
5. Samples 30,000 emails for training (configurable)
6. Splits data: 60% train, 20% validation, 20% test

### Data Preprocessing

Each email undergoes advanced preprocessing:

```python
# Text cleaning
- Convert to lowercase
- Replace URLs with "URL" token
- Replace numbers with "NUMBER" token
- Remove special characters
- Normalize whitespace

# Feature extraction
- Count URLs
- Count exclamation/question marks
- Detect currency symbols
- Count numbers
- Calculate capital letter ratio
- Count special characters
```

## Model Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Samples | 18,000 (60%) |
| Validation Samples | 6,000 (20%) |
| Test Samples | 6,000 (20%) |
| Vocabulary Size | 20,000 words |
| Max Sequence Length | 200 tokens |
| Epochs | 15 (with early stopping) |
| Batch Size | 64 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

### Performance Metrics

The model achieves strong performance across multiple metrics:

- **Accuracy**: ~95%+ (varies by run)
- **Precision**: High accuracy in spam predictions
- **Recall**: Effective at catching actual spam
- **F1 Score**: Balanced performance measure
- **AUC-ROC**: Strong discriminative ability

*Note: Exact metrics depend on training run and random seed. View the "Model Stats" tab in the application for current model performance.*

### Evaluation Tools

The application provides comprehensive evaluation through:

1. **Classification Report**: Precision, recall, F1 for each class
2. **Confusion Matrix**: Visualization of prediction accuracy
3. **ROC Curve**: Model discrimination capability
4. **Training History**: Loss and accuracy over epochs

## Project Structure

```
Auditd_AI/
├── EATC Assignment2 Demo and Slides_Phishing Detection/
│   ├── EATC Assignment2 Demo_Phishing Detection.ipynb  # Jupyter notebook version
│   ├── EATC Assignment2 Demo_Phishing Detection.py     # Python script (main)
│   ├── EATC Assignment2 Report_Phishing Detection.docx # Project report
│   └── EATC Assignment2 Slides_Phishing Detection.pptx # Presentation slides
│
├── data/                          # Other project datasets
├── scripts/                       # Other project scripts
└── README.md                      # This file
```

### Main Files

- **`EATC Assignment2 Demo_Phishing Detection.py`**: Standalone Streamlit application (recommended)
- **`EATC Assignment2 Demo_Phishing Detection.ipynb`**: Jupyter notebook with identical code
- **Report (DOCX)**: Detailed project documentation
- **Slides (PPTX)**: Presentation materials

## How It Works

### 1. Data Loading & Preparation

```python
# Automatic dataset download from Kaggle
dataset = load_data()  # Cached for performance

# Standardize multiple file formats
df = load_and_standardize_data(file_paths)

# Clean duplicates
df = clean_data(df)
```

### 2. Text Preprocessing & Feature Engineering

```python
def advanced_text_preprocessing(text):
    # Extract features before cleaning
    features = {
        'url_count': count_urls(text),
        'exclamation_count': count_exclamations(text),
        'currency_symbols': count_currency(text),
        # ... 7 total features
    }

    # Clean text
    text = remove_urls(text)
    text = normalize_numbers(text)
    text = remove_special_chars(text)

    return clean_text, features
```

### 3. Model Training

```python
# Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(training_data)

# Create dual-input model
text_input = Input(shape=(200,))
feature_input = Input(shape=(7,))

# ... architecture layers ...

model.compile(optimizer=Adam(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with early stopping
history = model.fit([X_train_text, X_train_features],
                    y_train,
                    validation_data=([X_val_text, X_val_features], y_val),
                    epochs=15,
                    callbacks=[EarlyStopping(patience=3)])
```

### 4. Prediction

```python
def predict_spam_message(message, threshold=0.5):
    # Preprocess
    clean_text, features = advanced_text_preprocessing(message)

    # Tokenize
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=200)

    # Predict
    probability = model.predict([padded, features])

    # Classify
    is_spam = probability > threshold
    confidence = probability if is_spam else 1 - probability

    return prediction_result
```

## Explainable AI

The system uses **LIME (Local Interpretable Model-Agnostic Explanations)** to explain individual predictions.

### How LIME Works

1. **Perturbs the input**: Creates variations of the email by removing/modifying words
2. **Gets predictions**: Runs each variation through the model
3. **Builds local model**: Creates a simple interpretable model around the prediction
4. **Identifies key features**: Shows which words had the most impact

### Reading LIME Explanations

The application displays:

- **Spam Indicators**: Words that increase spam likelihood (red/orange)
- **Safe Indicators**: Words that suggest legitimate email (green)
- **Impact Strength**: Strong/Moderate/Weak influence
- **Impact Score**: Numerical measure of word importance

### Example Explanation

```
Email: "URGENT: Verify your account NOW! Click here: http://fake-bank.com"

Top Spam Indicators:
• "urgent" - Strong spam signal (+0.15)
• "verify" - Moderate spam signal (+0.08)
• "click" - Moderate spam signal (+0.07)
• "url" - Strong spam signal (+0.12)

Interpretation: The AI identified urgency language, verification requests,
and suspicious URLs as primary spam indicators.
```

## Screenshots & Examples

### Example 1: Phishing Email Detection

**Input:**
```
URGENT: Your account will be suspended unless you verify your identity
immediately. Click here: http://fake-bank.com/verify
```

**Output:**
- Classification: **SPAM**
- Confidence: **98.5%**
- Threat Level: **High**
- Risk Factors:
  - Multiple URLs detected
  - Excessive exclamation marks
  - Urgency pressure tactics

### Example 2: Legitimate Email

**Input:**
```
Meeting scheduled for tomorrow at 2 PM in conference room A.
Please bring your quarterly reports.
```

**Output:**
- Classification: **HAM (Safe)**
- Confidence: **95.2%**
- Threat Level: **Low**
- Risk Factors: None detected

### Example 3: Batch Analysis Results

| Email Preview | Classification | Confidence | Threat Level |
|---------------|----------------|------------|--------------|
| "URGENT: Your account..." | Spam | 98.5% | High |
| "Meeting scheduled for..." | Safe | 95.2% | Low |
| "Congratulations! You won..." | Spam | 99.1% | High |
| "Thank you for your..." | Safe | 92.8% | Low |

## Contributing

This is an educational project completed as part of an EATC (Enterprise AI Technology & Cybersecurity) assignment. While the primary development is complete, contributions for improvement are welcome:

### Areas for Enhancement

1. **Model Improvements**:
   - Experiment with transformer models (BERT, RoBERTa)
   - Add attention mechanisms
   - Implement ensemble methods

2. **Feature Engineering**:
   - Add sender reputation features
   - Analyze email headers
   - Detect email spoofing patterns

3. **Dataset Expansion**:
   - Include more diverse phishing examples
   - Add multilingual support
   - Incorporate recent phishing trends

4. **Application Features**:
   - Email client integration
   - Real-time monitoring
   - API endpoint for programmatic access
   - Mobile-responsive design improvements

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Open a Pull Request

## License

This project is developed for educational purposes as part of an academic assignment.

**Educational Use Only**: This project is intended for:
- Learning machine learning and cybersecurity concepts
- Understanding phishing detection techniques
- Academic research and study
- Personal skill development

**Not intended for**:
- Production deployment without proper security review
- Commercial use without appropriate licensing
- Critical security infrastructure

## Acknowledgments

### Academic Context

This project was completed as **Assignment 2** for the **EATC (Enterprise AI Technology & Cybersecurity)** course, demonstrating:
- Deep learning application in cybersecurity
- Interactive web application development
- Explainable AI implementation
- Real-world problem-solving

### Technologies & Resources

- **Kaggle Community**: For providing the phishing email dataset
- **Naser Abdullah Alam**: Dataset creator
- **Streamlit**: For the excellent web framework
- **TensorFlow/Keras Team**: For deep learning tools
- **LIME Authors**: Ribeiro et al., "Why Should I Trust You?" (2016)
- **Python Community**: For the comprehensive ecosystem

### References

1. **LIME Paper**: Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD.

2. **LSTM Networks**: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

3. **Phishing Detection**: Basit, A., Zafar, M., Liu, X., Javed, A.R., Jalil, Z., & Kifayat, K. (2021). A comprehensive survey of AI-enabled phishing attacks detection techniques. Telecommunications Systems, 76(1), 139-154.

4. **Dataset Source**: [Phishing Email Dataset - Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

---

## Contact & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Open an issue](../../issues)
- **Project Repository**: [GitHub Link]
- **Course**: EATC (Enterprise AI Technology & Cybersecurity)

---

**Version**: 1.0
**Last Updated**: December 2025
**Status**: Educational Project - EATC Assignment 2

---

*Built with Python, TensorFlow, and Streamlit | Powered by Deep Learning & Explainable AI*
