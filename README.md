# Automated Essay Scorer (AES)

Automated Essay Scorer (AES) is a Django-based web application that leverages Deep Learning to provide instant grading for student essays. The system utilizes **DistilBERT** embeddings and a **BiLSTM** (Bidirectional Long Short-Term Memory) neural network to analyze and score submissions.

## 🚀 Features

* **User Authentication**: Custom user model with profile image support and mobile number fields.
* **Dual Submission Methods**: Users can either type their essay directly into a text area or upload a text file for grading.
* **Deep Learning Backend**: 
    * Uses **HuggingFace Transformers** (DistilBERT) for text tokenization and feature extraction.
    * Uses a **Keras/TensorFlow** BiLSTM model to predict essay scores.
* **Interactive UI**: Includes dedicated pages for student accounts, teacher information, and contact forms.

## 🛠️ Tech Stack

* **Backend**: Django 4.2.1
* **Database**: SQLite3
* **Machine Learning**:
    * TensorFlow / Keras
    * PyTorch
    * HuggingFace Transformers
    * Scikit-learn

## 📂 Project Structure

```text
aes/
├── aes/                  # Project configuration (settings, URLs)
├── automatedgrader/      # Main application logic
│   ├── ml_model/         # Pre-trained .h5 models
│   ├── static/           # CSS and JavaScript files
│   ├── templates/        # HTML layouts and views
│   ├── models.py         # Custom User database model
│   └── views.py          # Grading and Auth logic
├── manage.py             # Django management script
└── db.sqlite3            # Local database
