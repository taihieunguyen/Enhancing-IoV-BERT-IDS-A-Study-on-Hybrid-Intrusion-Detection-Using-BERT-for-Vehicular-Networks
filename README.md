# Enhancing-IoV-BERT-IDS-A-Study-on-Hybrid-Intrusion-Detection-Using-BERT-for-Vehicular-Networks
This project explores the use of Bidirectional Encoder Representations from Transformers (BERT) to build an AI-driven Intrusion Detection System (IDS) tailored for Internet of Vehicles (IoV) environments. By combining deep learning with network traffic analysis, the system aims to improve the accuracy and efficiency of detecting cyberattacks in vehicular networks.

The research evaluates various machine learning models, including a fine-tuned BERT architecture, and compares their performance on benchmark datasets. A hybrid BERT-based IDS is proposed to enhance security in latency-sensitive and safety-critical IoV applications.

## Key components
- **BERT Transformer Model**: Fine-tuned for network traffic classification in vehicular environments.
- **Data Preprocessing Module**:  Efficient cleaning, encoding, and normalization of network data.
- **Evaluation Framework**: Measures accuracy, precision, recall, F1-score, and AUC across models.
- **Hybrid Detection Architecture**: Combines BERT with traditional classifiers for optimized results.

## Architecture
![Infrastructure Architecture](https://github.com/taihieunguyen/Enhancing-IoV-BERT-IDS-A-Study-on-Hybrid-Intrusion-Detection-Using-BERT-for-Vehicular-Networks/blob/main/Architecture.png?raw=true)

## Key Features
- **BERT-Based IDS**: Leverages the power of Transformers to detect malicious traffic in vehicular networks.
- **Comprehensive Dataset Analysis**: Supports benchmark IoV network traffic datasets (CICIDS-2017, BoT-IoT, Car-Hacking,..)
- **Model Comparison**: Compares BERT with traditional ML models such as SVM, XGBoost, and Random Forest.
- **Performance Visualization**: Includes training metrics and confusion matrices for model evaluation.
- **Hybrid Ensemble**: Combines BERT with other models for improved precision and recall in anomaly detection.

## Repository Structure
── Article/                 # Academic articles, literature reviews, or related publications
├── Report/                  # Experiment reports, result analysis, and performance summaries
├── Training Model/          # Main module for model development and training
│   ├── Data/                # Dataset files or scripts for data loading
│   ├── result/              # Output files: model evaluation results, logs, and metrics
│   └── src/                 # Source code: data preprocessing, model training, and evaluation
├── Architecture.png         # Diagram showing system or model architecture
└── README.md                # Project overview and usage documentation

## Prerequisites
To run this project, install the following:
- **Python 3.10+.**
- **PyTorch.**
- **Scikit-learn.**
- **Pandas.**
- **NumPy.**
- **Matplotlib / Seaborn.**

## Usage
1. Preprocess the Dataset
   Prepare and clean the IoV dataset using scripts in /code/preprocessing.py.
2. Train BERT Model
   Fine-tune the BERT model using train_bert.py.
3. Evaluate Models
   Evaluate performance using confusion matrix, accuracy, and ROC curves via evaluate_models.py.
4. Compare Models
   Use compare_models.py to benchmark traditional ML vs. BERT-based models.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## Contact
For questions or support, open an issue on the GitHub repository or contact the maintainer at [taihieunguyen004@gmail.com].
