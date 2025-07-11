import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Trainer, TrainingArguments
from prepare_evaluation import load_dataset, load_model_and_tokenizer, preprocess_dataset, DATASET_CONFIG, RESULT_PATH
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_confusion_matrix(dataset_name, trainer, test_dataset, label_encoder, output_dir):
    """Tạo và lưu confusion matrix."""
    logger.info(f"Generating confusion matrix for {dataset_name}...")
    
    # Dự đoán trên tập kiểm tra
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(true_labels, pred_labels)
    class_names = label_encoder.classes_
    
    # Vẽ ma trận
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Lưu hình
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    for dataset_name in ['CICIDS-2017', 'BoT-IoT', 'Car-Hacking', 'IVN-IDS']:
        output_dir = os.path.join(RESULT_PATH, dataset_name)
        result_file = os.path.join(output_dir, 'eval_results.txt')
        
        # Kiểm tra xem dataset đã được đánh giá chưa
        if not os.path.exists(result_file):
            logger.warning(f"No evaluation results found for {dataset_name}, skipping...")
            continue
        
        # Tải dữ liệu
        dataset_dir = DATASET_CONFIG[dataset_name]['dir']
        features, labels = load_dataset(dataset_dir, dataset_name)
        if features is None or labels is None:
            logger.error(f"Failed to load dataset {dataset_name}")
            continue
        
        # Tải mô hình và tokenizer
        model_path = DATASET_CONFIG[dataset_name]['model_path']
        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {dataset_name}: {e}")
            continue
        
        # Chuẩn bị tập kiểm tra
        from sklearn.model_selection import train_test_split
        try:
            _, test_features, _, test_labels = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
            _, test_features, _, test_labels = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
        
        selected_classes = None
        if dataset_name == 'CICIDS-2017':
            selected_classes = ['BENIGN', 'Hulk', 'DDoS', 'PortScan']
        
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        test_dataset, label_encoder = preprocess_dataset(test_features, test_labels, tokenizer, encoder, selected_classes)
        
        # Thiết lập Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=32,
            logging_dir=output_dir,
            report_to='none',
            no_cuda=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args
        )
        
        # Tạo confusion matrix
        plot_confusion_matrix(dataset_name, trainer, test_dataset, label_encoder, output_dir)