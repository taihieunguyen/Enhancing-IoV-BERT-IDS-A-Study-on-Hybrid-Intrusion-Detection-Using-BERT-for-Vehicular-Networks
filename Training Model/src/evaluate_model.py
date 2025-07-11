import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import logging
import os
from sklearn.model_selection import train_test_split
from prepare_evaluation import load_dataset, load_model_and_tokenizer, preprocess_dataset, DATASET_CONFIG, RESULT_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    labels = labels
    accuracy = (predictions == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(dataset_name, features, labels, model, tokenizer, output_dir, selected_classes=None):
    logger.info(f"Evaluating model on {dataset_name}...")
    
    # Chia tập train/test
    try:
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
    
    # Chuẩn bị tập kiểm tra
    encoder = LabelEncoder()
    test_dataset, label_encoder = preprocess_dataset(test_features, test_labels, tokenizer, encoder, selected_classes)
    
    # Thiết lập Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        logging_dir=output_dir,
        logging_steps=100,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )
    
    # Đánh giá
    eval_results = trainer.evaluate(test_dataset)
    logger.info(f"Evaluation results for {dataset_name}: {eval_results}")
    
    return eval_results, trainer, test_dataset, label_encoder

if __name__ == "__main__":
    for dataset_name in ['CICIDS-2017', 'BoT-IoT', 'Car-Hacking', 'IVN-IDS']:
        # Kiểm tra xem dataset đã được đánh giá chưa
        output_dir = os.path.join(RESULT_PATH, dataset_name)
        result_file = os.path.join(output_dir, 'eval_results.txt')
        if os.path.exists(result_file):
            logger.info(f"Dataset {dataset_name} already evaluated, skipping...")
            continue
        
        # Tải dữ liệu
        dataset_dir = DATASET_CONFIG[dataset_name]['dir']
        features, labels = load_dataset(dataset_dir, dataset_name)
        if features is None or labels is None:
            logger.error(f"Failed to load dataset {dataset_name}")
            continue
        
        # Tải mô hình
        model_path = DATASET_CONFIG[dataset_name]['model_path']
        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {dataset_name}: {e}")
            continue
        
        # Đánh giá
        os.makedirs(output_dir, exist_ok=True)
        
        # Giới hạn lớp cho CICIDS-2017
        selected_classes = None
        if dataset_name == 'CICIDS-2017':
            selected_classes = ['BENIGN', 'Hulk', 'DDoS', 'PortScan']
            
        eval_results, trainer, test_dataset, label_encoder = evaluate_model(
            dataset_name, features, labels, model, tokenizer, output_dir, selected_classes
        )
        
        # Lưu kết quả
        with open(result_file, 'w') as f:
            f.write(str(eval_results))