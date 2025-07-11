import os
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_label_column(data, file_path, dataset_name):
    """Tìm cột nhãn dựa trên dataset và từ khóa, bỏ qua khoảng trắng."""
    if dataset_name == 'CICIDS-2017':
        for col in data.columns:
            if col.strip() == 'Label':
                logger.info(f"Using '{col}' column in {file_path}: {data[col].unique()[:5]}")
                return col
    elif dataset_name == 'BoT-IoT':
        for col in data.columns:
            col_stripped = col.strip()
            if col_stripped == 'attack':
                logger.info(f"Using '{col}' column in {file_path}: {data[col].unique()[:5]}")
                return col
            elif col_stripped == 'category':
                logger.info(f"Using '{col}' column in {file_path}: {data[col].unique()[:5]}")
                return col
    else:  # Car-Hacking, IVN-IDS
        label_col = data.columns[-1]
        logger.info(f"Using last column '{label_col}' in {file_path}: {data[label_col].unique()[:5]}")
        return label_col
    
    label_keywords = ['label', 'class', 'attack', 'category', 'flag', 'type']
    for col in data.columns:
        if any(keyword in col.lower().strip() for keyword in label_keywords):
            logger.info(f"Found potential label column '{col}' in {file_path}: {data[col].unique()[:5]}")
            return col
    logger.error(f"No label column found in {file_path}")
    return None

def load_dataset(dataset_dir, dataset_name):
    """Tải và gộp dữ liệu từ các tệp trong thư mục dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    features = []
    labels = []
    
    if dataset_name in ['CICIDS-2017', 'BoT-IoT']:
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dataset_dir, file_name)
                try:
                    data = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                    label_col = find_label_column(data, file_path, dataset_name)
                    if label_col is None:
                        continue
                    data_features = data.drop(label_col, axis=1).astype(str).apply(lambda x: ' '.join(x), axis=1).values
                    data_labels = data[label_col].astype(str).str.replace(r'[^\w\s]', '_', regex=True).fillna('Unknown').values
                    features.extend(data_features)
                    labels.extend(data_labels)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
    else:  # Car-Hacking, IVN-IDS
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith('_cbs.txt'):
                cbs_path = os.path.join(dataset_dir, file_name)
                csv_path = os.path.join(dataset_dir, file_name.replace('_cbs.txt', '.csv'))
                if not os.path.exists(csv_path):
                    logger.warning(f"No corresponding CSV for {cbs_path}")
                    continue
                try:
                    with open(cbs_path, 'r') as f:
                        data_features = [line.strip() for line in f.read().splitlines() if line.strip()]
                    try:
                        data = pd.read_csv(csv_path, low_memory=False)
                    except pd.errors.ParserError:
                        columns = [f'col_{i}' for i in range(12)]
                        columns[-1] = 'Flag'
                        data = pd.read_csv(csv_path, names=columns, low_memory=False)
                    label_col = find_label_column(data, csv_path, dataset_name)
                    if label_col is None:
                        continue
                    data_labels = data[label_col].astype(str).str.replace(r'[^\w\s]', '_', regex=True).fillna('Unknown').values
                    min_length = min(len(data_features), len(data_labels))
                    if len(data_features) != len(data_labels):
                        logger.warning(f"Mismatch between features ({len(data_features)}) and labels ({len(data_labels)}) in {file_name}. Truncating to {min_length} samples.")
                        data_features = data_features[:min_length]
                        data_labels = data_labels[:min_length]
                    features.extend(data_features)
                    labels.extend(data_labels)
                except Exception as e:
                    logger.error(f"Error loading {cbs_path} or {csv_path}: {e}")
    
    if not features or not labels:
        logger.error(f"No valid data loaded for {dataset_name}")
        return None, None
    
    unique_labels = np.unique(labels)
    logger.info(f"Loaded {len(features)} samples for {dataset_name} with {len(unique_labels)} unique labels: {unique_labels[:5]}")
    return np.array(features), np.array(labels)

def load_model_and_tokenizer(model_path):
    """Tải mô hình và tokenizer từ checkpoint cục bộ."""
    logger.info(f"Loading model and tokenizer from {model_path}...")
    
    # Kiểm tra sự tồn tại của thư mục checkpoint
    if not os.path.exists(model_path):
        logger.error(f"Checkpoint directory {model_path} does not exist")
        raise FileNotFoundError(f"Checkpoint directory {model_path} does not exist")
    
    # Kiểm tra các file cần thiết
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'vocab.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logger.error(f"Missing required file {file} in {model_path}")
            raise FileNotFoundError(f"Missing required file {file} in {model_path}")
    
    try:
        model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        logger.error(f"Error loading model or tokenizer from {model_path}: {e}")
        raise
    
    return model, tokenizer

def preprocess_dataset(features, labels, tokenizer, encoder, selected_classes=None):
    """Chuẩn bị dataset cho đánh giá."""
    logger.info("Preprocessing dataset...")
    if selected_classes:
        mask = np.isin(labels, selected_classes)
        features = features[mask]
        labels = labels[mask]
    encoded_labels = encoder.fit_transform(labels)
    dataset = Dataset.from_dict({
        'text': features,
        'labels': encoded_labels
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset, encoder

# Cấu hình
BASE_DIR = '/home/user/Desktop/AI/iov_bert_project'
DATA_PATH = f'{BASE_DIR}/data/finetune'
MODEL_PATH = f'{BASE_DIR}/models/finetuned_bert'
RESULT_PATH = f'{BASE_DIR}/result'

DATASET_CONFIG = {
    'CICIDS-2017': {
        'dir': f'{DATA_PATH}/CICIDS-2017',
        'model_path': f'{MODEL_PATH}/CICIDS-2017/checkpoint-5000'
    },
    'BoT-IoT': {
        'dir': f'{DATA_PATH}/BoT-IoT',
        'model_path': f'{MODEL_PATH}/BoT-IoT/checkpoint-5000'
    },
    'Car-Hacking': {
        'dir': f'{DATA_PATH}/Car-Hacking',
        'model_path': f'{MODEL_PATH}/Car-Hacking/checkpoint-5000'
    },
    'IVN-IDS': {
        'dir': f'{DATA_PATH}/IVN-IDS',
        'model_path': f'{MODEL_PATH}/IVN-IDS/checkpoint-5000'
    }
}