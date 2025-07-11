import torch
torch.set_num_threads(16)  # Tận dụng 16 CPU

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels.astype(np.int64)  # Ép kiểu nhãn thành int64
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze().to(dtype=torch.long),
            'attention_mask': encoding['attention_mask'].squeeze().to(dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Đảm bảo nhãn là torch.long
        }

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
                    # Chuẩn hóa đặc trưng: đảm bảo tất cả cột là chuỗi
                    data_features = data.drop(label_col, axis=1).astype(str).apply(lambda x: ' '.join(x), axis=1).values
                    # Chuẩn hóa nhãn: chuyển đổi sang chuỗi trước khi thay thế ký tự đặc biệt
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
                    # Đọc tệp _cbs.txt, bỏ qua dòng trống
                    with open(cbs_path, 'r') as f:
                        data_features = [line.strip() for line in f.read().splitlines() if line.strip()]
                    try:
                        # Đọc tệp .csv, bỏ qua dòng tiêu đề nếu cần
                        data = pd.read_csv(csv_path, low_memory=False)
                    except pd.errors.ParserError:
                        columns = [f'col_{i}' for i in range(12)]
                        columns[-1] = 'Flag'
                        data = pd.read_csv(csv_path, names=columns, low_memory=False)
                    label_col = find_label_column(data, csv_path, dataset_name)
                    if label_col is None:
                        continue
                    data_labels = data[label_col].astype(str).str.replace(r'[^\w\s]', '_', regex=True).fillna('Unknown').values
                    # Đồng bộ số dòng giữa đặc trưng và nhãn
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
    
    # Kiểm tra số nhãn duy nhất
    unique_labels = np.unique(labels)
    logger.info(f"Loaded {len(features)} samples for {dataset_name} with {len(unique_labels)} unique labels: {unique_labels[:5]}")
    if len(unique_labels) < 2:
        logger.error(f"Dataset {dataset_name} has fewer than 2 unique labels, cannot train classifier")
        return None, None
    
    return np.array(features), np.array(labels)

def get_latest_checkpoint(output_dir):
    """Tìm checkpoint mới nhất trong output_dir."""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def finetune_dataset(dataset_dir, dataset_name, pretrained_model_path, output_dir):
    """Fine-tuning trên một dataset."""
    logger.info(f"Fine-tuning on {dataset_name}")
    
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)
    
    features, labels = load_dataset(dataset_dir, dataset_name)
    if features is None or labels is None:
        return None
    
    # Giới hạn 100,000 mẫu với phân bố nhãn cân bằng
    if len(features) > 100000:
        logger.info(f"Reducing dataset to 100,000 samples from {len(features)}")
        df = pd.DataFrame({'features': features, 'labels': labels})
        # Đếm số mẫu mỗi nhãn
        label_counts = df['labels'].value_counts()
        # Loại nhãn có ít hơn 2 mẫu
        valid_labels = label_counts[label_counts >= 2].index
        df = df[df['labels'].isin(valid_labels)]
        if len(df) < 100000:
            logger.warning(f"After removing rare labels, only {len(df)} samples remain. Adjusting sample size.")
            df = df.sample(n=min(len(df), 100000), random_state=42)
        else:
            # Lấy mẫu phân tầng
            try:
                frac = 100000 / len(features)
                sampled_dfs = []
                for label in valid_labels:
                    label_df = df[df['labels'] == label]
                    sample_size = max(2, int(len(label_df) * frac))  # Đảm bảo ít nhất 2 mẫu
                    sampled_dfs.append(label_df.sample(n=sample_size, random_state=42))
                df = pd.concat(sampled_dfs)
                if len(df) > 100000:
                    df = df.sample(n=100000, random_state=42)
            except ValueError as e:
                logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
                df = df.sample(n=100000, random_state=42)
        features, labels = df['features'].values, df['labels'].values
    
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Kiểm tra số lớp sau mã hóa
    num_classes = len(encoder.classes_)
    logger.info(f"Number of classes after encoding for {dataset_name}: {num_classes}")
    if num_classes < 2:
        logger.error(f"After encoding, {dataset_name} has fewer than 2 classes, cannot train classifier")
        return None
    
    # Chia train/test, tắt stratify nếu số mẫu nhỏ
    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42
        )
    
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)
    
    try:
        config = BertConfig.from_pretrained(pretrained_model_path, num_labels=num_classes)
        model = BertForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        return None
    
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/{dataset_name}',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        logging_steps=100,
        report_to='none',
        dataloader_num_workers=12,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        accuracy = (predictions == torch.tensor(labels)).float().mean().item()
        return {"accuracy": accuracy}
    
    # Kiểm tra checkpoint
    latest_checkpoint = get_latest_checkpoint(f'{output_dir}/{dataset_name}')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    logger.info(f"Starting fine-tuning for {dataset_name}")
    if latest_checkpoint:
        logger.info(f"Resuming training from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        logger.info("Starting new training")
        trainer.train()
    logger.info(f"Fine-tuning completed for {dataset_name}")
    return trainer, test_dataset, test_labels

if __name__ == '__main__':
    base_dir = '/home/user/Desktop/AI/iov_bert_project'
    pretrained_model_path = f'{base_dir}/models/pretrained_bert'
    output_dir = f'{base_dir}/models/finetuned_bert'
    
    dataset_dirs = [
        (f'{base_dir}/data/finetune/CICIDS-2017', 'CICIDS-2017'),
        (f'{base_dir}/data/finetune/BoT-IoT', 'BoT-IoT'),
        (f'{base_dir}/data/finetune/Car-Hacking', 'Car-Hacking'),
        (f'{base_dir}/data/finetune/IVN-IDS', 'IVN-IDS')
    ]
    
    for dataset_dir, dataset_name in dataset_dirs:
        if not os.path.exists(dataset_dir):
            logger.error(f"Directory {dataset_dir} does not exist")
            continue
        result = finetune_dataset(dataset_dir, dataset_name, pretrained_model_path, output_dir)
        if result is None:
            logger.error(f"Failed to fine-tune {dataset_name}")
            continue