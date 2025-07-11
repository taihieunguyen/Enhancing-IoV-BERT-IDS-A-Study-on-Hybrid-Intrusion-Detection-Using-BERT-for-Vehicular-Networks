from transformers import BertConfig, DataCollatorForLanguageModeling
from transformers import BertTokenizerFast
from datasets import Dataset
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = '/home/user/Desktop/AI/iov_bert_project'
    bsl_file = f'{base_dir}/data/pretrain/bsl.txt'
    tokenized_dataset_path = f'{base_dir}/data/pretrain/tokenized_dataset'
    
    if not os.path.exists(bsl_file):
        logger.error(f"BSL file {bsl_file} does not exist")
        exit(1)
    
    logger.info("Initializing fast tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    if os.path.exists(tokenized_dataset_path):
        logger.info(f"Removing old tokenized dataset at {tokenized_dataset_path}")
        shutil.rmtree(tokenized_dataset_path)
    
    logger.info("Loading BSL dataset")
    dataset = Dataset.from_text(bsl_file, cache_dir=f'{base_dir}/data/cache')
    dataset = dataset.select(range(min(500000, len(dataset))))  # Giảm xuống 500000
    logger.info(f"Reduced dataset to {len(dataset)} samples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,  # Giảm để tiết kiệm dung lượng
            return_tensors='np'
        )
    
    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=16,  # Giảm để tránh quá tải CPU
        remove_columns=['text'],
        desc="Tokenizing BSL"
    )
    
    logger.info(f"Saving tokenized dataset to {tokenized_dataset_path}")
    tokenized_dataset.save_to_disk(tokenized_dataset_path)

if __name__ == '__main__':
    main()