import torch
torch.set_num_threads(16)  # Tận dụng 16 CPU

from transformers import BertForMaskedLM, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BertTokenizerFast
from datasets import Dataset
import logging
import os
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_latest_checkpoint(output_dir):
    """Tìm checkpoint mới nhất trong output_dir."""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    # Lấy checkpoint có số bước lớn nhất
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def main():
    base_dir = '/home/user/Desktop/AI/iov_bert_project'
    tokenized_dataset_path = f'{base_dir}/data/pretrain/tokenized_dataset'
    output_dir = f'{base_dir}/models/pretrained_bert'
    
    logger.info("Initializing fast tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    logger.info(f"Loading tokenized dataset from {tokenized_dataset_path}")
    if not os.path.exists(tokenized_dataset_path):
        logger.error(f"Tokenized dataset at {tokenized_dataset_path} does not exist")
        exit(1)
    tokenized_dataset = Dataset.load_from_disk(tokenized_dataset_path)
    
    logger.info("Initializing BERT model")
    config = BertConfig(
        vocab_size=65536,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024
    )
    model = BertForMaskedLM(config)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        num_train_epochs=1,
        save_steps=1000,  # Lưu mỗi 1000 bước (~16-32 phút)
        save_total_limit=6,  # Giữ tối đa 6 checkpoint
        fp16=False,
        logging_steps=500,
        report_to='none',
        dataloader_num_workers=14,
        dataloader_pin_memory=False
    )
    
    # Kiểm tra checkpoint
    latest_checkpoint = get_latest_checkpoint(output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    logger.info("Starting pre-training")
    if latest_checkpoint:
        logger.info(f"Resuming training from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        logger.info("Starting new training")
        trainer.train()
    logger.info("Pre-training completed")
    
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == '__main__':
    main()