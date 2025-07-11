import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def can_to_cbs(can_file, output_file):
    try:
        logger.info(f"Processing CAN file: {can_file}")
        cbs = []
        with open(can_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                data = parts[2]  # Dữ liệu CAN
                try:
                    byte_sentence = ' '.join([str(int(b, 16)) for b in data.split()])
                    cbs.append(byte_sentence)
                except ValueError:
                    continue
        with open(output_file, 'w', buffering=8192) as f:
            f.write('\n'.join(cbs))
        logger.info(f"Completed writing {len(cbs)} CBS to {output_file}")
    except Exception as e:
        logger.error(f"Error processing {can_file}: {e}")

if __name__ == '__main__':
    can_dirs = [
        '/home/user/Desktop/AI/iov_bert_project/data/finetune/Car-Hacking',
        '/home/user/Desktop/AI/iov_bert_project/data/finetune/IVN-IDS'
    ]
    for can_dir in can_dirs:
        if not os.path.exists(can_dir):
            logger.error(f"Directory {can_dir} does not exist")
            continue
        for f in os.listdir(can_dir):
            if f.endswith('.csv') or f.endswith('.log'):
                can_file = os.path.join(can_dir, f)
                output_file = os.path.join(can_dir, f'{f.split(".")[0]}_cbs.txt')
                can_to_cbs(can_file, output_file)