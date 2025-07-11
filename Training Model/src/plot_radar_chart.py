import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_radar_chart(datasets, metrics, output_path):
    """Tạo radar chart cho các dataset."""
    logger.info("Generating radar chart...")
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    N = len(categories)
    
    # Góc cho mỗi chỉ số
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for dataset, values in datasets.items():
        values += values[:1]  # Đóng vòng
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=dataset)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.title('Model Performance Comparison Across Datasets')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Radar chart saved to {output_path}")

if __name__ == "__main__":
    RESULT_PATH = '/home/user/Desktop/AI/iov_bert_project/result'
    datasets = {}
    
    for dataset_name in ['CICIDS-2017', 'BoT-IoT', 'Car-Hacking', 'IVN-IDS']:
        result_file = os.path.join(RESULT_PATH, dataset_name, 'eval_results.txt')
        if not os.path.exists(result_file):
            logger.warning(f"No evaluation results found for {dataset_name}, skipping...")
            continue
        
        with open(result_file, 'r') as f:
            try:
                result = eval(f.read())  # Chuyển string thành dict
                datasets[dataset_name] = [
                    result['eval_accuracy'],
                    result['eval_precision'],
                    result['eval_recall'],
                    result['eval_f1']
                ]
            except Exception as e:
                logger.error(f"Error parsing {result_file}: {e}")
                continue
    
    if datasets:
        output_path = os.path.join(RESULT_PATH, 'radar_chart.png')
        plot_radar_chart(datasets, None, output_path)
    else:
        logger.error("No valid datasets found for radar chart")