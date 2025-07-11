from scapy.all import rdpcap, PcapReader
import os
import multiprocessing as mp
from queue import Empty
import logging
from contextlib import contextmanager

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager cho multiprocessing.Pool để đảm bảo đóng pool."""
    pool = mp.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def process_pcap(pcap_file, window_size=2):
    """Xử lý từng tệp PCAP và trả về danh sách BSL."""
    try:
        logger.info(f"Processing {pcap_file}")
        sessions = {}
        # Sử dụng PcapReader để đọc luồng, tiết kiệm RAM
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                if not ('IP' in pkt and 'Raw' in pkt):
                    continue
                
                # Kiểm tra giao thức TCP hoặc UDP
                if 'TCP' in pkt:
                    proto = 'TCP'
                    sport = pkt['TCP'].sport
                    dport = pkt['TCP'].dport
                elif 'UDP' in pkt:
                    proto = 'UDP'
                    sport = pkt['UDP'].sport
                    dport = pkt['UDP'].dport
                else:
                    continue

                key = (pkt['IP'].src, pkt['IP'].dst, sport, dport, proto)
                if key not in sessions:
                    sessions[key] = []
                
                # Chuyển payload thành byte sentence hiệu quả hơn
                payload = pkt['Raw'].load
                # Sử dụng bytes.hex() để chuyển đổi nhanh hơn
                byte_sentence = ' '.join(f'{b:02x}' for b in payload)
                sessions[key].append(byte_sentence)
        
        # Tạo BSL từ sessions
        bsl = []
        for session in sessions.values():
            for i in range(len(session) - window_size + 1):
                bsl.append(session[i:i + window_size])
        
        logger.info(f"Completed {pcap_file}: {len(bsl)} BSL pairs")
        return bsl
    except Exception as e:
        logger.error(f"Error processing {pcap_file}: {e}")
        return []

def extract_bsl(pcap_files, output_file, window_size=2, num_workers=2):
    """Xử lý song song các tệp PCAP và lưu BSL."""
    bsl = []
    
    # Xử lý song song với multiprocessing
    with poolcontext(processes=num_workers) as pool:
        results = pool.starmap(process_pcap, [(f, window_size) for f in pcap_files])
    
    # Gộp kết quả
    for result in results:
        bsl.extend(result)
    
    # Lưu BSL với bộ đệm tối ưu
    logger.info(f"Writing {len(bsl)} BSL pairs to {output_file}")
    with open(output_file, 'w', buffering=8192) as f:
        for pair in bsl:
            f.write('\n'.join(pair) + '\n[SEP]\n')
    
    logger.info(f"Completed writing to {output_file}")

if __name__ == '__main__':
    pretrain_dir = '/home/user/Desktop/AI/iov_bert_project/data/pretrain'
    output_file = '/home/user/Desktop/AI/iov_bert_project/data/pretrain/bsl.txt'
    
    # Kiểm tra thư mục và tệp
    if not os.path.exists(pretrain_dir):
        logger.error(f"Directory {pretrain_dir} does not exist")
        exit(1)
    
    pcap_files = [os.path.join(pretrain_dir, f) for f in os.listdir(pretrain_dir) if f.endswith('.pcap')]
    if not pcap_files:
        logger.error(f"No .pcap files found in {pretrain_dir}")
        exit(1)
    
    logger.info(f"Found {len(pcap_files)} PCAP files")
    extract_bsl(pcap_files, output_file, window_size=2, num_workers=2)