"""
📊 Dataset Collector for Digital Twin IDS
Collects and generates synthetic network traffic data with Digital Twin features
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DigitalTwinDataCollector:
    """Collects and generates Digital Twin network data"""
    
    def __init__(self, output_dir: str = "data/raw_collected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_counter = 1
        self.collection_stats = {
            'total_samples': 0,
            'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'batches': []
        }
        
        logger.info(f"📊 Initialized Data Collector - Output: {self.output_dir}")
    
    def generate_network_features(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic network traffic features"""
        
        # Basic network features (similar to NSL-KDD)
        data = {
            'duration': np.random.exponential(2.0, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
            'src_bytes': np.random.lognormal(5, 2, n_samples).astype(int),
            'dst_bytes': np.random.lognormal(4, 2, n_samples).astype(int),
            'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.05, n_samples),
            'hot': np.random.poisson(0.2, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'num_compromised': np.random.poisson(0.05, n_samples),
            'root_shell': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'su_attempted': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            'num_root': np.random.poisson(0.1, n_samples),
            'num_file_creations': np.random.poisson(0.2, n_samples),
            'num_shells': np.random.poisson(0.1, n_samples),
            'num_access_files': np.random.poisson(0.15, n_samples),
            'num_outbound_cmds': np.random.poisson(0.05, n_samples),
            'is_host_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(8, n_samples),
            'serror_rate': np.random.beta(1, 10, n_samples),
            'srv_serror_rate': np.random.beta(1, 10, n_samples),
            'rerror_rate': np.random.beta(1, 15, n_samples),
            'srv_rerror_rate': np.random.beta(1, 15, n_samples),
            'same_srv_rate': np.random.beta(5, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 5, n_samples),
            'srv_diff_host_rate': np.random.beta(1, 10, n_samples),
            'dst_host_count': np.random.poisson(20, n_samples),
            'dst_host_srv_count': np.random.poisson(15, n_samples),
            'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 5, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(3, 7, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 10, n_samples),
            'dst_host_serror_rate': np.random.beta(1, 10, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 10, n_samples),
            'dst_host_rerror_rate': np.random.beta(1, 15, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 15, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_digital_twin_features(self, n_samples: int) -> pd.DataFrame:
        """Generate Digital Twin specific features"""
        
        # Sync quality and timing
        sync_delay = np.random.exponential(50, n_samples)  # milliseconds
        sync_accuracy = np.random.beta(8, 2, n_samples)  # 0-1 scale
        
        data = {
            'sync_delay': sync_delay,
            'sync_accuracy': sync_accuracy,
            'sync_quality_score': sync_accuracy / (1 + sync_delay/100),
            'model_drift_score': np.random.beta(1, 9, n_samples),
            'prediction_confidence': np.random.beta(6, 2, n_samples),
            'model_reliability': np.random.beta(7, 2, n_samples),
            'update_frequency': np.random.exponential(10, n_samples),
            'data_freshness': np.random.beta(5, 2, n_samples),
            'computational_overhead': np.random.lognormal(2, 1, n_samples),
            'memory_usage': np.random.lognormal(3, 0.5, n_samples),
            'network_latency': np.random.exponential(20, n_samples),
            'processing_time': np.random.lognormal(1, 0.8, n_samples),
            'twin_state_consistency': np.random.beta(6, 2, n_samples),
            'sensor_reliability': np.random.beta(7, 2, n_samples),
            'communication_quality': np.random.beta(5, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_system_features(self, n_samples: int) -> pd.DataFrame:
        """Generate system and resource features"""
        
        data = {
            'cpu_usage': np.random.beta(3, 5, n_samples),
            'memory_usage_percent': np.random.beta(4, 6, n_samples),
            'disk_io': np.random.lognormal(2, 1, n_samples),
            'network_io': np.random.lognormal(3, 1, n_samples),
            'active_connections': np.random.poisson(50, n_samples),
            'process_count': np.random.poisson(100, n_samples),
            'thread_count': np.random.poisson(200, n_samples),
            'file_descriptors': np.random.poisson(150, n_samples),
            'system_load': np.random.exponential(1, n_samples),
            'uptime': np.random.exponential(86400, n_samples),  # seconds
            'temperature': np.random.normal(45, 10, n_samples),  # Celsius
            'power_consumption': np.random.lognormal(3, 0.5, n_samples),  # Watts
            'error_rate': np.random.beta(1, 20, n_samples),
            'response_time': np.random.lognormal(1, 1, n_samples),
            'throughput': np.random.lognormal(4, 1, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_labels(self, n_samples: int) -> pd.DataFrame:
        """Generate attack labels and categories"""
        
        # Attack probability (20% attacks)
        is_attack = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Attack categories
        attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        attack_category = []
        
        for attack in is_attack:
            if attack == 0:
                attack_category.append('normal')
            else:
                attack_category.append(np.random.choice(['dos', 'probe', 'r2l', 'u2r'], 
                                                      p=[0.5, 0.3, 0.15, 0.05]))
        
        # Severity levels
        severity_map = {'normal': 0, 'probe': 1, 'r2l': 2, 'u2r': 3, 'dos': 4}
        severity_level = [severity_map[cat] for cat in attack_category]
        
        data = {
            'is_attack': is_attack,
            'attack_category': attack_category,
            'severity_level': severity_level,
            'is_malicious': is_attack  # Binary version
        }
        
        return pd.DataFrame(data)
    
    def collect_batch(self, batch_size: int = 5000) -> Tuple[pd.DataFrame, Dict]:
        """Collect a batch of synthetic data"""
        
        logger.info(f"📊 Collecting batch {self.batch_counter} ({batch_size} samples)...")
        
        start_time = time.time()
        
        # Generate different feature types
        network_features = self.generate_network_features(batch_size)
        dt_features = self.generate_digital_twin_features(batch_size)
        system_features = self.generate_system_features(batch_size)
        labels = self.generate_labels(batch_size)
        
        # Combine all features
        batch_data = pd.concat([network_features, dt_features, system_features, labels], axis=1)
        
        # Add metadata
        batch_data['timestamp'] = pd.Timestamp.now()
        batch_data['batch_id'] = self.batch_counter
        
        collection_time = time.time() - start_time
        
        # Batch metadata
        batch_metadata = {
            'batch_id': self.batch_counter,
            'timestamp': datetime.now().isoformat(),
            'samples': batch_size,
            'features': len(batch_data.columns),
            'collection_time': collection_time,
            'attack_rate': labels['is_attack'].mean(),
            'feature_types': {
                'network': len(network_features.columns),
                'digital_twin': len(dt_features.columns),
                'system': len(system_features.columns),
                'labels': len(labels.columns)
            }
        }
        
        logger.info(f"✅ Batch {self.batch_counter} collected in {collection_time:.2f}s")
        logger.info(f"   Attack rate: {batch_metadata['attack_rate']:.3f}")
        
        return batch_data, batch_metadata
    
    def save_batch(self, batch_data: pd.DataFrame, batch_metadata: Dict) -> None:
        """Save batch data and metadata"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{self.batch_counter:04d}_{timestamp}"
        
        # Save CSV
        csv_path = self.output_dir / f"{batch_id}.csv"
        batch_data.to_csv(csv_path, index=False)
        
        # Save metadata
        json_path = self.output_dir / f"{batch_id}.json"
        with open(json_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        # Update collection stats
        self.collection_stats['total_samples'] += len(batch_data)
        self.collection_stats['total_batches'] += 1
        self.collection_stats['batches'].append(batch_metadata)
        
        logger.info(f"💾 Saved batch to {csv_path}")
        
        self.batch_counter += 1
    
    def collect_multiple_batches(self, num_batches: int = 5, batch_size: int = 5000) -> None:
        """Collect multiple batches of data"""
        
        logger.info(f"🚀 Starting collection of {num_batches} batches...")
        
        for i in range(num_batches):
            try:
                batch_data, batch_metadata = self.collect_batch(batch_size)
                self.save_batch(batch_data, batch_metadata)
                
                # Small delay between batches
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error collecting batch {i+1}: {e}")
                continue
        
        # Save final stats
        self.save_collection_stats()
        
        logger.info(f"🎉 Collection completed!")
        logger.info(f"   Total samples: {self.collection_stats['total_samples']:,}")
        logger.info(f"   Total batches: {self.collection_stats['total_batches']}")
    
    def save_collection_stats(self) -> None:
        """Save collection statistics"""
        
        self.collection_stats['end_time'] = datetime.now().isoformat()
        
        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.collection_stats, f, indent=2)
        
        logger.info(f"📊 Collection stats saved to {stats_path}")

def main():
    """Main collection function"""
    
    # Create collector
    collector = DigitalTwinDataCollector()
    
    # Collect data
    collector.collect_multiple_batches(num_batches=15, batch_size=5000)

if __name__ == "__main__":
    main()