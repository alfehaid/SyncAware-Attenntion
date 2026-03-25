"""
📊 Large Synthetic Dataset Generator
Generates large-scale synthetic datasets for Digital Twin IDS training
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeSyntheticGenerator:
    """Generates large synthetic datasets for training"""
    
    def __init__(self, output_dir: str = "data/synthetic_large"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_counter = 1
        self.total_samples = 0
        
        logger.info(f"📊 Initialized Large Synthetic Generator - Output: {self.output_dir}")
    
    def generate_enhanced_features(self, n_samples: int) -> pd.DataFrame:
        """Generate enhanced network and DT features"""
        
        # Enhanced network features
        data = {
            # Basic network metrics
            'duration': np.random.exponential(3.0, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.75, 0.2, 0.05]),
            'service': np.random.choice(['http', 'https', 'ftp', 'ssh', 'telnet', 'smtp', 'dns'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'SH'], n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
            'src_bytes': np.random.lognormal(6, 2.5, n_samples).astype(int),
            'dst_bytes': np.random.lognormal(5, 2.2, n_samples).astype(int),
            'land': np.random.choice([0, 1], n_samples, p=[0.995, 0.005]),
            'wrong_fragment': np.random.poisson(0.08, n_samples),
            'urgent': np.random.poisson(0.03, n_samples),
            
            # Connection features
            'hot': np.random.poisson(0.25, n_samples),
            'num_failed_logins': np.random.poisson(0.12, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
            'num_compromised': np.random.poisson(0.04, n_samples),
            'root_shell': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
            'su_attempted': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'num_root': np.random.poisson(0.08, n_samples),
            'num_file_creations': np.random.poisson(0.18, n_samples),
            'num_shells': np.random.poisson(0.06, n_samples),
            'num_access_files': np.random.poisson(0.12, n_samples),
            'num_outbound_cmds': np.random.poisson(0.03, n_samples),
            'is_host_login': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
            
            # Traffic analysis features
            'count': np.random.poisson(12, n_samples),
            'srv_count': np.random.poisson(9, n_samples),
            'serror_rate': np.random.beta(1, 12, n_samples),
            'srv_serror_rate': np.random.beta(1, 12, n_samples),
            'rerror_rate': np.random.beta(1, 18, n_samples),
            'srv_rerror_rate': np.random.beta(1, 18, n_samples),
            'same_srv_rate': np.random.beta(6, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 6, n_samples),
            'srv_diff_host_rate': np.random.beta(1, 12, n_samples),
            
            # Host-based features
            'dst_host_count': np.random.poisson(25, n_samples),
            'dst_host_srv_count': np.random.poisson(18, n_samples),
            'dst_host_same_srv_rate': np.random.beta(6, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 6, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(4, 8, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 12, n_samples),
            'dst_host_serror_rate': np.random.beta(1, 12, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 12, n_samples),
            'dst_host_rerror_rate': np.random.beta(1, 18, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 18, n_samples),
            
            # Enhanced Digital Twin features
            'sync_delay': np.random.exponential(45, n_samples),
            'sync_accuracy': np.random.beta(9, 2, n_samples),
            'sync_quality_score': np.random.beta(7, 3, n_samples),
            'model_drift_score': np.random.beta(1, 12, n_samples),
            'prediction_confidence': np.random.beta(8, 2, n_samples),
            'model_reliability': np.random.beta(8, 2, n_samples),
            'update_frequency': np.random.exponential(8, n_samples),
            'data_freshness': np.random.beta(6, 2, n_samples),
            'computational_overhead': np.random.lognormal(2.2, 1.1, n_samples),
            'memory_usage': np.random.lognormal(3.2, 0.6, n_samples),
            'network_latency': np.random.exponential(18, n_samples),
            'processing_time': np.random.lognormal(1.2, 0.9, n_samples),
            'twin_state_consistency': np.random.beta(7, 2, n_samples),
            'sensor_reliability': np.random.beta(8, 2, n_samples),
            'communication_quality': np.random.beta(6, 2, n_samples),
            
            # System resource features
            'cpu_usage': np.random.beta(4, 7, n_samples),
            'memory_usage_percent': np.random.beta(5, 6, n_samples),
            'disk_io': np.random.lognormal(2.5, 1.2, n_samples),
            'network_io': np.random.lognormal(3.5, 1.1, n_samples),
            'active_connections': np.random.poisson(55, n_samples),
            'process_count': np.random.poisson(110, n_samples),
            'thread_count': np.random.poisson(220, n_samples),
            'file_descriptors': np.random.poisson(160, n_samples),
            'system_load': np.random.exponential(1.2, n_samples),
            'uptime': np.random.exponential(90000, n_samples),
            'temperature': np.random.normal(47, 12, n_samples),
            'power_consumption': np.random.lognormal(3.2, 0.6, n_samples),
            'error_rate': np.random.beta(1, 25, n_samples),
            'response_time': np.random.lognormal(1.1, 1.1, n_samples),
            'throughput': np.random.lognormal(4.2, 1.1, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_realistic_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic attack labels based on features"""
        
        n_samples = len(features)
        
        # Create attack probability based on features
        attack_indicators = (
            (features['num_failed_logins'] > 2) * 0.3 +
            (features['root_shell'] == 1) * 0.4 +
            (features['num_compromised'] > 0) * 0.5 +
            (features['serror_rate'] > 0.5) * 0.2 +
            (features['dst_host_serror_rate'] > 0.3) * 0.2 +
            (features['sync_quality_score'] < 0.3) * 0.3 +
            (features['model_drift_score'] > 0.7) * 0.2
        )
        
        # Base attack probability (20%) modified by indicators
        base_prob = 0.2
        attack_probs = np.clip(base_prob + attack_indicators * 0.1, 0.05, 0.8)
        
        # Generate attacks based on probabilities
        is_attack = np.random.binomial(1, attack_probs)
        
        # Attack categories with realistic distribution
        attack_categories = []
        severity_levels = []
        
        for attack in is_attack:
            if attack == 0:
                attack_categories.append('normal')
                severity_levels.append(0)
            else:
                # Attack type distribution
                attack_type = np.random.choice(
                    ['dos', 'probe', 'r2l', 'u2r'], 
                    p=[0.45, 0.35, 0.15, 0.05]
                )
                attack_categories.append(attack_type)
                
                # Severity based on attack type
                severity_map = {'dos': 4, 'probe': 1, 'r2l': 2, 'u2r': 3}
                severity_levels.append(severity_map[attack_type])
        
        labels = {
            'is_attack': is_attack,
            'is_malicious': is_attack,  # Binary version
            'attack_category': attack_categories,
            'severity_level': severity_levels
        }
        
        return pd.DataFrame(labels)
    
    def generate_batch(self, batch_size: int = 5000) -> tuple:
        """Generate a batch of synthetic data"""
        
        logger.info(f"📊 Generating synthetic batch {self.batch_counter} ({batch_size} samples)...")
        
        start_time = time.time()
        
        # Generate features and labels
        features = self.generate_enhanced_features(batch_size)
        labels = self.generate_realistic_labels(features)
        
        # Combine data
        batch_data = pd.concat([features, labels], axis=1)
        
        # Add metadata
        batch_data['timestamp'] = pd.Timestamp.now()
        batch_data['batch_id'] = self.batch_counter
        batch_data['dataset_type'] = 'synthetic_large'
        
        generation_time = time.time() - start_time
        
        # Batch metadata
        batch_metadata = {
            'batch_id': self.batch_counter,
            'timestamp': datetime.now().isoformat(),
            'samples': batch_size,
            'features': len(batch_data.columns),
            'generation_time': generation_time,
            'attack_rate': labels['is_attack'].mean(),
            'attack_distribution': labels['attack_category'].value_counts().to_dict(),
            'dataset_type': 'synthetic_large'
        }
        
        logger.info(f"✅ Batch {self.batch_counter} generated in {generation_time:.2f}s")
        logger.info(f"   Attack rate: {batch_metadata['attack_rate']:.3f}")
        
        return batch_data, batch_metadata
    
    def save_batch(self, batch_data: pd.DataFrame, batch_metadata: dict):
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
        
        self.total_samples += len(batch_data)
        
        logger.info(f"💾 Saved synthetic batch to {csv_path}")
        
        self.batch_counter += 1
    
    def generate_dataset(self, num_batches: int = 10, batch_size: int = 5000):
        """Generate complete synthetic dataset"""
        
        logger.info(f"🚀 Starting large synthetic dataset generation...")
        logger.info(f"   Target: {num_batches} batches × {batch_size} samples = {num_batches * batch_size:,} total")
        
        start_time = time.time()
        
        for i in range(num_batches):
            try:
                batch_data, batch_metadata = self.generate_batch(batch_size)
                self.save_batch(batch_data, batch_metadata)
                
                # Small delay between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error generating batch {i+1}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Save collection stats
        stats = {
            'total_samples': self.total_samples,
            'total_batches': self.batch_counter - 1,
            'generation_time': total_time,
            'samples_per_second': self.total_samples / total_time,
            'dataset_type': 'synthetic_large',
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"🎉 Large synthetic dataset generation completed!")
        logger.info(f"   Total samples: {self.total_samples:,}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Speed: {stats['samples_per_second']:.0f} samples/second")

def main():
    """Main generation function"""
    
    # Create generator
    generator = LargeSyntheticGenerator()
    
    # Generate dataset (smaller for testing)
    generator.generate_dataset(num_batches=5, batch_size=5000)

if __name__ == "__main__":
    main()