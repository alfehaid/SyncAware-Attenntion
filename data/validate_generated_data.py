"""
🔍 Data Validation Script
Quick validation of generated synthetic datasets
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataset(data_dir: str, dataset_name: str):
    """Validate a generated dataset"""
    
    logger.info(f"🔍 Validating {dataset_name} dataset...")
    
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        logger.error(f"❌ No CSV files found in {data_dir}")
        return False
    
    total_samples = 0
    total_features = 0
    attack_rates = []
    
    # Validate each batch
    for csv_file in csv_files[:3]:  # Check first 3 files
        try:
            df = pd.read_csv(csv_file)
            
            # Basic validation
            assert len(df) > 0, f"Empty file: {csv_file}"
            assert not df.isnull().all().any(), f"All-null columns in {csv_file}"
            
            # Check required columns
            required_cols = ['is_attack', 'attack_category']
            for col in required_cols:
                assert col in df.columns, f"Missing column {col} in {csv_file}"
            
            # Calculate stats
            total_samples += len(df)
            total_features = len(df.columns)
            attack_rate = df['is_attack'].mean()
            attack_rates.append(attack_rate)
            
            logger.info(f"   ✅ {csv_file.name}: {len(df):,} samples, {attack_rate:.3f} attack rate")
            
        except Exception as e:
            logger.error(f"   ❌ {csv_file.name}: {e}")
            return False
    
    # Overall stats
    avg_attack_rate = np.mean(attack_rates)
    
    logger.info(f"📊 {dataset_name} Summary:")
    logger.info(f"   Files checked: {len(csv_files)}")
    logger.info(f"   Total samples: {total_samples:,}")
    logger.info(f"   Features per sample: {total_features}")
    logger.info(f"   Average attack rate: {avg_attack_rate:.3f}")
    
    # Validation checks
    checks = [
        (total_samples > 0, "Has samples"),
        (total_features > 50, "Sufficient features"),
        (0.1 < avg_attack_rate < 0.3, "Realistic attack rate"),
        (len(csv_files) >= 3, "Multiple batches")
    ]
    
    all_passed = True
    for check, description in checks:
        status = "✅" if check else "❌"
        logger.info(f"   {status} {description}")
        if not check:
            all_passed = False
    
    return all_passed

def main():
    """Main validation function"""
    
    logger.info("🔍 Starting Data Validation...")
    
    datasets = [
        ("data/raw_collected", "Raw Collected"),
        ("data/synthetic_large", "Synthetic Large")
    ]
    
    results = {}
    
    for data_dir, dataset_name in datasets:
        logger.info(f"\n{'='*50}")
        try:
            result = validate_dataset(data_dir, dataset_name)
            results[dataset_name] = result
        except Exception as e:
            logger.error(f"❌ Error validating {dataset_name}: {e}")
            results[dataset_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📋 VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for dataset_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {dataset_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("🎉 All datasets passed validation!")
    else:
        logger.warning("⚠️  Some datasets failed validation")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)