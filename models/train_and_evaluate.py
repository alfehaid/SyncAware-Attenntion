"""
🎯 Complete Training and Evaluation Pipeline
Train and evaluate DT-Hybrid model on real and synthetic datasets
"""

import pandas as pd
import numpy as np
import sys
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))
from models.dt_hybrid_simplified import DTHybridSimplified, create_dt_hybrid_simplified
from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DTIDSTrainer:
    """Complete training and evaluation system for DT-IDS"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.trained_models = {}
        self.preprocessors = {}
        
        logger.info("🎯 Initializing DT-IDS Training Pipeline...")
    
    def load_dataset(self, dataset_type: str, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Load dataset (real or synthetic)"""
        
        if dataset_type == "real":
            logger.info("📂 Loading Real NSL-KDD Dataset...")
            data_path = Path("data/real_datasets/nsl_kdd_with_dt_processed.csv")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Real dataset not found at {data_path}")
            
            if sample_size:
                df = pd.read_csv(data_path, nrows=sample_size)
            else:
                df = pd.read_csv(data_path)
            
            # Prepare features and target
            target_col = 'is_attack'
            exclude_cols = ['is_attack', 'attack_category', 'label', 'difficulty']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df[target_col]
            dataset_name = f"Real_NSL-KDD_{len(df)}"
            
        elif dataset_type == "synthetic":
            logger.info("📂 Loading Synthetic Large Dataset...")
            data_dir = Path("data/synthetic_large")
            csv_files = sorted(list(data_dir.glob("*.csv")))
            
            if not csv_files:
                raise FileNotFoundError("No synthetic data found")
            
            # Load specified number of batches
            num_batches = min(3, len(csv_files)) if sample_size is None else max(1, sample_size // 5000)
            
            dataframes = []
            for csv_file in csv_files[:num_batches]:
                batch_df = pd.read_csv(csv_file)
                dataframes.append(batch_df)
                logger.info(f"Loaded {csv_file.name}: {batch_df.shape}")
            
            df = pd.concat(dataframes, ignore_index=True)
            
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Prepare features and target
            target_col = 'is_malicious'
            exclude_cols = ['is_malicious', 'is_attack', 'attack_category', 'severity_level']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df[target_col]
            dataset_name = f"Synthetic_Large_{len(df)}"
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        logger.info(f"✅ Loaded {dataset_name}: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, dataset_name
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess data using DT-Aware preprocessor"""
        
        logger.info(f"🔧 Preprocessing {dataset_name}...")
        
        # Create preprocessor
        config = PreprocessingConfig(
            numerical_imputation="knn",
            scaling_method="adaptive",
            create_dt_specific_features=True,
            create_temporal_features=True,
            create_interaction_features=True,
            feature_selection_method="mutual_info",
            n_features_to_select=50
        )
        
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit preprocessor on training data
        start_time = time.time()
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        preprocessing_time = time.time() - start_time
        
        # Store preprocessor
        self.preprocessors[dataset_name] = preprocessor
        
        logger.info(f"✅ Preprocessing completed in {preprocessing_time:.2f}s")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Processed shape: {X_train_processed.shape}")
        logger.info(f"  Features reduced: {X.shape[1] - X_train_processed.shape[1]}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, dataset_name: str, config: Dict = None) -> DTHybridSimplified:
        """Train DT-Hybrid model"""
        
        logger.info(f"🧠 Training DT-Hybrid model on {dataset_name}...")
        
        # Create model
        model = create_dt_hybrid_simplified(config)
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Store model
        self.trained_models[dataset_name] = model
        
        logger.info(f"✅ Training completed in {training_time:.2f}s")
        
        return model
    
    def evaluate_model(self, model: DTHybridSimplified, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        logger.info(f"📊 Evaluating model on {dataset_name}...")
        
        start_time = time.time()
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        evaluation_results = model.evaluate(X_test, y_test)
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(y_test, y_proba)
            evaluation_results['auc_score'] = auc_score
        except:
            evaluation_results['auc_score'] = 0.5
        
        # Performance by attack type (if available)
        if 'attack_category' in X_test.columns:
            attack_performance = {}
            for attack_type in X_test['attack_category'].unique():
                mask = X_test['attack_category'] == attack_type
                if mask.sum() > 0:
                    attack_acc = (y_pred[mask] == y_test[mask]).mean()
                    attack_performance[attack_type] = attack_acc
            evaluation_results['attack_type_performance'] = attack_performance
        
        # Timing
        evaluation_time = time.time() - start_time
        evaluation_results['evaluation_time'] = evaluation_time
        evaluation_results['prediction_speed'] = len(X_test) / evaluation_time
        
        # Store results
        self.results[dataset_name] = evaluation_results
        
        logger.info(f"✅ Evaluation completed:")
        logger.info(f"  Accuracy: {evaluation_results['overall_accuracy']:.4f}")
        logger.info(f"  AUC Score: {evaluation_results['auc_score']:.4f}")
        logger.info(f"  Prediction Speed: {evaluation_results['prediction_speed']:.0f} samples/sec")
        
        return evaluation_results
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, dataset_name: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation"""
        
        logger.info(f"🔄 Cross-validating on {dataset_name} ({cv_folds} folds)...")
        
        # Create model
        model = create_dt_hybrid_simplified()
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        logger.info(f"✅ Cross-validation results:")
        logger.info(f"  Mean Accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        logger.info(f"  Range: [{cv_results['cv_min']:.4f}, {cv_results['cv_max']:.4f}]")
        
        return cv_results
    
    def compare_datasets(self):
        """Compare performance across datasets"""
        
        if len(self.results) < 2:
            logger.warning("Need at least 2 datasets to compare")
            return
        
        logger.info("📈 Comparing Dataset Performance...")
        
        print("\n" + "="*80)
        print("📊 DATASET PERFORMANCE COMPARISON")
        print("="*80)
        
        comparison_data = []
        for dataset_name, results in self.results.items():
            comparison_data.append({
                'Dataset': dataset_name,
                'Accuracy': results['overall_accuracy'],
                'AUC Score': results['auc_score'],
                'Precision': results['classification_report']['1']['precision'],
                'Recall': results['classification_report']['1']['recall'],
                'F1-Score': results['classification_report']['1']['f1-score'],
                'Prediction Speed': results['prediction_speed']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "dataset_comparison.csv", index=False)
        logger.info(f"💾 Comparison saved to {self.output_dir / 'dataset_comparison.csv'}")
    
    def save_results(self):
        """Save all results"""
        
        # Save detailed results
        import json
        
        # Convert numpy types for JSON serialization
        results_serializable = {}
        for dataset_name, results in self.results.items():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_results[key] = value.item()
                else:
                    serializable_results[key] = value
            results_serializable[dataset_name] = serializable_results
        
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save models
        for dataset_name, model in self.trained_models.items():
            model_path = self.output_dir / f"model_{dataset_name}.joblib"
            model.save_model(str(model_path))
        
        # Save preprocessors
        import joblib
        for dataset_name, preprocessor in self.preprocessors.items():
            prep_path = self.output_dir / f"preprocessor_{dataset_name}.joblib"
            joblib.dump(preprocessor, prep_path)
        
        logger.info(f"💾 All results saved to {self.output_dir}")
    
    def run_complete_pipeline(self, datasets: List[str] = ["real", "synthetic"], sample_sizes: Dict[str, int] = None):
        """Run complete training and evaluation pipeline"""
        
        logger.info("🚀 Starting Complete DT-IDS Training Pipeline...")
        
        if sample_sizes is None:
            sample_sizes = {"real": 20000, "synthetic": 15000}  # Reasonable sizes for testing
        
        pipeline_start = time.time()
        
        for dataset_type in datasets:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"PROCESSING {dataset_type.upper()} DATASET")
                logger.info(f"{'='*60}")
                
                # Load dataset
                X, y, dataset_name = self.load_dataset(dataset_type, sample_sizes.get(dataset_type))
                
                # Preprocess data
                X_train, X_test, y_train, y_test = self.preprocess_data(X, y, dataset_name)
                
                # Train model
                model = self.train_model(X_train, y_train, dataset_name)
                
                # Evaluate model
                results = self.evaluate_model(model, X_test, y_test, dataset_name)
                
                # Cross-validation on subset
                if len(X) > 5000:
                    X_cv = X.sample(n=5000, random_state=42)
                    y_cv = y[X_cv.index]
                    cv_results = self.cross_validate_model(X_cv, y_cv, f"{dataset_name}_CV")
                    results['cross_validation'] = cv_results
                
                logger.info(f"✅ {dataset_type.upper()} dataset processing completed!")
                
            except Exception as e:
                logger.error(f"❌ Error processing {dataset_type} dataset: {e}")
                continue
        
        # Compare results
        self.compare_datasets()
        
        # Save all results
        self.save_results()
        
        total_time = time.time() - pipeline_start
        
        logger.info(f"\n🎉 Complete pipeline finished in {total_time:.2f} seconds!")
        logger.info(f"📊 Results summary:")
        for dataset_name, results in self.results.items():
            logger.info(f"  {dataset_name}: {results['overall_accuracy']:.4f} accuracy")

def main():
    """Main execution function"""
    logger.info("🎯 Starting DT-IDS Training and Evaluation...")
    
    # Create trainer
    trainer = DTIDSTrainer(output_dir="results")
    
    # Run complete pipeline
    trainer.run_complete_pipeline(
        datasets=["real", "synthetic"],
        sample_sizes={"real": 10000, "synthetic": 10000}  # Manageable sizes
    )
    
    logger.info("✅ Training and evaluation completed!")

if __name__ == "__main__":
    main()