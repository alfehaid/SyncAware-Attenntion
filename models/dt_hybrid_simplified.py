"""
🧠 Simplified DT-Hybrid Model using Scikit-Learn
A simplified version of the hybrid model using classical ML techniques
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DTHybridSimplified:
    """
    Simplified Digital Twin Hybrid Model
    Combines multiple classifiers to simulate hybrid deep learning approach
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.is_fitted = False
        self.feature_importance = {}
        self.training_history = []
        
        logger.info("🧠 Initializing Simplified DT-Hybrid Model...")
        self._initialize_models()
    
    def _default_config(self) -> Dict:
        """Default configuration for the model"""
        return {
            'use_random_forest': True,
            'use_gradient_boosting': True,
            'use_mlp': True,
            'use_svm': False,  # Slower for large datasets
            'use_logistic': True,
            'ensemble_method': 'voting',  # 'voting' or 'stacking'
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10
        }
    
    def _initialize_models(self):
        """Initialize individual models"""
        
        # Network Features Branch (Random Forest)
        if self.config['use_random_forest']:
            self.models['network_rf'] = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        # DT Features Branch (Gradient Boosting)
        if self.config['use_gradient_boosting']:
            self.models['dt_gb'] = GradientBoostingClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state']
            )
        
        # System Features Branch (MLP - Neural Network)
        if self.config['use_mlp']:
            self.models['system_mlp'] = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=self.config['random_state']
            )
        
        # Traditional ML models
        if self.config['use_logistic']:
            self.models['logistic'] = LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000
            )
        
        if self.config['use_svm']:
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config['random_state']
            )
        
        # Meta-classifier for ensemble
        self.meta_classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.config['random_state']
        )
        
        logger.info(f"✅ Initialized {len(self.models)} base models")
    
    def _split_features(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Split features into different branches"""
        
        # Identify feature types based on names
        network_features = [col for col in X.columns if any(
            keyword in col.lower() for keyword in ['duration', 'bytes', 'count', 'rate', 'protocol', 'service', 'flag']
        )]
        
        dt_features = [col for col in X.columns if any(
            keyword in col.lower() for keyword in ['sync', 'twin', 'model', 'drift', 'confidence', 'accuracy']
        )]
        
        system_features = [col for col in X.columns if any(
            keyword in col.lower() for keyword in ['load', 'latency', 'health', 'overhead', 'device']
        )]
        
        # If categorization fails, split by position
        if not network_features:
            total_features = len(X.columns)
            network_features = X.columns[:total_features//3].tolist()
            dt_features = X.columns[total_features//3:2*total_features//3].tolist()
            system_features = X.columns[2*total_features//3:].tolist()
        
        return {
            'network': X[network_features].values,
            'dt': X[dt_features].values if dt_features else X.values[:, :min(20, X.shape[1])],
            'system': X[system_features].values if system_features else X.values[:, -min(15, X.shape[1]):],
            'all': X.values
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DTHybridSimplified':
        """Train the hybrid model"""
        logger.info(f"🔄 Training DT-Hybrid model on {len(X)} samples...")
        
        start_time = time.time()
        
        # Split features
        feature_splits = self._split_features(X)
        
        # Train individual models
        predictions = {}
        
        # Network features with Random Forest
        if 'network_rf' in self.models:
            self.models['network_rf'].fit(feature_splits['network'], y)
            predictions['network_rf'] = self.models['network_rf'].predict_proba(feature_splits['network'])[:, 1]
            self.feature_importance['network_rf'] = self.models['network_rf'].feature_importances_
        
        # DT features with Gradient Boosting
        if 'dt_gb' in self.models:
            self.models['dt_gb'].fit(feature_splits['dt'], y)
            predictions['dt_gb'] = self.models['dt_gb'].predict_proba(feature_splits['dt'])[:, 1]
            self.feature_importance['dt_gb'] = self.models['dt_gb'].feature_importances_
        
        # System features with MLP
        if 'system_mlp' in self.models:
            self.models['system_mlp'].fit(feature_splits['system'], y)
            predictions['system_mlp'] = self.models['system_mlp'].predict_proba(feature_splits['system'])[:, 1]
        
        # All features with traditional models
        if 'logistic' in self.models:
            self.models['logistic'].fit(feature_splits['all'], y)
            predictions['logistic'] = self.models['logistic'].predict_proba(feature_splits['all'])[:, 1]
        
        if 'svm' in self.models:
            self.models['svm'].fit(feature_splits['all'], y)
            predictions['svm'] = self.models['svm'].predict_proba(feature_splits['all'])[:, 1]
        
        # Create meta-features for ensemble
        meta_features = np.column_stack(list(predictions.values()))
        
        # Train meta-classifier
        self.meta_classifier.fit(meta_features, y)
        
        training_time = time.time() - start_time
        self.is_fitted = True
        
        # Save training history
        self.training_history.append({
            'timestamp': time.time(),
            'samples': len(X),
            'features': X.shape[1],
            'training_time': training_time,
            'models_trained': list(self.models.keys())
        })
        
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"Models trained: {list(self.models.keys())}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Split features
        feature_splits = self._split_features(X)
        
        # Get predictions from all models
        predictions = {}
        
        if 'network_rf' in self.models:
            predictions['network_rf'] = self.models['network_rf'].predict_proba(feature_splits['network'])[:, 1]
        
        if 'dt_gb' in self.models:
            predictions['dt_gb'] = self.models['dt_gb'].predict_proba(feature_splits['dt'])[:, 1]
        
        if 'system_mlp' in self.models:
            predictions['system_mlp'] = self.models['system_mlp'].predict_proba(feature_splits['system'])[:, 1]
        
        if 'logistic' in self.models:
            predictions['logistic'] = self.models['logistic'].predict_proba(feature_splits['all'])[:, 1]
        
        if 'svm' in self.models:
            predictions['svm'] = self.models['svm'].predict_proba(feature_splits['all'])[:, 1]
        
        # Create meta-features
        meta_features = np.column_stack(list(predictions.values()))
        
        # Meta-classifier prediction
        final_predictions = self.meta_classifier.predict(meta_features)
        
        return final_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Split features
        feature_splits = self._split_features(X)
        
        # Get predictions from all models
        predictions = {}
        
        if 'network_rf' in self.models:
            predictions['network_rf'] = self.models['network_rf'].predict_proba(feature_splits['network'])[:, 1]
        
        if 'dt_gb' in self.models:
            predictions['dt_gb'] = self.models['dt_gb'].predict_proba(feature_splits['dt'])[:, 1]
        
        if 'system_mlp' in self.models:
            predictions['system_mlp'] = self.models['system_mlp'].predict_proba(feature_splits['system'])[:, 1]
        
        if 'logistic' in self.models:
            predictions['logistic'] = self.models['logistic'].predict_proba(feature_splits['all'])[:, 1]
        
        if 'svm' in self.models:
            predictions['svm'] = self.models['svm'].predict_proba(feature_splits['all'])[:, 1]
        
        # Create meta-features
        meta_features = np.column_stack(list(predictions.values()))
        
        # Meta-classifier probability prediction
        probabilities = self.meta_classifier.predict_proba(meta_features)
        
        return probabilities
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)
        
        # Individual model performance
        feature_splits = self._split_features(X)
        individual_performance = {}
        
        for model_name, model in self.models.items():
            if model_name == 'network_rf':
                features = feature_splits['network']
            elif model_name == 'dt_gb':
                features = feature_splits['dt']
            elif model_name == 'system_mlp':
                features = feature_splits['system']
            else:
                features = feature_splits['all']
            
            individual_preds = model.predict(features)
            individual_performance[model_name] = accuracy_score(y, individual_preds)
        
        evaluation_results = {
            'overall_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'individual_model_accuracy': individual_performance,
            'ensemble_improvement': accuracy - max(individual_performance.values())
        }
        
        return evaluation_results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models"""
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'meta_classifier': self.meta_classifier,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.meta_classifier = model_data['meta_classifier']
        self.config = model_data['config']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"📂 Model loaded from: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum([
            model.n_features_in_ if hasattr(model, 'n_features_in_') else 0 
            for model in self.models.values()
        ])
        
        return {
            'model_type': 'DT-Hybrid-Simplified',
            'base_models': list(self.models.keys()),
            'ensemble_method': self.config['ensemble_method'],
            'total_parameters': total_params,
            'is_fitted': self.is_fitted,
            'training_history': len(self.training_history)
        }

def create_dt_hybrid_simplified(config: Dict = None) -> DTHybridSimplified:
    """Create simplified DT-Hybrid model"""
    
    model = DTHybridSimplified(config)
    
    logger.info("📊 Created DT-Hybrid Simplified Model:")
    model_info = model.get_model_info()
    logger.info(f"  Base models: {model_info['base_models']}")
    logger.info(f"  Ensemble method: {model_info['ensemble_method']}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    logger.info("🧪 Testing DT-Hybrid Simplified Model...")
    
    # Create model
    model = create_dt_hybrid_simplified()
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_test = np.random.randint(0, 2, n_samples)
    
    # Test training
    model.fit(X_test, y_test)
    
    # Test prediction
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Test evaluation
    results = model.evaluate(X_test, y_test)
    
    logger.info("✅ DT-Hybrid Simplified Model test completed!")
    logger.info(f"Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Individual model accuracies: {results['individual_model_accuracy']}")
    logger.info(f"Ensemble improvement: {results['ensemble_improvement']:.4f}")