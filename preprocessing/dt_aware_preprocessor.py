"""
🔧 Digital Twin-Aware Preprocessing Framework
Advanced preprocessing system specifically designed for Digital Twin IDS datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import IsolationForest
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for DT-aware preprocessing"""
    
    # Missing value handling
    numerical_imputation: str = "knn"  # "mean", "median", "knn", "dt_aware"
    categorical_imputation: str = "most_frequent"
    
    # Outlier detection
    outlier_method: str = "isolation_forest"  # "zscore", "iqr", "isolation_forest"
    outlier_threshold: float = 0.1
    
    # Scaling
    scaling_method: str = "adaptive"  # "standard", "minmax", "robust", "adaptive"
    
    # Feature engineering
    create_temporal_features: bool = True
    create_interaction_features: bool = True
    create_dt_specific_features: bool = True
    
    # Dimensionality reduction
    apply_pca: bool = False
    pca_variance_threshold: float = 0.95
    
    # Feature selection
    feature_selection_method: str = "mutual_info"  # "mutual_info", "dt_aware", "none"
    n_features_to_select: int = 50

class DigitalTwinAwarePreprocessor:
    """
    Advanced preprocessing system for Digital Twin IDS data
    Handles DT-specific characteristics and network security features
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.is_fitted = False
        
        # Store preprocessing components
        self.scalers = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.outlier_detectors = {}
        
        # Feature categories
        self.feature_categories = self._initialize_feature_categories()
        
        # Statistics for validation
        self.preprocessing_stats = {}
        
    def _initialize_feature_categories(self) -> Dict[str, List[str]]:
        """Initialize feature categorization for DT-aware processing"""
        return {
            'network_flow': [
                'duration', 'protocol_type', 'service', 'flag',
                'src_bytes', 'dst_bytes', 'land', 'wrong_fragment'
            ],
            'traffic_statistics': [
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'same_srv_rate', 'diff_srv_rate'
            ],
            'dt_synchronization': [
                'sync_delay_ms', 'sync_accuracy_score', 'update_frequency_hz',
                'convergence_time_ms', 'sync_status_binary'
            ],
            'dt_model_performance': [
                'prediction_confidence', 'model_drift_score',
                'anomaly_detection_score', 'classification_entropy'
            ],
            'dt_system_metrics': [
                'computational_load', 'network_overhead_bytes',
                'detection_latency_ms', 'twin_health_score'
            ],
            'temporal': [
                'timestamp', 'time_of_day_encoded'
            ],
            'categorical': [
                'protocol_type', 'service', 'flag', 'attack_category',
                'attack_subcategory', 'severity_level'
            ]
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DigitalTwinAwarePreprocessor':
        """
        Fit the preprocessing pipeline to the data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            self
        """
        logger.info("🔧 Fitting Digital Twin-Aware Preprocessor...")
        
        X_copy = X.copy()
        
        # 1. Data validation and quality assessment
        self._validate_data(X_copy)
        
        # 2. Handle missing values
        X_copy = self._fit_missing_value_handlers(X_copy)
        
        # 3. Detect and handle outliers
        self._fit_outlier_detectors(X_copy)
        
        # 4. Feature engineering parameters
        self._fit_feature_engineering_params(X_copy)
        
        # 4b. Create features for fitting scalers and selectors
        X_copy = self._transform_feature_engineering(X_copy)
        
        # 5. Scaling and normalization
        self._fit_scalers(X_copy)
        
        # 6. Feature selection (after all features are created)
        if y is not None:
            self._fit_feature_selectors(X_copy, y)
        
        self.is_fitted = True
        logger.info("✅ Preprocessing pipeline fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing pipeline
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        logger.info("🔄 Applying Digital Twin-Aware Preprocessing...")
        
        X_transformed = X.copy()
        
        # 1. Handle missing values
        X_transformed = self._transform_missing_values(X_transformed)
        
        # 2. Handle outliers
        X_transformed = self._transform_outliers(X_transformed)
        
        # 3. Feature engineering
        X_transformed = self._transform_feature_engineering(X_transformed)
        
        # 4. Scaling
        X_transformed = self._transform_scaling(X_transformed)
        
        # 5. Feature selection
        X_transformed = self._transform_feature_selection(X_transformed)
        
        logger.info(f"✅ Preprocessing completed. Shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _validate_data(self, X: pd.DataFrame):
        """Validate input data quality and characteristics"""
        logger.info("📊 Validating data quality...")
        
        stats = {
            'total_samples': len(X),
            'total_features': len(X.columns),
            'missing_percentage': (X.isnull().sum().sum() / (len(X) * len(X.columns))) * 100,
            'duplicate_rows': X.duplicated().sum(),
            'constant_features': (X.nunique() == 1).sum()
        }
        
        self.preprocessing_stats['validation'] = stats
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Missing data: {stats['missing_percentage']:.2f}%")
        logger.info(f"Duplicate rows: {stats['duplicate_rows']}")
        logger.info(f"Constant features: {stats['constant_features']}")
    
    def _fit_missing_value_handlers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit missing value imputation strategies"""
        logger.info("🔍 Fitting missing value handlers...")
        
        # Separate numerical and categorical features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # DT-aware imputation for numerical features
        if self.config.numerical_imputation == "dt_aware":
            self.imputers['numerical'] = self._create_dt_aware_imputer(X, numerical_features)
        elif self.config.numerical_imputation == "knn":
            self.imputers['numerical'] = KNNImputer(n_neighbors=5)
        else:
            strategy = self.config.numerical_imputation
            self.imputers['numerical'] = SimpleImputer(strategy=strategy)
        
        # Fit numerical imputer
        if numerical_features:
            self.imputers['numerical'].fit(X[numerical_features])
        
        # Categorical imputation
        if categorical_features:
            self.imputers['categorical'] = SimpleImputer(strategy=self.config.categorical_imputation)
            self.imputers['categorical'].fit(X[categorical_features])
        
        return X
    
    def _create_dt_aware_imputer(self, X: pd.DataFrame, features: List[str]) -> Any:
        """Create Digital Twin-aware imputation strategy"""
        # For DT features, use correlation-based imputation
        # For network features, use standard KNN
        dt_features = []
        for category in ['dt_synchronization', 'dt_model_performance', 'dt_system_metrics']:
            dt_features.extend([f for f in self.feature_categories[category] if f in features])
        
        if dt_features:
            return KNNImputer(n_neighbors=3)  # Smaller neighborhood for DT features
        else:
            return KNNImputer(n_neighbors=5)
    
    def _fit_outlier_detectors(self, X: pd.DataFrame):
        """Fit outlier detection methods"""
        logger.info("🚨 Fitting outlier detectors...")
        
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.config.outlier_method == "isolation_forest":
            self.outlier_detectors['isolation_forest'] = IsolationForest(
                contamination=self.config.outlier_threshold,
                random_state=42
            )
            self.outlier_detectors['isolation_forest'].fit(X[numerical_features].fillna(0))
    
    def _fit_feature_engineering_params(self, X: pd.DataFrame):
        """Fit parameters for feature engineering (without creating features)"""
        logger.info("⚙️ Fitting feature engineering parameters...")
        
        if self.config.create_dt_specific_features:
            # Store parameters for DT-specific feature creation
            self._fit_dt_feature_engineering(X)
    
    def _fit_dt_feature_engineering(self, X: pd.DataFrame):
        """Fit Digital Twin specific feature engineering"""
        
        # Sync quality metrics
        if 'sync_delay_ms' in X.columns and 'sync_accuracy_score' in X.columns:
            self.preprocessing_stats['sync_quality_params'] = {
                'sync_delay_mean': X['sync_delay_ms'].mean(),
                'sync_delay_std': X['sync_delay_ms'].std(),
                'accuracy_threshold': X['sync_accuracy_score'].quantile(0.9)
            }
        
        # Model health indicators
        if 'prediction_confidence' in X.columns and 'model_drift_score' in X.columns:
            self.preprocessing_stats['model_health_params'] = {
                'confidence_baseline': X['prediction_confidence'].median(),
                'drift_threshold': X['model_drift_score'].quantile(0.95)
            }
    
    def _fit_scalers(self, X: pd.DataFrame):
        """Fit scaling transformations"""
        logger.info("📏 Fitting scalers...")
        
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.config.scaling_method == "adaptive":
            # Adaptive scaling based on feature characteristics
            for feature in numerical_features:
                if feature in self.feature_categories.get('dt_synchronization', []):
                    # DT sync features: MinMax scaling (bounded values)
                    self.scalers[feature] = MinMaxScaler()
                elif feature in self.feature_categories.get('dt_model_performance', []):
                    # Model performance: Standard scaling
                    self.scalers[feature] = StandardScaler()
                elif feature in ['src_bytes', 'dst_bytes', 'network_overhead_bytes']:
                    # Network bytes: Robust scaling (handle outliers)
                    self.scalers[feature] = RobustScaler()
                else:
                    # Default: Standard scaling
                    self.scalers[feature] = StandardScaler()
                
                # Fit individual scaler
                feature_data = X[[feature]].fillna(X[feature].median())
                self.scalers[feature].fit(feature_data)
        else:
            # Single scaler for all features
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            self.scalers['global'] = scaler
            scaler.fit(X[numerical_features].fillna(0))
    
    def _fit_feature_selectors(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selection methods"""
        logger.info("🎯 Fitting feature selectors...")
        
        if self.config.feature_selection_method == "mutual_info":
            # Prepare data for feature selection
            X_for_selection = X.select_dtypes(include=[np.number]).fillna(0)
            
            # Convert y to numerical if needed
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            
            self.feature_selectors['mutual_info'] = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.config.n_features_to_select, X_for_selection.shape[1])
            )
            
            self.feature_selectors['mutual_info'].fit(X_for_selection, y_encoded)
    
    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers"""
        X_transformed = X.copy()
        
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Transform numerical features
        if numerical_features and 'numerical' in self.imputers:
            X_transformed[numerical_features] = self.imputers['numerical'].transform(X_transformed[numerical_features])
        
        # Transform categorical features
        if categorical_features and 'categorical' in self.imputers:
            X_transformed[categorical_features] = self.imputers['categorical'].transform(X_transformed[categorical_features])
        
        return X_transformed
    
    def _transform_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform outliers using fitted detectors"""
        if 'isolation_forest' not in self.outlier_detectors:
            return X
        
        X_transformed = X.copy()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Detect outliers
        outlier_mask = self.outlier_detectors['isolation_forest'].predict(X_transformed[numerical_features]) == -1
        
        # Cap outliers at 99th percentile
        for feature in numerical_features:
            if feature in self.preprocessing_stats.get('outlier_caps', {}):
                cap_value = self.preprocessing_stats['outlier_caps'][feature]
                X_transformed.loc[outlier_mask, feature] = np.minimum(X_transformed.loc[outlier_mask, feature], cap_value)
        
        return X_transformed
    
    def _transform_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        X_transformed = X.copy()
        
        if self.config.create_dt_specific_features:
            X_transformed = self._create_dt_specific_features(X_transformed)
        
        if self.config.create_temporal_features:
            X_transformed = self._create_temporal_features(X_transformed)
        
        if self.config.create_interaction_features:
            X_transformed = self._create_interaction_features(X_transformed)
        
        return X_transformed
    
    def _create_dt_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create Digital Twin specific engineered features"""
        
        # Sync quality metrics
        if 'sync_delay_ms' in X.columns and 'sync_accuracy_score' in X.columns:
            X['sync_quality_score'] = X['sync_accuracy_score'] / (1 + X['sync_delay_ms'] / 100)
            
        if 'sync_accuracy_score' in X.columns and 'update_frequency_hz' in X.columns:
            X['sync_efficiency'] = X['sync_accuracy_score'] * X['update_frequency_hz']
        
        # Model health indicators
        if 'prediction_confidence' in X.columns and 'model_drift_score' in X.columns:
            X['model_reliability'] = X['prediction_confidence'] * (1 - X['model_drift_score'])
            X['prediction_stability'] = X['prediction_confidence'] / (1 + X['classification_entropy'])
        
        # System performance ratios
        if 'twin_health_score' in X.columns and 'computational_load' in X.columns:
            X['efficiency_ratio'] = X['twin_health_score'] / (X['computational_load'] + 0.001)
            X['system_balance'] = X['twin_health_score'] - X['computational_load']
        
        return X
    
    def _create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp"""
        if 'timestamp' in X.columns:
            X['timestamp'] = pd.to_datetime(X['timestamp'])
            X['hour'] = X['timestamp'].dt.hour
            X['day_of_week'] = X['timestamp'].dt.dayofweek
            X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
            X['is_business_hours'] = ((X['hour'] >= 9) & (X['hour'] <= 17)).astype(int)
        
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        # Network-DT interactions
        if 'src_bytes' in X.columns and 'sync_delay_ms' in X.columns:
            X['traffic_sync_interaction'] = X['src_bytes'] * X['sync_delay_ms']
        
        if 'count' in X.columns and 'computational_load' in X.columns:
            X['load_traffic_interaction'] = X['count'] * X['computational_load']
        
        return X
    
    def _transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformations"""
        X_transformed = X.copy()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.config.scaling_method == "adaptive":
            # Apply individual scalers
            for feature in numerical_features:
                if feature in self.scalers:
                    X_transformed[[feature]] = self.scalers[feature].transform(X_transformed[[feature]])
        else:
            # Apply global scaler
            if 'global' in self.scalers:
                X_transformed[numerical_features] = self.scalers['global'].transform(X_transformed[numerical_features])
        
        return X_transformed
    
    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if self.config.feature_selection_method == "none" or 'mutual_info' not in self.feature_selectors:
            return X
        
        X_numerical = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])
        
        # Apply feature selection to numerical features
        X_selected = self.feature_selectors['mutual_info'].transform(X_numerical)
        
        # Get selected feature names
        selected_features = X_numerical.columns[self.feature_selectors['mutual_info'].get_support()].tolist()
        
        # Combine selected numerical with categorical
        result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        for col in X_categorical.columns:
            result[col] = X_categorical[col]
        
        return result
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Generate comprehensive preprocessing report"""
        return {
            'config': self.config.__dict__,
            'statistics': self.preprocessing_stats,
            'feature_categories': self.feature_categories,
            'fitted_components': {
                'scalers': list(self.scalers.keys()),
                'imputers': list(self.imputers.keys()),
                'feature_selectors': list(self.feature_selectors.keys()),
                'outlier_detectors': list(self.outlier_detectors.keys())
            }
        }

# Example usage function
def create_dt_preprocessor(config_params: Dict = None) -> DigitalTwinAwarePreprocessor:
    """Create a configured DT-aware preprocessor"""
    
    config = PreprocessingConfig()
    if config_params:
        for key, value in config_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return DigitalTwinAwarePreprocessor(config)

if __name__ == "__main__":
    # Example usage
    logger.info("🔧 Digital Twin-Aware Preprocessor initialized!")
    
    # Create sample configuration
    sample_config = {
        'numerical_imputation': 'dt_aware',
        'scaling_method': 'adaptive',
        'create_dt_specific_features': True,
        'feature_selection_method': 'mutual_info',
        'n_features_to_select': 50
    }
    
    preprocessor = create_dt_preprocessor(sample_config)
    logger.info("✅ Preprocessor ready for fitting and transformation!")