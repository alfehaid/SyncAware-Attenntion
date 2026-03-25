"""
🧪 Test DT-Aware Preprocessor
Comprehensive testing for the Digital Twin-aware preprocessing pipeline
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
import time

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_preprocessing():
    """Test basic preprocessing functionality"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing basic preprocessing...")
        
        # Create test data
        n_samples = 1000
        data = {
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.randn(n_samples),
        }
        
        # Add some missing values
        data['feature_1'][::10] = np.nan
        data['feature_2'][::15] = np.nan
        
        X = pd.DataFrame(data)
        y = np.random.randint(0, 2, n_samples)
        
        # Create preprocessor
        config = PreprocessingConfig()
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        # Test fit_transform
        X_processed = preprocessor.fit_transform(X, y)
        
        # Verify results
        assert X_processed.shape[0] == n_samples, "Sample count mismatch"
        assert not X_processed.isnull().any().any(), "Missing values not handled"
        
        logger.info("✅ Basic preprocessing test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic preprocessing test failed: {e}")
        return False

def test_feature_engineering():
    """Test Digital Twin feature engineering"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing feature engineering...")
        
        # Create realistic network data
        n_samples = 500
        data = {
            'duration': np.random.exponential(2.0, n_samples),
            'src_bytes': np.random.lognormal(5, 2, n_samples),
            'dst_bytes': np.random.lognormal(4, 2, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(8, n_samples),
        }
        
        X = pd.DataFrame(data)
        y = np.random.randint(0, 2, n_samples)
        
        # Create preprocessor with feature engineering enabled
        config = PreprocessingConfig(
            create_dt_specific_features=True,
            create_temporal_features=True,
            create_interaction_features=True
        )
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        # Test feature engineering
        X_processed = preprocessor.fit_transform(X, y)
        
        # Verify new features were created
        original_features = len(X.columns)
        processed_features = len(X_processed.columns)
        
        assert processed_features > original_features, "No new features created"
        
        logger.info(f"✅ Feature engineering test passed!")
        logger.info(f"   Original features: {original_features}")
        logger.info(f"   Processed features: {processed_features}")
        logger.info(f"   New features: {processed_features - original_features}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False

def test_scaling_methods():
    """Test different scaling methods"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing scaling methods...")
        
        # Create test data with different scales
        n_samples = 300
        data = {
            'small_values': np.random.uniform(0, 1, n_samples),
            'medium_values': np.random.uniform(10, 100, n_samples),
            'large_values': np.random.uniform(1000, 10000, n_samples),
        }
        
        X = pd.DataFrame(data)
        y = np.random.randint(0, 2, n_samples)
        
        scaling_methods = ['standard', 'minmax', 'robust', 'adaptive']
        
        for method in scaling_methods:
            config = PreprocessingConfig(scaling_method=method)
            preprocessor = DigitalTwinAwarePreprocessor(config)
            
            X_scaled = preprocessor.fit_transform(X, y)
            
            # Verify scaling worked
            assert not X_scaled.isnull().any().any(), f"NaN values in {method} scaling"
            
            logger.info(f"   ✅ {method} scaling: OK")
        
        logger.info("✅ All scaling methods test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Scaling methods test failed: {e}")
        return False

def test_feature_selection():
    """Test feature selection methods"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing feature selection...")
        
        # Create data with many features
        n_samples = 400
        n_features = 50
        
        # Create some informative and some random features
        X_informative = np.random.randn(n_samples, 10)
        X_random = np.random.randn(n_samples, n_features - 10)
        X_array = np.hstack([X_informative, X_random])
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(X_array, columns=feature_names)
        
        # Create target correlated with informative features
        y = (X_informative.sum(axis=1) > 0).astype(int)
        
        # Test feature selection
        config = PreprocessingConfig(
            feature_selection_method='mutual_info',
            n_features_to_select=20
        )
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        X_selected = preprocessor.fit_transform(X, y)
        
        # Verify feature selection
        assert X_selected.shape[1] <= 20, "Too many features selected"
        assert X_selected.shape[0] == n_samples, "Sample count changed"
        
        logger.info(f"✅ Feature selection test passed!")
        logger.info(f"   Original features: {n_features}")
        logger.info(f"   Selected features: {X_selected.shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature selection test failed: {e}")
        return False

def test_performance():
    """Test preprocessing performance"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing preprocessing performance...")
        
        # Create large dataset
        n_samples = 10000
        n_features = 100
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Add some categorical features
        for i in range(5):
            X[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        
        # Test performance
        config = PreprocessingConfig()
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        start_time = time.time()
        X_processed = preprocessor.fit_transform(X, y)
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        samples_per_second = n_samples / processing_time
        
        logger.info(f"✅ Performance test passed!")
        logger.info(f"   Samples: {n_samples:,}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Speed: {samples_per_second:.0f} samples/second")
        
        # Performance should be reasonable
        assert samples_per_second > 1000, "Processing too slow"
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def run_all_tests():
    """Run all preprocessor tests"""
    logger.info("🚀 Starting Preprocessor Tests...")
    
    tests = [
        ("Basic Preprocessing", test_basic_preprocessing),
        ("Feature Engineering", test_feature_engineering),
        ("Scaling Methods", test_scaling_methods),
        ("Feature Selection", test_feature_selection),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 PREPROCESSOR TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All preprocessor tests passed successfully!")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)