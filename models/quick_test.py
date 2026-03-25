"""
🧪 Quick Test for DT-HybridNet Models
Fast testing and validation of the hybrid deep learning models
"""

import torch
import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytorch_model():
    """Test PyTorch DT-HybridNet model"""
    try:
        from models.dt_hybrid_net import create_dt_hybrid_net
        
        logger.info("🧠 Testing PyTorch DT-HybridNet...")
        
        # Create model
        model = create_dt_hybrid_net(input_features=56, fusion_method="adaptive")
        
        # Test forward pass
        batch_size = 8
        input_features = 56
        test_input = torch.randn(batch_size, input_features)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        logger.info("✅ PyTorch model test passed!")
        logger.info(f"  Binary output shape: {outputs['binary_logits'].shape}")
        logger.info(f"  Multiclass output shape: {outputs['multiclass_logits'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch model test failed: {e}")
        return False

def test_sklearn_model():
    """Test Scikit-learn DT-Hybrid model"""
    try:
        from models.dt_hybrid_simplified import create_dt_hybrid_simplified
        
        logger.info("🔬 Testing Scikit-learn DT-Hybrid...")
        
        # Create model
        model = create_dt_hybrid_simplified()
        
        # Generate test data
        n_samples = 100
        n_features = 56
        X_test = np.random.randn(n_samples, n_features)
        y_test = np.random.randint(0, 2, n_samples)
        
        # Train model
        model.fit(X_test, y_test)
        
        # Test predictions
        predictions = model.predict(X_test[:10])
        probabilities = model.predict_proba(X_test[:10])
        
        logger.info("✅ Scikit-learn model test passed!")
        logger.info(f"  Predictions shape: {predictions.shape}")
        logger.info(f"  Probabilities shape: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scikit-learn model test failed: {e}")
        return False

def test_preprocessor():
    """Test DT-Aware Preprocessor"""
    try:
        from preprocessing.dt_aware_preprocessor import DigitalTwinAwarePreprocessor, PreprocessingConfig
        
        logger.info("🔧 Testing DT-Aware Preprocessor...")
        
        # Create test data
        n_samples = 100
        n_features = 56
        
        # Generate realistic network features
        data = {
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        }
        
        # Add some categorical features
        data['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
        data['service'] = np.random.choice(['http', 'ftp', 'smtp'], n_samples)
        
        X = pd.DataFrame(data)
        y = np.random.randint(0, 2, n_samples)
        
        # Create preprocessor
        config = PreprocessingConfig()
        preprocessor = DigitalTwinAwarePreprocessor(config)
        
        # Test preprocessing
        X_processed = preprocessor.fit_transform(X, y)
        
        logger.info("✅ Preprocessor test passed!")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Processed shape: {X_processed.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessor test failed: {e}")
        return False

def run_all_tests():
    """Run all quick tests"""
    logger.info("🚀 Starting Quick Tests for DT-HybridNet...")
    
    tests = [
        ("PyTorch Model", test_pytorch_model),
        ("Scikit-learn Model", test_sklearn_model),
        ("Preprocessor", test_preprocessor)
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
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)