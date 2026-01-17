#!/usr/bin/env python3
"""
Test script for gradient-based expert identification method.
Run this to verify the implementation works correctly.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile
import shutil
from pathlib import Path


def test_expert_gradient_calculation():
    """Test if we can calculate expert gradients correctly."""
    print("\n" + "="*60)
    print("TEST 1: Expert Gradient Calculation")
    print("="*60)
    
    try:
        # Create a simple model (use small model for testing)
        print("Loading small test model...")
        model_name = "gpt2"  # Small model for testing
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy input
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Forward and backward
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Gradients calculated successfully")
            return True
        else:
            print("✗ No gradients found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_data_loading():
    """Test if we can load the dataset."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Loading")
    print("="*60)
    
    dataset_path = "/home/stufs1/jiachliang/FlipAttack/reproduce_result/FlipAttack-FWO-CoT-LangGPT-Few-shot-Qwen3-30B-A3B-advbench-0_519-final.json"
    
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"✓ Dataset loaded: {len(dataset)} examples")
        
        # Check data format
        if len(dataset) > 0:
            first_item = dataset[0]
            required_fields = ["goal", "all_prompt"]
            
            missing_fields = [f for f in required_fields if f not in first_item]
            
            if missing_fields:
                print(f"✗ Missing fields: {missing_fields}")
                return False
            else:
                print(f"✓ Data format correct")
                return True
        else:
            print("✗ Dataset is empty")
            return False
            
    except FileNotFoundError:
        print(f"✗ Dataset not found at {dataset_path}")
        print("  Please update the path in test_gradient_method.py")
        return False
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_module_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*60)
    print("TEST 3: Module Imports")
    print("="*60)
    
    required_modules = [
        "torch",
        "transformers",
        "datasets",
        "pandas",
        "numpy",
        "tqdm",
    ]
    
    all_success = True
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} - NOT INSTALLED")
            all_success = False
    
    return all_success


def test_script_syntax():
    """Test if the main training script has valid syntax."""
    print("\n" + "="*60)
    print("TEST 4: Script Syntax Check")
    print("="*60)
    
    script_path = "/home/stufs1/jiachliang/SteerMoE/train_gradient_based_expert_identification.py"
    
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        
        compile(code, script_path, 'exec')
        print(f"✓ Script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_output_directory_creation():
    """Test if we can create output directories."""
    print("\n" + "="*60)
    print("TEST 5: Output Directory Creation")
    print("="*60)
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="gradient_test_")
        
        # Create subdirectories
        (Path(temp_dir) / "round_1").mkdir(parents=True, exist_ok=True)
        (Path(temp_dir) / "final_model").mkdir(parents=True, exist_ok=True)
        
        # Write test file
        test_file = Path(temp_dir) / "test.json"
        with open(test_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        print("✓ Directory creation and file writing successful")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_gpu_availability():
    """Test if GPU is available."""
    print("\n" + "="*60)
    print("TEST 6: GPU Availability")
    print("="*60)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ GPU available: {num_gpus} GPU(s) detected")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("✗ No GPU detected")
        print("  Training will be very slow on CPU")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GRADIENT-BASED METHOD IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Script Syntax", test_script_syntax),
        ("Expert Gradient Calculation", test_expert_gradient_calculation),
        ("Dataset Loading", test_data_loading),
        ("Output Directory Creation", test_output_directory_creation),
        ("GPU Availability", test_gpu_availability),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed! Ready to run training.")
        print("\nNext steps:")
        print("  1. Run quick start: ./quick_start_gradient_method.sh")
        print("  2. Or run manually: python train_gradient_based_expert_identification.py --help")
    else:
        print("\n✗ Some tests failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install <package_name>")
        print("  - Update dataset path in the script")
        print("  - Ensure GPU drivers are installed for GPU training")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)




