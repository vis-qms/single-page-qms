#!/usr/bin/env python3

# Test script for median stabilization feature

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from state import CountStabilizer

def test_median_stabilization():
    print("Testing Median Stabilization...")
    
    # Test median stabilization with various inputs
    stabilizer = CountStabilizer(method="Median", max_delta=10)
    
    test_cases = [
        ([5, 3, 7], 5, "Basic median test"),
        ([1, 10, 2], 2, "Median with outlier"),
        ([8, 8, 8], 8, "All same values"),
        ([0, 1, 2], 1, "Sequential values"),
        ([10, 5, 15], 10, "Wide range values"),
    ]
    
    print("\nTest Results:")
    print("=" * 50)
    
    for i, (inputs, expected_median, description) in enumerate(test_cases, 1):
        # Reset stabilizer for each test
        stabilizer = CountStabilizer(method="Median", max_delta=10)
        
        results = []
        for value in inputs:
            result = stabilizer.update(value)
            results.append(result)
        
        final_result = results[-1]
        passed = final_result == expected_median
        
        print(f"Test {i}: {description}")
        print(f"  Input sequence: {inputs}")
        print(f"  Expected median: {expected_median}")
        print(f"  Actual result: {final_result}")
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
        print()
        
    # Test that it uses latest value when less than 3 frames
    print("Testing with less than 3 frames:")
    stabilizer = CountStabilizer(method="Median", max_delta=10)
    
    result1 = stabilizer.update(5)
    print(f"  Frame 1 (value=5): result={result1}, expected=5, {'✓' if result1 == 5 else '✗'}")
    
    result2 = stabilizer.update(3)
    print(f"  Frame 2 (value=3): result={result2}, expected=3, {'✓' if result2 == 3 else '✗'}")
    
    result3 = stabilizer.update(7)
    print(f"  Frame 3 (value=7): result={result3}, expected=5 (median of 5,3,7), {'✓' if result3 == 5 else '✗'}")

def test_other_methods_still_work():
    print("\n" + "=" * 50)
    print("Testing that EMA and Rolling methods still work...")
    
    # Test EMA
    ema_stabilizer = CountStabilizer(method="EMA", ema_alpha=0.5, max_delta=10)
    ema_result = ema_stabilizer.update(10)
    print(f"EMA method: {'✓ PASS' if ema_result == 10 else '✗ FAIL'}")
    
    # Test Rolling
    rolling_stabilizer = CountStabilizer(method="Rolling", window_frames=3, max_delta=10)
    rolling_result = rolling_stabilizer.update(5)
    print(f"Rolling method: {'✓ PASS' if rolling_result == 5 else '✗ FAIL'}")

if __name__ == "__main__":
    test_median_stabilization()
    test_other_methods_still_work()
    print("\n" + "=" * 50)
    print("Testing complete!")
