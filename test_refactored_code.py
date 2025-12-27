#!/usr/bin/env python3
"""
Test script to verify the refactored code structure works correctly.
This script tests all imports and basic functionality.

To run this script:
    conda activate seismo
    python test_refactored_code.py
"""

import sys
import os

print("=" * 70)
print("Testing Refactored Code Structure")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print()

# Test 1: Import all modules
print("Test 1: Testing module imports...")
try:
    from file_utils import ensure_run_directory
    print("  ✓ file_utils imported successfully")
except ImportError as e:
    print(f"  ✗ file_utils import failed: {e}")
    sys.exit(1)

try:
    from event_retrieval import get_recent_quakes, get_event_details, get_mechanism
    print("  ✓ event_retrieval imported successfully")
except ImportError as e:
    print(f"  ✗ event_retrieval import failed: {e}")
    sys.exit(1)

try:
    from station_operations import get_nearest_stations, get_waveforms_and_pga
    print("  ✓ station_operations imported successfully")
except ImportError as e:
    print(f"  ✗ station_operations import failed: {e}")
    sys.exit(1)

try:
    from bbp_generation import (select_1d_velocity_model, calculate_fault_dims,
                                generate_bbp_src, generate_bbp_stl, generate_bbp_input_text)
    print("  ✓ bbp_generation imported successfully")
except ImportError as e:
    print(f"  ✗ bbp_generation import failed: {e}")
    sys.exit(1)

try:
    from bbp_execution import run_bbp_simulation, get_simulated_pgas
    print("  ✓ bbp_execution imported successfully")
except ImportError as e:
    print(f"  ✗ bbp_execution import failed: {e}")
    sys.exit(1)

try:
    from visualization import generate_display_map, compare_results
    print("  ✓ visualization imported successfully")
except ImportError as e:
    print(f"  ✗ visualization import failed: {e}")
    sys.exit(1)

print()

# Test 2: Verify backward compatibility through tools.py
print("Test 2: Testing backward compatibility (tools.py)...")
try:
    from tools import (get_recent_quakes, get_mechanism, generate_bbp_src,
                      get_nearest_stations, get_waveforms_and_pga,
                      generate_bbp_stl, generate_bbp_input_text,
                      run_bbp_simulation, get_simulated_pgas,
                      generate_display_map, compare_results, ensure_run_directory)
    print("  ✓ All functions available through tools.py (backward compatibility)")
except ImportError as e:
    print(f"  ✗ tools.py backward compatibility failed: {e}")
    sys.exit(1)

print()

# Test 3: Test display_monitor_live.py imports
print("Test 3: Testing display_monitor_live.py imports...")
try:
    # Read the file and check imports
    with open('display_monitor_live.py', 'r') as f:
        content = f.read()
        if 'from file_utils import' in content:
            print("  ✓ display_monitor_live.py uses new module structure")
        else:
            print("  ⚠ display_monitor_live.py may still use old imports")
except Exception as e:
    print(f"  ✗ Could not verify display_monitor_live.py: {e}")

print()

# Test 4: Test basic function signatures
print("Test 4: Testing function availability...")
functions_to_test = [
    ('ensure_run_directory', ensure_run_directory),
    ('get_recent_quakes', get_recent_quakes),
    ('get_mechanism', get_mechanism),
    ('get_nearest_stations', get_nearest_stations),
    ('calculate_fault_dims', calculate_fault_dims),
    ('select_1d_velocity_model', select_1d_velocity_model),
]

for func_name, func_obj in functions_to_test:
    if callable(func_obj):
        print(f"  ✓ {func_name} is callable")
    else:
        print(f"  ✗ {func_name} is not callable")

print()

# Test 5: Test a simple utility function
print("Test 5: Testing ensure_run_directory function...")
try:
    test_dir = ensure_run_directory("test_event_12345", base_dir="test_outdata")
    if os.path.exists(test_dir):
        print(f"  ✓ ensure_run_directory created directory: {test_dir}")
        # Cleanup
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
            print(f"  ✓ Cleaned up test directory")
    else:
        print(f"  ✗ ensure_run_directory did not create directory")
except Exception as e:
    print(f"  ✗ ensure_run_directory test failed: {e}")

print()

# Test 6: Test calculate_fault_dims
print("Test 6: Testing calculate_fault_dims function...")
try:
    length, width = calculate_fault_dims(4.5)
    print(f"  ✓ calculate_fault_dims(4.5) returned: length={length:.2f}km, width={width:.2f}km")
except Exception as e:
    print(f"  ✗ calculate_fault_dims test failed: {e}")

print()

# Test 7: Check module file existence
print("Test 7: Verifying all module files exist...")
module_files = [
    'file_utils.py',
    'event_retrieval.py',
    'station_operations.py',
    'bbp_generation.py',
    'bbp_execution.py',
    'visualization.py',
    'tools.py',
    'display_monitor_live.py'
]

for module_file in module_files:
    if os.path.exists(module_file):
        print(f"  ✓ {module_file} exists")
    else:
        print(f"  ✗ {module_file} missing!")

print()

# Test 8: Verify we're using the right Python environment
print("Test 8: Verifying Python environment...")
if 'seismo' in sys.executable or 'miniconda3' in sys.executable:
    print(f"  ✓ Using conda environment: {sys.executable}")
else:
    print(f"  ⚠ Not using conda environment: {sys.executable}")
    print("  ⚠ Consider running: conda activate seismo")

print()
print("=" * 70)
print("All tests completed!")
print("=" * 70)
print()
print("To verify you're using the updated code:")
print("  1. Make sure you're in the seismo conda environment")
print("  2. Check function locations:")
print("     python -c 'from tools import get_recent_quakes; print(get_recent_quakes.__module__)'")
print("  3. This should show: 'event_retrieval' (not 'tools')")
print()
