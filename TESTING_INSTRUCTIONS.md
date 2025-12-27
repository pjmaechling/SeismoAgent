# Testing Instructions for Refactored Code

## Prerequisites

This project uses a conda environment called `seismo` which contains all the required packages:
- obspy (seismology library)
- numpy, matplotlib, pandas (data processing)
- contextily (map visualization)
- libcomcat (USGS earthquake data)

## Activating the Conda Environment

### Option 1: Command Line (Recommended)

```bash
# Navigate to the project directory
cd /Users/pjmaechling/PycharmProjects/PythonProject

# Activate the seismo conda environment
conda activate seismo

# Verify you're using the right Python
python --version
# Should show: Python 3.10.15

# Verify packages are available
python -c "import obspy; print('obspy version:', obspy.__version__)"
```

### Option 2: PyCharm Configuration

1. Open PyCharm Settings/Preferences
2. Go to: **Project → Python Interpreter**
3. Click the gear icon → **Add...**
4. Select **Conda Environment → Existing environment**
5. Navigate to: `/Users/pjmaechling/miniconda3/envs/seismo/bin/python`
6. Click **OK**

## Running the Test Script

After activating the environment:

```bash
# Run the comprehensive test
python test_refactored_code.py
```

This will verify:
- ✓ All new modules can be imported
- ✓ Backward compatibility through tools.py
- ✓ All functions are callable
- ✓ Basic functionality works

## Verifying You're Using the Updated Code

### Method 1: Check Function Module Location

```bash
conda activate seismo
python -c "from tools import get_recent_quakes; print('Module:', get_recent_quakes.__module__)"
```

**Expected output:** `event_retrieval` (not `tools`)

This confirms that `tools.py` is correctly re-exporting from the new modules.

### Method 2: Check Import Paths

```bash
conda activate seismo
python -c "import sys; from tools import get_recent_quakes; print('File:', sys.modules['event_retrieval'].__file__)"
```

This will show the actual file path being used.

### Method 3: Test Individual Modules

```bash
conda activate seismo

# Test each module directly
python -c "from event_retrieval import get_recent_quakes; print('✓ event_retrieval works')"
python -c "from station_operations import get_nearest_stations; print('✓ station_operations works')"
python -c "from bbp_generation import generate_bbp_src; print('✓ bbp_generation works')"
python -c "from bbp_execution import run_bbp_simulation; print('✓ bbp_execution works')"
python -c "from visualization import generate_display_map; print('✓ visualization works')"
python -c "from file_utils import ensure_run_directory; print('✓ file_utils works')"
```

## Running the Main Application

```bash
conda activate seismo
python display_monitor_live.py
```

## Quick Verification Checklist

- [ ] `conda activate seismo` works without errors
- [ ] `python --version` shows Python 3.10.15
- [ ] `python test_refactored_code.py` passes all tests
- [ ] `python -c "from tools import get_recent_quakes; print(get_recent_quakes.__module__)"` shows `event_retrieval`
- [ ] All module files exist in the project directory

## Troubleshooting

### Issue: "ModuleNotFoundError" when importing

**Solution:** Make sure you've activated the seismo environment:
```bash
conda activate seismo
```

### Issue: "Command not found: conda"

**Solution:** Initialize conda in your shell:
```bash
# For bash/zsh
source ~/miniconda3/etc/profile.d/conda.sh

# Then activate
conda activate seismo
```

### Issue: Wrong Python version

**Solution:** Verify you're using the seismo environment:
```bash
which python
# Should show: /Users/pjmaechling/miniconda3/envs/seismo/bin/python
```

### Issue: Packages missing

**Solution:** Install missing packages in the seismo environment:
```bash
conda activate seismo
conda install -c conda-forge obspy numpy matplotlib pandas contextily libcomcat
```

## Environment Information

- **Environment name:** seismo
- **Python version:** 3.10.15
- **Location:** `/Users/pjmaechling/miniconda3/envs/seismo`
- **Key packages:**
  - obspy 1.4.2
  - numpy 2.2.6
  - matplotlib 3.10.8
  - pandas 2.2.3
  - contextily 1.7.0
  - libcomcat 2.0.15

