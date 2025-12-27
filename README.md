# ⚛️ SeismoAgent: Autonomous Earthquake Simulation & Validation

**SeismoAgent** is an AI-driven seismological workflow automation tool designed to bridge the gap between observed earthquake data and physics-based ground motion simulations.

This Python-based agent autonomously detects significant earthquakes in California, retrieves real-time sensor data (seismograms), calculates observed ground motions (PGA), and then configures and executes a full-physics simulation using the **SCEC Broadband Platform (BBP)** via Docker.

The goal is to automate the **"validation" loop**: rapidly comparing how well our physics models (simulation) match reality (observation) immediately after a significant event occurs.

---

## Table of Contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage](#usage)
7. [Workflow Details](#workflow-details)
8. [Project Structure](#project-structure)
9. [License](#license)

---

## Features

* **Autonomous Event Retrieval:** Queries USGS ComCat for event details (Hypocenter, Magnitude, Time) and Focal Mechanisms.
* **Smart Station Scouting:** Uses an "Inventory-First" approach with the SCEDC/IRIS FDSN clients to identify stations with valid digital data, avoiding "No Data" errors common with older events.
* **Physics-Based Data Processing:** Downloads waveforms, removes instrument response (converts Counts to $m/s^2$), and calculates observed **Peak Ground Acceleration (PGA)**.
* **Simulation Prep:** Automatically generates SCEC BBP input files:
    * **Source File (`.src`):** Derives fault dimensions ($L, W$) from Magnitude using Wells & Coppersmith (1994) and formats geometry for the BBP.
    * **Station List (`.stl`):** Formats observed station metadata for the simulation.
    * **Input Script (`bbp_input.txt`):** Automates the BBP's interactive menu system by generating a precise keystroke sequence.
* **Hybrid Execution:** Orchestrates a Docker container to run the computationally intensive Broadband Platform simulations (Fortran/C++) directly on a local machine by piping input commands into the container.
* **Validation:** Parses simulation output files (`.rd50`) and generates a comparison table of **Simulated vs. Observed** ground motions.

---

## Architecture

SeismoAgent acts as a central orchestrator. It runs locally (Python) and interfaces with external Data Centers (Cloud) and a local computational engine (Docker). The architecture is defined by a shared **Tool Library** and three distinct **Execution Agents**. 

### 1. The Core Tool Library (Modular Structure) 🛠️

The codebase has been refactored into a modular structure for better organization and maintainability:

| Module | Purpose | Key Functions |
| :--- | :--- | :--- |
| `file_utils.py` | Directory management | `ensure_run_directory` |
| `event_retrieval.py` | USGS/ComCat data retrieval | `get_recent_quakes`, `get_event_details`, `get_mechanism` |
| `station_operations.py` | Station finding and waveform processing | `get_nearest_stations`, `get_waveforms_and_pga` |
| `bbp_generation.py` | BBP file generation | `generate_bbp_src`, `generate_bbp_stl`, `generate_bbp_input_text` |
| `bbp_execution.py` | BBP simulation execution | `run_bbp_simulation`, `get_simulated_pgas` |
| `visualization.py` | Map generation and result comparison | `generate_display_map`, `compare_results` |
| `tools.py` | Backward compatibility layer | Re-exports all functions for existing code |

**Note:** The `tools.py` file maintains backward compatibility by re-exporting all functions, so existing code continues to work without changes. New code should import directly from the specific modules.

### 2. Execution Agents 🤖

The system operates across three distinct modes, each defined by its own primary execution file:

| Agent File | Primary Goal | Mode | Key Outputs |
| :--- | :--- | :--- | :--- |
| **`findPGAs.py`** | **Validation & Debugging.** Runs a full simulation pipeline for a **single, known event** for verification and testing purposes. | One-Off Test | `comparison_map.png` |
| **`night_watch.py`** | **Automated Monitoring.** Runs in an infinite loop, querying USGS for **new M3.0+ events** across California and executing the full pipeline. | Continuous Loop | `comparison_map_[ID].png` |
| **`display_monitor.py`**| **Visualization & Comparison.** Generates **dual maps** (Observed vs. Simulated) for a target event to allow for direct visual residual analysis. | Test / Display | `display_obs_map_pga.png` and `display_sim_map_pga.png` |

---

### External Components

1.  **The Librarian (USGS/SCEDC):** Provides earthquake catalogs (ComCat) and waveform data (FDSN web services).
2.  **The Engine (Docker + SCEC BBP):** A containerized code environment that runs the rupture generation (Genslip) and wave propagation (Graves & Pitarka) physics.

---

## Prerequisites

* **Operating System:** macOS (Apple Silicon or Intel) or Linux.
* **Python:** 3.10 or newer.
* **Docker:** Docker Desktop must be installed and running.
* **Disk Space:** ~30GB (The SCEC BBP Docker image is large).

---

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/SeismoAgent.git](https://github.com/yourusername/SeismoAgent.git)
    cd SeismoAgent
    ```

2.  **Set up the Python Environment:**
    ```bash
    # 1. Create and activate
    conda create -n seismo python=3.10
    conda activate seismo
    
    # 2. Install dependencies
    conda install -c conda-forge numpy obspy matplotlib contextily libcomcat docker
    ```

3.  **Pull the Simulation Engine:**
    This step downloads the physics codes and velocity models (approx. 24GB).
    ```bash
    docker pull sceccode/bbp_22_4:latest
    ```

---

## Quick Start

### 1. Activate the Conda Environment

This project uses a conda environment called `seismo` which contains all required packages:

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

### 2. Test the Refactored Code

Run the comprehensive test script to verify all modules work correctly:

```bash
python test_refactored_code.py
```

This will verify:
- ✓ All new modules can be imported
- ✓ Backward compatibility through tools.py
- ✓ All functions are callable
- ✓ Basic functionality works

### 3. Verify You're Using the Updated Code

To confirm you're using the refactored modular structure:

```bash
python -c "from tools import get_recent_quakes; print('Module:', get_recent_quakes.__module__)"
```

**Expected output:** `event_retrieval` (not `tools`)

This confirms that `tools.py` is correctly re-exporting from the new modules.

### 4. Run the Application

```bash
# Run the live monitoring agent
python display_monitor_live.py
```

---

## Usage

**Run the Validation Agent (One-Off Test):**

```bash
python findPGAs.py
python night_watch.py
python display_monitor.py
```

## Workflow Details

1. Event Discovery: Agent queries USGS for M > 3.0 events in the last 24 hours.
1. Source Modeling: Derives L,W,Ztop​ and generates the .src file.
1. Data Acquisition: Downloads waveforms, calculates observed PGA, and generates the station list for simulation (.stl).
1. Simulation Execution: The run_bbp_simulation function mounts the project folder to the Docker container and executes the pre-generated bbp_input.txt script.
1. Validation & Display: Parses the simulation output (.rd50 files) and generates visual comparisons.

## Project Structure

## License

BSD-3 License