# SeismoAgent: Autonomous Earthquake Simulation & Validation

**SeismoAgent** is an AI-driven seismological workflow automation tool designed to bridge the gap between observed earthquake data and physics-based ground motion simulations.

This Python-based agent autonomously detects significant earthquakes in California, retrieves real-time sensor data (seismograms), calculates observed ground motions (PGA), and then configures and executes a full-physics simulation using the **SCEC Broadband Platform (BBP)** via Docker.

The goal is to automate the "validation" loop: rapidly comparing how well our physics models (simulation) match reality (observation) immediately after a significant event occurs.

---

## Table of Contents
1.  [Features](#features)
2.  [Architecture](#architecture)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
5.  [Usage](#usage)
6.  [Workflow Details](#workflow-details)
7.  [Project Structure](#project-structure)
8.  [License](#license)

---

## Features

* **Autonomous Event Retrieval:** Queries USGS ComCat for event details (Hypocenter, Magnitude, Time) and Focal Mechanisms.
* **Smart Station Scouting:** Uses an "Inventory-First" approach with the SCEDC/IRIS FDSN clients to identify stations with valid digital data, avoiding "No Data" errors common with older events.
* **Physics-Based Data Processing:** Downloads waveforms, removes instrument response (converts Counts to $m/s^2$), and calculates observed Peak Ground Acceleration (PGA).
* **Simulation Prep:** Automatically generates SCEC BBP input files:
    * **Source File (`.src`):** Derives fault dimensions ($L, W$) from Magnitude using Wells & Coppersmith (1994) and formats geometry for the BBP.
    * **Station List (`.stl`):** Formats observed station metadata for the simulation.
    * **Input Script (`bbp_input.txt`):** Automates the BBP's interactive menu system by generating a precise keystroke sequence.
* **Hybrid Execution:** Orchestrates a Docker container to run the computationally intensive Broadband Platform simulations (Fortran/C++) directly on a local machine by piping input commands into the container.
* **Validation:** Parses simulation output files (`.rd50`) and generates a comparison table of **Simulated vs. Observed** ground motions.

---

## Architecture

SeismoAgent acts as a central orchestrator. It runs locally (Python) and interfaces with external Data Centers (Cloud) and a local computational engine (Docker).

1.  **The Brain (Python):** Manages logic, fetches data, generates configuration files.
2.  **The Librarian (USGS/SCEDC):** Provides earthquake catalogs and waveform data.
3.  **The Engine (Docker + SCEC BBP):** A containerized supercomputer code that runs the rupture generation (Genslip) and wave propagation (Graves & Pitarka) physics.

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
    # Recommended: Create a virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install libcomcat obspy docker numpy
    ```

3.  **Pull the Simulation Engine:**
    This step downloads the physics codes and velocity models (approx. 24GB).
    ```bash
    docker pull sceccode/bbp_22_4:latest
    ```

---

## Usage

**Run the Agent:**
```bash
python findPGAs.py