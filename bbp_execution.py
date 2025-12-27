"""
BBP simulation execution and result parsing functions.
"""
import os
import subprocess
import glob
import time


def run_bbp_simulation(input_text_filename):
    """
    Runs the SCEC BBP Docker container.
    FIXED: 'cat' uses Host Path, Docker Volume uses Project Root.
    """
    print(f"\n--- TOOL: Launching Docker Simulation ---")

    # 1. Determine Paths
    try:
        # Host Path (Absolute) to Project Root
        # We assume the script is running from the project root
        host_project_root_abs = os.path.abspath(os.getcwd())

        # Container Mount Point
        container_mount_point = "/app/target"

        # Determine Event ID (for verification later)
        event_id_part = os.path.basename(input_text_filename).replace("bbp_input_", "").replace(".txt", "")

        # Directory where we expect results on the Host
        run_dir = os.path.join("outdata", f"Event_{event_id_part}")

    except Exception as e:
        print(f"    Path determination error: {e}")
        return False

    # 2. Construct Docker Command
    # -v maps the WHOLE project to /app/target
    # This ensures that /app/target/outdata/... matches host outdata/...
    docker_cmd = (
        f"docker run --rm -i "
        f"-v {host_project_root_abs}:{container_mount_point} "
        f"--ulimit stack=-1 "
        f"--platform linux/amd64 "
        f"sceccode/bbp_22_4:latest "
        f"/home/scecuser/bbp/bbp/comps/run_bbp.py"
    )

    # 3. Execute
    # CRITICAL FIX: 'cat' runs on the Host, so it needs the HOST path (input_text_filename)
    # The output is piped INTO Docker, which is what we want.
    full_cmd = f"cat {input_text_filename} | {docker_cmd}"

    # Debug print to confirm what we are running
    print(f"    Host Input File: {input_text_filename}")

    try:
        # Run process
        process = subprocess.run(full_cmd, shell=True, check=False, text=True, capture_output=True)
        time.sleep(1.0)  # Brief sync pause

        # 4. Verify Success
        if run_dir and os.path.exists(run_dir):
            # Look for ANY .rd50 file in the BBP numeric subfolders
            rd50_files = glob.glob(f"{run_dir}/*/*.rd50")

            if rd50_files:
                print(f"    Docker finished. Found {len(rd50_files)} result files.")
                return True
            else:
                print(f"    Docker finished, but NO result files (.rd50) found.")
                # Print stderr to see why BBP failed (if it wasn't the 'cat' error)
                if process.stderr:
                    print(f"    Docker Stderr: {process.stderr[:500]}...")
                return False

        return process.returncode == 0

    except Exception as e:
        print(f"    Docker Execution Exception: {e}")
        return False


def get_simulated_pgas(pga_data, event_id, output_dir="outdata"):
    """
    Parses BBP output (.rd50 files) for simulated PGA values.
    """
    print(f"\n--- TOOL: Parsing Simulated PGA Results for {event_id} ---")

    # 1. Locate the correct event directory
    # FIX: Ensure we don't double-nest (e.g. outdata/Event_X/Event_X)
    # We assume output_dir is relative ("outdata") or absolute.

    # If output_dir is relative, this makes it absolute: /path/to/project/outdata
    abs_out_dir = os.path.abspath(output_dir)

    # Construct: /path/to/project/outdata/Event_ci38443303
    target_event_dir = os.path.join(abs_out_dir, f"Event_{event_id}")
    target_event_dir = "outdata"
    if not os.path.isdir(target_event_dir):
        print(f"    Event directory not found: {target_event_dir}")
        return []

    # 2. Find the BBP internal run folder (numeric folder)
    numbered_run_dirs = [d for d in glob.glob(f"{target_event_dir}/*")
                         if os.path.isdir(d) and os.path.basename(d).isdigit()]

    if not numbered_run_dirs:
        print(f"    Could not find BBP numbered output job folders inside {target_event_dir}.")
        return []

    # Use the latest numbered folder
    latest_run_dir = max(numbered_run_dirs, key=os.path.getmtime)
    print(f"    Reading results from BBP job folder: {os.path.basename(latest_run_dir)}")

    # 3. Parse .rd50 files
    rd50_files = glob.glob(f"{latest_run_dir}/*.rd50")
    sim_pgas_by_station = {}

    if not rd50_files:
        print("    No .rd50 result files found.")
        return []
    else:
        print(rd50_files)

    for rfile in rd50_files:
        fname = os.path.basename(rfile)
        station_id = fname.split('.')[1]

        with open(rfile, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                try:
                    # Period < 0.02 is treated as PGA
                    if float(parts[0]) < 0.02:
                        sim_pgas_by_station[station_id] = float(parts[1])
                        break
                except:
                    continue

    # 4. Align with Observed Data
    simulated_pga_list = []
    print(sim_pgas_by_station)
    print(pga_data)
    for obs in pga_data:
        sid = obs['station']
        sid = sid.split('.')[1]
        if sid in sim_pgas_by_station:
            sim_pga_g = sim_pgas_by_station[sid]
            simulated_pga_list.append({
                "station": sid,
                "latitude": obs['latitude'],
                "longitude": obs['longitude'],
                "distance_km": obs['distance_km'],
                "pga_g": sim_pga_g,
                "pga_m_s2": sim_pga_g * 9.81
            })

    print(simulated_pga_list)
    return simulated_pga_list

