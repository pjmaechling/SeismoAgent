import time
import datetime  # <--- THIS WAS MISSING
import os
import subprocess
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
# ... (rest of the file remains the same)
from obspy import UTCDateTime
from obspy.clients.fdsn.header import URL_MAPPINGS
from libcomcat.search import search, get_event_by_id

# Fix for NCEDC redirects
URL_MAPPINGS['SCEDC'] = "https://service.scedc.caltech.edu"
URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"


# --- HELPER: Directory Management ---
def ensure_run_directory(event_id, base_dir="outdata"):
    """
    Creates a unique directory for the event run if it doesn't exist.
    Returns the path to that directory.
    """
    # Structure: outdata/Event_ci12345678
    run_dir = os.path.join(base_dir, f"Event_{event_id}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir


# --- DATA RETRIEVAL ---

def get_recent_quakes(min_magnitude=3.0, lookback_hours=24):
    """
    Searches USGS ComCat for recent earthquakes in California.
    """
    start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=lookback_hours)
    print(f"--- TOOL: Scanning USGS for M{min_magnitude}+ CA events since {start_time.strftime('%H:%M')} UTC ---")

    try:
        events = search(starttime=start_time, minmagnitude=min_magnitude,
                        maxlatitude=42.0, minlatitude=32.0,
                        maxlongitude=-114.0, minlongitude=-125.0)
    except Exception as e:
        print(f"    USGS Search Error: {e}")
        return []

    structured_events = []
    if events:
        sorted_events = sorted(events, key=lambda x: x.magnitude, reverse=True)
        for event in sorted_events:
            structured_events.append({
                "id": event.id,
                "magnitude": event.magnitude,
                "time": event.time.isoformat(),
                "location": event.location,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth
            })
    return structured_events


def get_event_details(event_id):
    """Retrieves details for a specific earthquake ID."""
    print(f"--- TOOL: Retrieving details for Event ID: {event_id} ---")
    try:
        event = get_event_by_id(event_id)
        return {
            "id": event.id,
            "magnitude": event.magnitude,
            "time": event.time.isoformat(),
            "location": event.location,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth
        }
    except Exception as e:
        print(f"Error finding event {event_id}: {e}")
        return None


def get_mechanism(event_id):
    """Fetches Focal Mechanism (Strike/Dip/Rake)."""
    print(f"--- TOOL: Searching for Focal Mechanism/Moment Tensor ---")
    try:
        detail = get_event_by_id(event_id)
        try:
            mt = detail.getProducts('moment-tensor')
        except Exception:
            mt = None
        if not mt:
            try:
                mt = detail.getProducts('focal-mechanism')
            except Exception:
                mt = None

        if not mt:
            print("    No mechanism product found. Using generic defaults.")
            return {"strike": 0, "dip": 90, "rake": 0}

        tensor = mt[0]
        props = tensor.properties
        s = float(props.get('nodal-plane-1-strike') or props.get('np1_strike') or 0)
        d = float(props.get('nodal-plane-1-dip') or props.get('np1_dip') or 90)
        r = float(props.get('nodal-plane-1-rake') or props.get('np1_rake') or 0)

        print(f"    Found Mechanism: Strike={s}, Dip={d}, Rake={r}")
        return {"strike": s, "dip": d, "rake": r}
    except Exception as e:
        print(f"    Warning: Could not retrieve mechanism ({e}). Using defaults.")
        return {"strike": 0, "dip": 90, "rake": 0}


# --- STATION & WAVEFORMS ---

def get_nearest_stations(event, max_radius_deg=1.5, max_stations=40):
    """Finds nearest stations (State-Aware: NCEDC vs SCEDC)."""
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
    else:
        client_name = "SCEDC"

    client = Client(client_name)
    event_time = UTCDateTime(event['time'])

    print(f"--- TOOL: Searching stations ({max_radius_deg:.2f} deg) using {client_name} ---")
    print("Event:", event)
    print("RADIUS_Deg:", max_radius_deg)
    try:
        inventory = client.get_stations(
            latitude=event['latitude'], longitude=event['longitude'],
            maxradius=max_radius_deg, starttime=event_time, endtime=event_time + 600,
            network="CI,NC,BK", channel="HN*,BH*,EH*", level="station"
        )
    except Exception as e:
        print(f"    Station search warning: {e}")
        return []

    station_list = []
    for network in inventory:
        for station in network:
            from obspy.geodetics import gps2dist_azimuth
            distance_m, _, _ = gps2dist_azimuth(event['latitude'], event['longitude'], station.latitude,
                                                station.longitude)
            station_list.append({
                "network": network.code,
                "station": station.code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "distance_km": distance_m / 1000.0,
            })

    sorted_stations = sorted(station_list, key=lambda x: x['distance_km'])
    unique_stations = []
    seen = set()
    for s in sorted_stations:
        uid = f"{s['network']}.{s['station']}"
        if uid not in seen:
            unique_stations.append(s)
            seen.add(uid)

    final_list = unique_stations[:max_stations]
    if final_list:
        print(f"    Found {len(final_list)} stations.")
    return final_list


# tools.py (Replace the get_waveforms_and_pga function)

def get_waveforms_and_pga(event, stations):
    """
    Downloads waveforms once per station, calculates PGA, and plots the processed acceleration.
    (REVERTED: Local caching has been removed.)
    """
    import os
    import time
    import numpy as np
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    # Assuming 'ensure_run_directory' is available in the current scope or imported from 'tools'
    # from tools import ensure_run_directory
    #print ("before Datetime")
    event_time = UTCDateTime(event['time'])
    results = []
    #print("before configuration for this function and setup rundirs")
    # Configuration for this function
    MAX_ATTEMPTS = 1
    BASE_SLEEP = 2.0
    MIN_TRACE_COUNT = 2  # Required for RotD50 calculation

    # Setup directories
    run_dir = ensure_run_directory(event['id'])
    plot_dir = os.path.join(run_dir, "observed_plots")
    #print("before test for path",plot_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    # Client Selection logic
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
    else:
        client_name = "SCEDC"
    client = Client(client_name)

    print(f"--- TOOL: Downloading Waveforms & Plotting to {plot_dir} ---")

    for i, sta in enumerate(stations):
        net = sta['network']
        code = sta['station']

        # Politeness Delay between stations
        time.sleep(BASE_SLEEP)

        for attempt in range(MAX_ATTEMPTS):
            try:
                # 1. Inventory Check
                inventory = client.get_stations(
                    network=net, station=code, starttime=event_time, endtime=event_time + 60,
                    level="channel", channel="HN*,BH*"
                )

                # --- Channel Selection Logic (Find suitable pair) ---
                valid_pair = None
                used_instr = None

                for n in inventory:
                    for s in n:
                        if s.channels:
                            used_instr = s.channels[0].code[:2]
                            valid_pair = (s.channels[0], s.channels[1])
                            break
                    if valid_pair: break

                if not valid_pair:
                    print(f"    -> Warning: No valid channel pair found for {net}.{code}. Skipping.")
                    break  # Move to next station

                chan1, chan2 = valid_pair[0].code, valid_pair[1].code
                location_code = valid_pair[0].location_code
                print(f"  - Attempting: {net}.{code} -> {chan1}/{chan2} (Single Request)")

                # 2. Download Waveforms
                st_filtered = client.get_waveforms(
                    net, code, location_code, f"{used_instr}*",  # Use location_code
                    event_time - 10, event_time + 60, attach_response=True
                )

                # Select only the specific two channels to avoid processing extra traces
                st_filtered = st_filtered.select(channel=chan1) + st_filtered.select(channel=chan2)

                # --- CRITICAL CHECK: Ensure we have both required traces ---
                if len(st_filtered) < MIN_TRACE_COUNT:
                    print(f"    -> Warning: Received only {len(st_filtered)} trace(s) for {net}.{code}. Skipping.")
                    break  # Move to next station

                # 3. Response Removal (Conversion to m/s²)
                st_accel = st_filtered.copy()
                pre_filt = [0.05, 0.1, 35, 40]
                # If removal fails, it will raise an Exception handled below
                st_accel.remove_response(output="ACC", pre_filt=pre_filt, water_level=60)

                # 4. Calculate PGA (using m/s² data)
                # Ensure tr.data is used only if it's a valid numpy array
                try:
                    pgas_m_s2 = [np.max(np.abs(tr.data)) for tr in st_accel if len(tr.data) > 0]
                    if len(pgas_m_s2) < MIN_TRACE_COUNT:
                        print(f"    -> Warning: Trace data failed processing after filtering. Skipping.")
                        break  # Failed data integrity check

                    rotd50_m_s2 = np.sqrt(pgas_m_s2[0] * pgas_m_s2[1])
                    rotd50_g = rotd50_m_s2 / 9.81
                except Exception as pga_e:
                    print(f"    -> Error during PGA calculation for {net}.{code}: {pga_e}. Skipping.")
                    break  # Move to next station

                # 5. Scaling and Plotting (cm/s²)
                plot_filename = os.path.join(plot_dir, f"{net}.{code}_{event['id']}_ACCEL_cms2.png")

                st_plot = st_accel.copy()
                for tr in st_plot:
                    tr.data *= 100.0  # m/s^2 -> cm/s^2
                    tr.stats.unit = "cm/s²"
                    tr.stats.channel = f"ACCEL ({tr.stats.channel.split('_')[0]})"

                st_plot.plot(outfile=plot_filename, show=False)

                # 6. Save PGA Result
                print(f"    -> Success! RotD50={rotd50_m_s2:.4f} m/s²")

                # The crash fix (implicit): Ensure a valid dictionary is ALWAYS appended on success
                results.append({
                    "station": f"{net}.{code}",
                    "latitude": sta['latitude'],
                    "longitude": sta['longitude'],
                    "distance_km": sta['distance_km'],
                    "pga_m_s2": rotd50_m_s2,  # This is a float
                    "pga_g": rotd50_g  # This is a float
                })
                break  # Success break

            except Exception as e:
                # Generalized failure handler for Obspy/Network issues
                if "500" in str(e):
                    print(f"    -> Server busy, skipping station {net}.{code}.")
                else:
                    print(f"    -> Error processing station {net}.{code}: {e}")

                break  # Stop trying and move to next station

    # The returned 'results' list contains only valid dictionaries with float values,
    # preventing the 'dict > float' error in subsequent functions.
    return results

# --- BBP GENERATION & EXECUTION ---

def select_1d_velocity_model(event):
    # print("event[lat]",event['latitude'])
    """Returns '2' for Northern CA (Lat > 36), '1' for Southern CA."""
    # hard code this to southern california until GF for NorCal are added
    if event['latitude'] > 36.0: return "1"  # NOCAL
    return "1"  # LA_BASIN


def calculate_fault_dims(magnitude):
    """Wells & Coppersmith (1994)"""
    area = 10 ** (-3.49 + 0.91 * magnitude)
    width = 10 ** (-1.01 + 0.32 * magnitude)
    if width > 20.0: width = 20.0
    length = area / width
    return length, width


def generate_bbp_src(event_data, mechanism, output_file=None):
    """
    Generates a SCEC Broadband Platform .src file.
    UPDATED: Clamps minimum fault dimensions to 1.5km to prevent 'dx=inf' crashes.
    """
    run_dir = ensure_run_directory(event_data['id'])

    if output_file is None:
        filename = f"event_{event_data['id']}.src"
        output_path = os.path.join(run_dir, filename)
    elif os.path.dirname(output_file) == "":
        output_path = os.path.join(run_dir, output_file)
    else:
        output_path = output_file

    print(f"--- TOOL: Generating BBP Source File: {output_path} ---")

    mag = event_data['magnitude']
    hypo_lat = event_data['latitude']
    hypo_lon = event_data['longitude']
    hypo_depth = event_data['depth_km']

    strike = mechanism['strike']
    dip = mechanism['dip']
    rake = mechanism['rake']

    # 1. Calculate Dimensions
    length, width = calculate_fault_dims(mag)

    # --- CRITICAL STABILITY FIX ---
    # The BBP HfSims module often fails if the fault is smaller than the
    # stochastic grid (approx 1.0km). We clamp the minimum size to 1.5km.
    # This prevents the "dx=inf" / Floating Point Exception.
    MIN_DIM = 1.5
    if length < MIN_DIM:
        print(f"    Notice: Clamping Fault Length from {length:.2f}km to {MIN_DIM}km for stability.")
        length = MIN_DIM
    if width < MIN_DIM:
        print(f"    Notice: Clamping Fault Width from {width:.2f}km to {MIN_DIM}km for stability.")
        width = MIN_DIM

    # 2. Geometry Calculations
    hypo_along_stk = 0.0
    hypo_down_dip = width / 2.0

    rad_dip = math.radians(dip)
    depth_to_top = hypo_depth - (hypo_down_dip * math.sin(rad_dip))

    if depth_to_top < 0:
        depth_to_top = 0.0
        hypo_down_dip = hypo_depth / math.sin(rad_dip)

    offset_dist_km = (width / 2.0) * math.cos(rad_dip)
    if offset_dist_km > 0.001:
        azimuth_rad = math.radians(strike - 90)
        d_lat = (offset_dist_km * math.cos(azimuth_rad)) / 111.0
        d_lon = (offset_dist_km * math.sin(azimuth_rad)) / (111.0 * math.cos(math.radians(hypo_lat)))
        lat_top_center = hypo_lat + d_lat
        lon_top_center = hypo_lon + d_lon
    else:
        lat_top_center = hypo_lat
        lon_top_center = hypo_lon

    # 3. Write File
    with open(output_path, "w") as f:
        f.write(f"MAGNITUDE = {mag:.2f}\n")
        f.write(f"FAULT_LENGTH = {length:.4f}\n")
        f.write(f"FAULT_WIDTH = {width:.4f}\n")
        f.write(f"DEPTH_TO_TOP = {depth_to_top:.2f}\n")
        f.write(f"STRIKE = {strike:.1f}\n")
        f.write(f"DIP = {dip:.1f}\n")
        f.write(f"RAKE = {rake:.1f}\n")
        f.write(f"LAT_TOP_CENTER = {lat_top_center:.5f}\n")
        f.write(f"LON_TOP_CENTER = {lon_top_center:.5f}\n")
        f.write(f"HYPO_ALONG_STK = {hypo_along_stk:.1f}\n")
        f.write(f"HYPO_DOWN_DIP = {hypo_down_dip:.2f}\n")

        # Standard Grid Params (Safe now that L/W are clamped)
        f.write(f"DLEN = 0.1\n")
        f.write(f"DWID = 0.1\n")
        f.write(f"DT = 0.1\n")
        f.write(f"HF_DT = 0.01\n")
        f.write(f"SEED = 12345\n")

    print(f"    File written. L={length:.2f}km, W={width:.2f}km (Clamped)")
    return output_path

def generate_bbp_stl(event_data, stations,output_file=None):
    """
        Generates the SCEC Broadband Platform Station List (.stl) file.

        Args:
            event (obj): The unique ID of the earthquake event.
            stations (list): List of dictionaries, each describing a station.

        Returns:
            str: The full path to the generated .stl file.
        """
    import os

    run_dir = ensure_run_directory(event_data['id'])
    #print("rundir",run_dir)

    if output_file is None:
        filename = f"event_{event_data['id']}.stl"
        output_path = os.path.join(run_dir, filename)
    elif os.path.dirname(output_file) == "":
        output_path = os.path.join(run_dir, output_file)
    else:
        output_path = output_file

    print(f"--- TOOL: Generating BBP Source File: {output_path} ---")

    # Ensure run_dir exists (assuming you have an ensure_run_directory function)
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    print(f"--- TOOL: Generating BBP Station List: {output_path} ---")

    # 2. Write File Content
    try:
        with open(output_path, "w") as f:
            # BBP STL files often require a header, followed by station data
            f.write("# Station List File\n")
            f.write("# STA NET LOC LAT LON \n")

            for sta in stations:
                # The format must match what BBP expects (typically space or comma separated)
                #f.write(f"{sta['station']} {sta['network']} -- {sta['latitude']} {sta['longitude']}\n")
                f.write(f"{sta['longitude']} {sta['latitude']} {sta['station']}\n")
        print(f"    Station List file written successfully.")

        # 3. Return the file path (The Fix!)
        return output_path

    except Exception as e:
        print(f"    Error writing Station List file: {e}")
        # Return a non-Path value (e.g., None) if it fails
        return None


def generate_bbp_input_text(event_data, src_file, stl_file, output_file=None):
    """
    Generates the input text file for Docker.
    UPDATED: Fixes the path mismatch by including 'outdata' in the Docker path.
    """
    ev_id = event_data['id']
    run_dir = ensure_run_directory(ev_id)

    if output_file is None:
        filename = f"bbp_input_{ev_id}.txt"
        output_path = os.path.join(run_dir, filename)
    elif os.path.dirname(output_file) == "":
        output_path = os.path.join(run_dir, output_file)
    else:
        output_path = output_file

    print(f"--- TOOL: Generating BBP Input Script: {output_path} ---")
    #print(event_data)
    #print(src_file)
    #print(stl_file)
    #print(ev_id)
    vel_model_id = select_1d_velocity_model(event_data)

    # --- FIX IS HERE ---
    # We must include 'outdata' because Docker mounts the project root to /app/target
    docker_src = f"/app/target/outdata/Event_{ev_id}/{os.path.basename(src_file)}"
    docker_stl = f"/app/target/outdata/Event_{ev_id}/{os.path.basename(stl_file)}"

    # Input Sequence:
    # n (Validation=No) -> {vel_model} -> 1 (GP) -> 2 (Src Path) -> {src} -> 2 (Stl Path) -> {stl} -> y (Run)
    input_content = f"n\n{vel_model_id}\n1\n2\n{docker_src}\n2\n{docker_stl}\ny\n"

    with open(output_path, "w") as f:
        f.write(input_content)
    return output_path


def run_bbp_simulation(input_text_filename):
    """
    Runs the SCEC BBP Docker container.
    UPDATED: Checks for output files to verify success, even if Docker exit code is non-zero.
    """
    print(f"\n--- TOOL: Launching Docker Simulation ---")

    # 1. Determine Event ID from filename (bbp_input_ci12345.txt)
    try:
        event_id_part = os.path.basename(input_text_filename).replace("bbp_input_", "").replace(".txt", "")
        run_dir = os.path.join("outdata", f"Event_{event_id_part}")
    except:
        run_dir = None

    cwd = os.getcwd()

    docker_cmd = (
        f"docker run --rm -i "
        f"-v {cwd}:/app/target "
        f"--ulimit stack=-1 "
        f"--platform linux/amd64 "
        f"sceccode/bbp_22_4:latest "
        f"/home/scecuser/bbp/bbp/comps/run_bbp.py"
    )

    full_cmd = f"cat {input_text_filename} | {docker_cmd}"
    print(f"    Executing with input: {input_text_filename}")

    try:
        # We allow check=False so we can inspect the output even if it 'fails'
        process = subprocess.run(full_cmd, shell=True, check=False, text=True)

        # --- FIX: Add a brief sync delay to fix an exit error on client ---
        time.sleep(0.5)

        # 2. VERIFY BY FILE EXISTENCE
        # If the simulation worked, we should see .rd50 files in a numbered folder inside the event dir
        if run_dir and os.path.exists(run_dir):
            # The BBP creates a numbered folder (e.g. 5142394) inside the run_dir
            # We look for ANY .rd50 file in any subdirectory
            rd50_files = glob.glob(f"{run_dir}/*/*.rd50")

            if rd50_files:
                print(f"    Docker finished. Found {len(rd50_files)} result files.")
                return True
            else:
                print(f"    Docker finished, but NO result files (.rd50) found.")
                return False

        # Fallback if we can't check files
        return process.returncode == 0

    except Exception as e:
        print(f"    Docker Execution Exception: {e}")
        return False



# --- OUTPUT & MAPPING ---

# tools.py (Replace the get_simulated_pgas function)

def get_simulated_pgas(pga_data, event_id, output_dir="outdata"):
    """
    Parses BBP output (.rd50 files) for simulated PGA values.
    Uses the known event_id to locate the final BBP job folder.
    """
    import glob
    print(f"\n--- TOOL: Parsing Simulated PGA Results for {event_id} ---")

    abs_out_dir = os.path.abspath(output_dir)

    # 1. Locate the correct event directory (e.g., outdata/Event_ci12345678)
    target_event_dir = os.path.join(abs_out_dir, f"Event_{event_id}")

    if not os.path.isdir(target_event_dir):
        print(f"    Event directory not found: {target_event_dir}")
        return []

    # 2. Find the BBP internal run folder (e.g., '5158010')
    # This is the directory BBP names after the random job ID
    # We look for any folder named purely with digits inside the event folder
    numbered_run_dirs = [d for d in glob.glob(f"{target_event_dir}/*")
                         if os.path.isdir(d) and os.path.basename(d).isdigit()]

    if not numbered_run_dirs:
        print(f"    Could not find BBP numbered output job folders inside {target_event_dir}.")
        return []

    # Use the latest numbered folder created
    latest_run_dir = max(numbered_run_dirs, key=os.path.getmtime)
    print(f"    Reading results from BBP job folder: {os.path.basename(latest_run_dir)}")

    # 3. Parse .rd50 files (search inside the final directory)
    rd50_files = glob.glob(f"{latest_run_dir}/*.rd50")
    # ... (Rest of function remains the same: parsing logic) ...
    # ... (Please paste the rest of your original get_simulated_pgas code here) ...

    # --- Start of parsing logic that needs to be retained ---
    sim_vals_by_filename = {}
    for rfile in rd50_files:
        # ... (parsing loop as before) ...
        # ...
        pass

    # 4. Combine Simulated PGAs with Station Coordinates
    # ... (Combine logic as before) ...

    return simulated_pga_list


# tools.py (Add or replace generate_display_map)

def generate_display_map(pga_data, event_data, filename, run_dir, title):
    """
    Generates a map showing PGA data (either observed or simulated)
    at station locations.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Setup Data for Plotting
    lats = [d['latitude'] for d in pga_data]
    lons = [d['longitude'] for d in pga_data]

    # Plot PGA in cm/s² (100 * g) for visualization
    pgas_cm_s2 = [d['pga_g'] * 981.0 for d in pga_data]

    # 2. Create the Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(lons, lats, c=pgas_cm_s2, cmap='viridis', s=200, edgecolor='black', alpha=0.8)

    # Plot Earthquake Epicenter
    plt.plot(event_data['longitude'], event_data['latitude'],
             'r*', markersize=20, label=f"Epicenter (M{event_data['magnitude']})")

    # 3. Aesthetics
    print("before title",title,type(title))
    plt.colorbar(label=title + ' (cm/s²)')
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    #plt.title(f"{event_data['title']}\n{title}")
    plt.title(f"{title}")
    plt.grid(True)

    # 4. Save the File
    output_path = os.path.join(run_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"    Map saved to: {output_path}")


# tools.py (Update the compare_results function definition)

# tools.py (Update the compare_results function)

# tools.py (Replace the compare_results function)

def compare_results(pga_data, event_id, event_data, output_dir="outdata"):
    """
    Compares observed PGA vs simulated PGA, generates the simulated map,
    and calculates validation residuals.
    """
    import os
    import numpy as np  # Needed for array operations

    # CRITICAL: Define the run directory once at the top
    run_dir = os.path.join(output_dir, f"Event_{event_id}")

    print("\n--- TOOL: Generating Simulated Maps & Calculating Residuals ---")

    # 1. Get Simulated PGAs
    sim_data = get_simulated_pgas(pga_data, event_id, output_dir=output_dir)

    if not sim_data:
        print("    No simulated PGA data found to compare. Cannot proceed.")
        return False

    # 2. Generate the map for SIMULATED data
    sim_map_filename = f"display_sim_map_pga_{event_id}.png"

    # Call the mapping function with all required positional arguments
    generate_display_map(
        pga_data=sim_data,
        event_data=event_data,
        filename=sim_map_filename,
        run_dir=run_dir,
        title=f"BBP Simulated PGA (GP Method)"
    )

    # 3. Calculate Residuals (Validation Metrics)

    # Ensure both lists are sorted/aligned based on station code
    obs_data_dict = {d['station']: d['pga_g'] for d in pga_data}
    sim_data_dict = {d['station']: d['pga_g'] for d in sim_data}

    stations_to_compare = sorted(list(obs_data_dict.keys() & sim_data_dict.keys()))

    residuals = []

    # Calculate Residual: log10(Simulated) - log10(Observed)
    print("\n| Station | Observed PGA (g) | Simulated PGA (g) | Residual (log10)|")
    print("|:--- |:---:|:---:|:---:|")

    for station in stations_to_compare:
        obs_pga = obs_data_dict[station]
        sim_pga = sim_data_dict[station]

        # Calculate log residual only if both values are greater than zero
        if obs_pga > 0 and sim_pga > 0:
            residual = np.log10(sim_pga) - np.log10(obs_pga)
            residuals.append(residual)
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | {residual:.3f} |")
        else:
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | N/A (Zero Value) |")

    # 4. Final Summary
    if residuals:
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        print(f"\n✅ Validation Summary:")
        print(f"   Mean Residual (Bias): {mean_residual:.3f} (Lower bias is better)")
        print(f"   Std. Dev. Residual: {std_residual:.3f} (Lower scatter is better)")

        # Optional: Generate a map showing residuals (e.g., color-coded dots)
        # generate_residual_map(...)

    return True