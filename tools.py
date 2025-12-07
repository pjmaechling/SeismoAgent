# tools.py
import subprocess
import contextily
import os
import sys
# ... (rest of your imports like numpy, obspy, etc.)
import matplotlib.pyplot as plt
import datetime
import math
import time
from libcomcat.search import search
from obspy.geodetics import gps2dist_azimuth
# tools.py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


def generate_comparison_map(pga_data, sim_vals, event_data, output_file="comparison_map.png"):
    """
    Generates a map showing Observed vs Simulated residuals.
    UPDATED: Adds an OpenStreetMap basemap with roads using contextily.
    """
    import matplotlib.pyplot as plt
    # New import for basemaps
    try:
        import contextily as cx
        has_contextily = True
    except ImportError:
        print("    Warning: 'contextily' library not found. Map will be generated without roads.")
        print("    To see roads, run: pip install contextily")
        has_contextily = False

    print(f"--- TOOL: Generating Comparison Map: {output_file} ---")

    lats, lons, diffs = [], [], []

    for obs in pga_data:
        sid = obs['station']
        obs_g = obs['pga_g']

        # Find matching simulation value
        matched_sim_g = None
        for fname, val in sim_vals.items():
            if sid in fname:
                matched_sim_g = val
                break

        if matched_sim_g:
            # Calculate % Difference
            diff = (matched_sim_g - obs_g) / obs_g * 100.0
            lats.append(obs['latitude'])
            lons.append(obs['longitude'])
            diffs.append(diff)

    if not lats:
        print("    No matched data to plot.")
        return

    # Use subplots to get the axes object required by contextily
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot Stations (Increased zorder to sit on top of map)
    scatter = ax.scatter(lons, lats, c=diffs, cmap='seismic', vmin=-100, vmax=100,
                         s=150, edgecolors='k', zorder=10)

    # 2. Plot Earthquake Epicenter (Highest zorder)
    ax.scatter(event_data['longitude'], event_data['latitude'], s=500, c='gold',
               marker='*', edgecolors='black', label='Epicenter', zorder=15)

    # 3. Add Basemap (Roads via OpenStreetMap)
    if has_contextily:
        try:
            print("    Downloading basemap tiles (this might take a moment)...")
            # crs='EPSG:4326' tells contextily our data is in standard Lat/Lon degrees.
            # It automatically reprojects the web tiles to match.
            cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, alpha=0.8)
        except Exception as e:
            print(f"    Warning: Could not fetch basemap tiles ({e}). Plotting without roads.")

    # 4. Decoration
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Simulation Bias (%) \n(Blue = Under-Predict, Red = Over-Predict)', fontsize=10)

    ax.set_title(f"Simulation Accuracy: Event {event_data['id']}\n(M{event_data['magnitude']})", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper right')

    # Ensure map doesn't look stretched
    ax.set_aspect('equal')

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Map saved to {output_file}")
    plt.close()


from obspy.clients.fdsn.header import URL_MAPPINGS
URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"
# Explicit mapping to ensure we hit the SCEDC server correctly
URL_MAPPINGS['SCEDC'] = "https://service.scedc.caltech.edu"

# ... (Keep select_1d_velocity_model, generate_bbp_input_text, run_bbp_simulation, etc.) ...

def get_waveforms_and_pga(event, stations):
    """
    Downloads waveforms & Plots.
    UPDATED: Includes Rate Limiting (Sleep) and Retry Logic to avoid HTTP 500 errors.
    """
    event_time = UTCDateTime(event['time'])
    results = []

    # Dynamic Data Center Selection
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
        URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"
    else:
        client_name = "SCEDC"

    client = Client(client_name)

    if not os.path.exists("observed_plots"):
        os.makedirs("observed_plots")

    print(f"--- TOOL: Downloading Horizontal Waveforms & Plotting ---")

    for i, sta in enumerate(stations):
        net = sta['network']
        code = sta['station']
        dist = sta['distance_km']

        # STRICT FILTER: Only process target networks
        if net not in ["CI", "NC", "BK"]:
            continue

        # RATE LIMITING: Pause every 5 stations to let the server connection close
        if i > 0 and i % 5 == 0:
            time.sleep(2)

            # RETRY LOOP: Try up to 3 times if we get a server error
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Ask for channels (Lightweight metadata query)
                try:
                    inventory = client.get_stations(
                        network=net, station=code,
                        starttime=event_time, endtime=event_time + 60,
                        level="channel", channel="HN*,BH*"
                    )
                except Exception:
                    # If metadata fails, it's usually not a server overload, just no data.
                    # Try IRIS once, then skip.
                    try:
                        client_fallback = Client("IRIS")
                        inventory = client_fallback.get_stations(
                            network=net, station=code,
                            starttime=event_time, endtime=event_time + 60,
                            level="channel", channel="HN*,BH*"
                        )
                        client_active = client_fallback
                    except:
                        break  # Break inner retry loop, move to next station
                else:
                    client_active = client

                # 2. Find Pair (Logic unchanged)
                valid_pair = None
                used_instr = None

                instr_groups = {}
                for n in inventory:
                    for s in n:
                        for c in s:
                            instr = c.code[:2]
                            if instr not in instr_groups: instr_groups[instr] = []
                            instr_groups[instr].append(c)

                for instr in ["HN", "BH"]:
                    if instr in instr_groups:
                        chans = instr_groups[instr]
                        horizontals = [c for c in chans if c.code[2] in ['N', 'E', '1', '2']]
                        if len(horizontals) >= 2:
                            horizontals.sort(key=lambda x: x.code)
                            valid_pair = (horizontals[0], horizontals[1])
                            used_instr = instr
                            break

                if not valid_pair:
                    break  # Stop retrying if simply no pair exists

                # 3. Download (The heavy part that causes 500 errors)
                chan1 = valid_pair[0]
                chan2 = valid_pair[1]

                print(
                    f"  - Attempting: {net}.{code:<5} ({dist:.1f} km) -> {chan1.code}/{chan2.code} (Try {attempt + 1})")

                st = client_active.get_waveforms(
                    net, code, chan1.location_code, f"{used_instr}*",
                    event_time - 10, event_time + 60,
                    attach_response=True
                )

                st_filtered = st.select(channel=chan1.code) + st.select(channel=chan2.code)
                if len(st_filtered) < 2:
                    print(f"    -> Failed: Incomplete pair.")
                    break

                    # 4. PLOT & PHYSICS (Success path)
                plot_filename = f"observed_plots/{net}.{code}.png"
                st_filtered.plot(outfile=plot_filename)

                pgas = []
                pre_filt = [0.05, 0.1, 35, 40]

                for tr in st_filtered:
                    tr.remove_response(output="ACC", pre_filt=pre_filt, water_level=60)
                    pgas.append(np.max(np.abs(tr.data)))

                rotd50_m_s2 = np.sqrt(pgas[0] * pgas[1])
                rotd50_g = rotd50_m_s2 / 9.81

                print(f"    -> Success! RotD50~={rotd50_m_s2:.4f} m/s²")

                results.append({
                    "station": f"{net}.{code}",
                    "latitude": sta['latitude'],
                    "longitude": sta['longitude'],
                    "distance_km": sta['distance_km'],
                    "pga_m_s2": rotd50_m_s2,
                    "pga_g": rotd50_g
                })

                # Success! Break the retry loop and go to next station
                break

            except Exception as e:
                # Catch the 500 Error
                err_msg = str(e)
                if "500" in err_msg or "Internal Server Error" in err_msg:
                    print(f"    -> Server Overload (500). Waiting 10s...")
                    time.sleep(10)  # Back off significantly
                else:
                    print(f"    -> Error: {e}")
                    break  # If it's not a server error (e.g. 404), don't retry.

        # Gentle pause between every station
        time.sleep(1.0)

    return results

# tools.py (Replace compare_results)

def compare_results(pga_data, sim_id, event_data, output_dir="outdata"):
    """
    Parses BBP output, prints table, and generates the COMPARISON MAP inside the run folder.
    """
    print(f"\n--- TOOL: Validating Simulation Results ---")

    # 1. Find the latest run directory
    abs_out_dir = os.path.abspath(output_dir)
    try:
        subdirs = [d for d in glob.glob(f"{abs_out_dir}/*") if os.path.isdir(d)]
        latest_run_dir = max(subdirs, key=os.path.getmtime)
        run_id = os.path.basename(latest_run_dir)
        print(f"    Analysis directory: {latest_run_dir}")
    except ValueError:
        print("    No output directory found.")
        return

    # 2. Parse .rd50 files
    rd50_files = glob.glob(f"{latest_run_dir}/*.rd50")
    sim_vals = {}

    for rfile in rd50_files:
        fname = os.path.basename(rfile)
        with open(rfile, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                try:
                    if float(parts[0]) < 0.02:  # PGA
                        sim_vals[fname] = float(parts[1])
                        break
                except:
                    continue

    # 3. Print Table
    print(f"\n{'STATION':<10} | {'OBS (g)':<10} | {'SIM (g)':<10} | {'DIFF':<8} | {'VERDICT':<15}")
    print("-" * 65)

    for obs in pga_data:
        sid = obs['station']
        obs_g = obs['pga_g']

        matched_sim_g = None
        for fname in sim_vals:
            if sid in fname:
                matched_sim_g = sim_vals[fname]
                break

        if matched_sim_g:
            diff = (matched_sim_g - obs_g) / obs_g * 100.0
            verdict = "Accurate"
            if diff > 50:
                verdict = "Over-Predict"
            elif diff < -50:
                verdict = "Under-Predict"

            print(f"{sid:<10} | {obs_g:<10.4f} | {matched_sim_g:<10.4f} | {diff:+.0f}%    | {verdict}")
        else:
            print(f"{sid:<10} | {obs_g:<10.4f} | {'N/A':<10} | --       | --")

    # 4. Generate Map INSIDE the run directory
    # We name it clearly with the Run ID
    map_filename = os.path.join(latest_run_dir, f"comparison_map_{run_id}.png")

    generate_comparison_map(pga_data, sim_vals, event_data, output_file=map_filename)

# tools.py (Replace existing get_mechanism)
def get_mechanism(event_id):
    """
    Fetches the best available Focal Mechanism.
    Handles libcomcat/pandas compatibility issues.
    """
    print(f"--- TOOL: Searching for Focal Mechanism/Moment Tensor ---")
    try:
        detail = get_event_by_id(event_id)

        # Robust product lookup
        try:
            mt = detail.getProducts('moment-tensor')
        except Exception:
            mt = None

        if not mt:
            try:
                mt = detail.getProducts('focal-mechanism')
            except Exception:
                pass

        if not mt:
            print("    No mechanism product found. Using generic defaults.")
            return {"strike": 0, "dip": 90, "rake": 0}

        # Parse the first available tensor properties directly
        tensor = mt[0]
        props = tensor.properties

        # Robust extraction of Nodal Plane 1
        s = float(props.get('nodal-plane-1-strike') or props.get('np1_strike') or 0)
        d = float(props.get('nodal-plane-1-dip') or props.get('np1_dip') or 90)
        r = float(props.get('nodal-plane-1-rake') or props.get('np1_rake') or 0)

        print(f"    Found Mechanism: Strike={s}, Dip={d}, Rake={r}")
        return {"strike": s, "dip": d, "rake": r}

    except Exception as e:
        print(f"    Warning: Could not retrieve mechanism ({e}).")
        return {"strike": 0, "dip": 90, "rake": 0}


def calculate_fault_dims(magnitude):
    """
    Estimates Fault Length and Width using Wells & Coppersmith (1994)
    Empirical relationships for 'All' slip types.
    """
    # Rupture Area (A) = 10 ^ (-3.49 + 0.91 * Mw)
    area = 10 ** (-3.49 + 0.91 * magnitude)

    # Rupture Width (W) = 10 ^ (-1.01 + 0.32 * Mw)
    width = 10 ** (-1.01 + 0.32 * magnitude)

    # Ensure Width doesn't exceed seismogenic thickness (e.g. 15-20km)
    if width > 20.0:
        width = 20.0

    length = area / width

    return length, width


# tools.py (Replace generate_bbp_src)

# tools.py (Replace generate_bbp_src)

def generate_bbp_src(event_data, mechanism, output_file="event.src"):
    """
    Generates a SCEC Broadband Platform .src file.
    FINAL FIX: Uses correct variable names (HYPO_ALONG_STK) and includes DLEN/DWID.
    """
    print(f"--- TOOL: Generating BBP Source File: {output_file} ---")

    mag = event_data['magnitude']
    hypo_lat = event_data['latitude']
    hypo_lon = event_data['longitude']
    hypo_depth = event_data['depth_km']

    strike = mechanism['strike']
    dip = mechanism['dip']
    rake = mechanism['rake']

    length, width = calculate_fault_dims(mag)

    # Geometry
    hypo_along_stk = 0.0  # Centroid (0.0 is center in BBP convention)
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

    with open(output_file, "w") as f:
        f.write(f"MAGNITUDE = {mag:.2f}\n")
        f.write(f"FAULT_LENGTH = {length:.2f}\n")
        f.write(f"FAULT_WIDTH = {width:.2f}\n")
        f.write(f"DEPTH_TO_TOP = {depth_to_top:.2f}\n")
        f.write(f"STRIKE = {strike:.1f}\n")
        f.write(f"DIP = {dip:.1f}\n")
        f.write(f"RAKE = {rake:.1f}\n")
        f.write(f"LAT_TOP_CENTER = {lat_top_center:.5f}\n")
        f.write(f"LON_TOP_CENTER = {lon_top_center:.5f}\n")

        # FIX: Renamed HYPO_ALONG_STRIKE -> HYPO_ALONG_STK
        f.write(f"HYPO_ALONG_STK = {hypo_along_stk:.1f}\n")
        f.write(f"HYPO_DOWN_DIP = {hypo_down_dip:.2f}\n")

        f.write(f"DLEN = 0.1\n")
        f.write(f"DWID = 0.1\n")
        f.write(f"SEED = 12345\n")

    print(f"    File written. L={length:.1f}km, W={width:.1f}km, Ztop={depth_to_top:.1f}km")
    return output_file

# ... (keep other functions like get_event_details, get_nearest_stations as is) ...

# tools.py
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
# FIX for NCEDC Redirect Issue (common in older ObsPy versions)


# tools.py
from libcomcat.search import get_event_by_id  # <--- Import this


def get_event_details(event_id):
    """
    Retrieves details for a specific historic earthquake by its USGS ID.
    """
    print(f"--- TOOL: Retrieving details for Event ID: {event_id} ---")

    try:
        event = get_event_by_id(event_id)

        return {
            "id": event.id,
            "magnitude": event.magnitude,
            # Use isoformat() for ObsPy compatibility
            "time": event.time.isoformat(),
            "location": event.location,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth
        }
    except Exception as e:
        print(f"Error finding event {event_id}: {e}")
        return None


# tools.py
from libcomcat.search import search
import datetime


def get_recent_quakes(min_magnitude=3.0, lookback_hours=24):
    """
    Searches USGS ComCat for recent earthquakes in California.
    Returns a list of event dictionaries formatted for the BBP pipeline.
    """
    # 1. Calculate time window (UTC)
    start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=lookback_hours)

    print(f"--- TOOL: Scanning USGS for M{min_magnitude}+ CA events since {start_time.strftime('%H:%M')} UTC ---")

    try:
        # Search CA Bounding Box
        events = search(
            starttime=start_time,
            minmagnitude=min_magnitude,
            maxlatitude=42.0, minlatitude=32.0,
            maxlongitude=-114.0, minlongitude=-125.0
        )
    except Exception as e:
        print(f"    USGS Search Error: {e}")
        return []

    structured_events = []

    # Sort: Largest first
    if events:
        sorted_events = sorted(events, key=lambda x: x.magnitude, reverse=True)
        for event in sorted_events:
            structured_events.append({
                "id": event.id,
                "magnitude": event.magnitude,
                "time": event.time.isoformat(),  # ISO format for ObsPy
                "location": event.location,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth
            })

    return structured_events


# tools.py (Replace get_nearest_stations)

def get_nearest_stations(event, max_radius_deg=1.5, max_stations=40):
    """
    Finds nearest stations.
    UPDATED: Prints a table of the stations found and their distances.
    """
    # Dynamic Client Switching
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
        URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"
    else:
        client_name = "SCEDC"

    client = Client(client_name)

    event_time = UTCDateTime(event['time'])
    lat = event['latitude']
    lon = event['longitude']

    print(
        f"--- TOOL: Searching for stations within {max_radius_deg:.2f} deg of {lat:.3f}, {lon:.3f} using {client_name} ---")

    try:
        # RESTRICTED SEARCH: Only CI (SoCal), NC (NoCal), BK (Berkeley)
        inventory = client.get_stations(
            latitude=lat, longitude=lon, maxradius=max_radius_deg,
            starttime=event_time, endtime=event_time + 600,
            network="CI,NC,BK",  # <--- CRITICAL CHANGE HERE
            channel="HN*,BH*",
            level="station"
        )
    except Exception as e:
        print(f"    Station search warning: {e}")
        return []

    station_list = []

    for network in inventory:
        for station in network:
            from obspy.geodetics import gps2dist_azimuth
            distance_m, azimuth, _ = gps2dist_azimuth(lat, lon, station.latitude, station.longitude)

            station_list.append({
                "network": network.code,
                "station": station.code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "distance_km": distance_m / 1000.0,
            })

    # Sort by distance
    sorted_stations = sorted(station_list, key=lambda x: x['distance_km'])

    # Deduplicate
    unique_stations = []
    seen = set()
    for s in sorted_stations:
        uid = f"{s['network']}.{s['station']}"
        if uid not in seen:
            unique_stations.append(s)
            seen.add(uid)

    # --- NEW: Print the Table ---
    final_list = unique_stations[:max_stations]

    if final_list:
        print(f"\n    {'STATION':<12} | {'DIST (km)':<10} | {'NETWORK':<10}")
        print("    " + "-" * 38)
        for s in final_list:
            print(f"    {s['network']}.{s['station']:<9} | {s['distance_km']:<10.1f} | {s['network']}")
        print(f"    Total: {len(final_list)} stations selected.\n")
    else:
        print("    No stations found.")

    return final_list


# tools.py (Replace generate_bbp_stl)

def generate_bbp_stl(pga_data, output_file="stations.stl"):
    """
    Generates a SCEC Broadband Platform Station List (.stl).
    FIX: Corrects column order to [Lon, Lat, ID, Vs30, LoCut, HiCut]
    """
    print(f"--- TOOL: Generating BBP Station List: {output_file} ---")

    with open(output_file, "w") as f:
        for entry in pga_data:
            sid = entry['station']
            lon = entry['longitude']
            lat = entry['latitude']

            # Physics defaults
            vs30 = 863  # Standard LA Basin rock reference
            lo_cut = 0.1
            hi_cut = 50.0

            # CORRECT BBP FORMAT: Longitude Latitude StationID Vs30 LoCut HiCut
            f.write(f"{lon:.4f} {lat:.4f} {sid} {vs30} {lo_cut} {hi_cut}\n")

    print(f"    Written {len(pga_data)} stations to {output_file}")
    return output_file

def select_1d_velocity_model(latitude):
    """
    Selects the appropriate SCEC BBP 1D Velocity Model based on latitude.
    Returns the INTEGER MENU CHOICE for BBP 22.4 interactive mode.
    """
    # Heuristic: North of 36.0 is Northern CA, South is Southern CA
    if latitude > 36.0:
        print(f"    Location (Lat {latitude:.2f}) is Northern CA -> Selecting Option 2 (NOCAL)")
        return "2" # 2 = NOCAL
    else:
        print(f"    Location (Lat {latitude:.2f}) is Southern CA -> Selecting Option 1 (LA_BASIN_500)")
        return "1" # 1 = LA_BASIN_500


# tools.py (Replace generate_bbp_input_text)

def generate_bbp_input_text(event_data, src_file, stl_file, output_file="bbp_input.txt"):
    """
    Generates a text file containing the keystrokes to drive the BBP interactive menu.
    UPDATED for BBP 22.4 Menu Sequence.
    """
    print(f"--- TOOL: Generating BBP Input Script: {output_file} ---")

    # 1. Select Velocity Model (Integer ID: 1=LABasin, 2=NOCAL)
    vel_model_id = select_1d_velocity_model(event_data['latitude'])

    # 2. Define Paths inside Docker
    docker_src = f"/app/target/{src_file}"
    docker_stl = f"/app/target/{stl_file}"

    # 3. Construct Input Sequence (The "Keystrokes")
    # ---------------------------------------------------------
    # Q1: Validation run?                   -> n
    # Q2: Velocity Model?                   -> {vel_model_id}
    # Q3: Choose Method (GP)?               -> 1
    # Q4: Source File: (1) List or (2) Path?-> 2
    # Q5: Enter Source Path:                -> {docker_src}
    # Q6: Station File: (1) List or (2) Path?-> 2
    # Q7: Enter Station Path:               -> {docker_stl}
    # Q8: Run Simulation?                   -> y
    # ---------------------------------------------------------

    input_content = f"n\n{vel_model_id}\n1\n2\n{docker_src}\n2\n{docker_stl}\ny\n"

    with open(output_file, "w") as f:
        f.write(input_content)

    print(f"    Input script written. Content:\n{input_content.strip().replace(chr(10), ' ')}")
    return output_file


def run_bbp_simulation(input_text_filename):
    """
    Runs the SCEC BBP Docker container in Interactive Mode, piping in our input text file.
    """
    print(f"\n--- TOOL: Launching Docker Simulation ---")

    cwd = os.getcwd()

    # We use shell piping: cat input.txt | docker run -i ...
    # This simulates a user typing the answers.

    docker_cmd = (
        f"docker run --rm -i "  # -i is CRITICAL (Interactive/Stdin open)
        f"-v {cwd}:/app/target "
        f"--ulimit stack=-1 "
        f"--platform linux/amd64 "
        f"sceccode/bbp_22_4:latest "
        f"/home/scecuser/bbp/bbp/comps/run_bbp.py"
    )

    full_cmd = f"cat {input_text_filename} | {docker_cmd}"

    print(f"    Executing: {full_cmd}")

    try:
        # shell=True is required for the pipe (|) to work
        process = subprocess.run(full_cmd, shell=True, check=True, text=True)
        print("    Docker simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Docker Execution Failed! Return Code: {e.returncode}")
        return False

    def generate_comparison_map(pga_data, sim_vals, event_data, output_file="comparison_map.png"):
        """
        Generates a map showing Observed vs Simulated residuals.
        Red = Over-prediction, Blue = Under-prediction.
        """
        print(f"--- TOOL: Generating Comparison Map: {output_file} ---")

        lats, lons, diffs = [], [], []

        for obs in pga_data:
            sid = obs['station']
            obs_g = obs['pga_g']

            # Find matching simulation value
            matched_sim_g = None
            for fname, val in sim_vals.items():
                if sid in fname:
                    matched_sim_g = val
                    break

            if matched_sim_g:
                # Calculate % Difference
                diff = (matched_sim_g - obs_g) / obs_g * 100.0
                lats.append(obs['latitude'])
                lons.append(obs['longitude'])
                diffs.append(diff)

        if not lats:
            print("    No matched data to plot.")
            return

        plt.figure(figsize=(10, 8))

        # 1. Plot Stations (Color-coded by difference)
        # Vmin/Vmax set to +/- 100% to keep colors consistent
        scatter = plt.scatter(lons, lats, c=diffs, cmap='seismic', vmin=-100, vmax=100, s=100, edgecolors='k')

        # 2. Plot Earthquake Epicenter
        plt.scatter(event_data['longitude'], event_data['latitude'], s=300, c='gold', marker='*', edgecolors='black',
                    label='Epicenter')

        # 3. Decoration
        cbar = plt.colorbar(scatter)
        cbar.set_label('Simulation Bias (%) \n(Blue=Under-predict, Red=Over-predict)')

        plt.title(f"Simulation Accuracy: Event {event_data['id']}\n(M{event_data['magnitude']})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Save
        plt.savefig(output_file)
        print(f"    Map saved to {output_file}")
        plt.close()
