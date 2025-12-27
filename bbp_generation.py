"""
BBP file generation functions.
"""
import os
import math
from file_utils import ensure_run_directory


def select_1d_velocity_model(event):
    """Returns '2' for Northern CA (Lat > 36), '1' for Southern CA."""
    # hard code this to southern california until GF for NorCal are added
    if event['latitude'] > 36.0:
        return "1"  # NOCAL
    return "1"  # LA_BASIN


def calculate_fault_dims(magnitude):
    """Wells & Coppersmith (1994)"""
    area = 10 ** (-3.49 + 0.91 * magnitude)
    width = 10 ** (-1.01 + 0.32 * magnitude)
    if width > 20.0:
        width = 20.0
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


def generate_bbp_stl(event_data, stations, output_file=None):
    """
    Generates the SCEC Broadband Platform Station List (.stl) file.

    Args:
        event_data (dict): The event data dictionary containing 'id'.
        stations (list): List of dictionaries, each describing a station.

    Returns:
        str: The full path to the generated .stl file.
    """
    run_dir = ensure_run_directory(event_data['id'])

    if output_file is None:
        filename = f"event_{event_data['id']}.stl"
        output_path = os.path.join(run_dir, filename)
    elif os.path.dirname(output_file) == "":
        output_path = os.path.join(run_dir, output_file)
    else:
        output_path = output_file

    print(f"--- TOOL: Generating BBP Source File: {output_path} ---")

    # Ensure run_dir exists
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    print(f"--- TOOL: Generating BBP Station List: {output_path} ---")

    # 2. Write File Content
    try:
        with open(output_path, "w") as f:
            # BBP STL files often require a header, followed by station data
            f.write("# Station List File\n")
            f.write("# STA NET LOC LAT LON \n")

            for sta in stations:
                # The format must match what BBP expects (typically space or comma separated)
                f.write(f"{sta['longitude']} {sta['latitude']} {sta['station']}\n")
        print(f"    Station List file written successfully.")

        # 3. Return the file path
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

