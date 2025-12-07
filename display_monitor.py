import time
import datetime
import numpy as np
import os

# --- Import all necessary BBP tools from tools.py ---
from tools import (get_recent_quakes, get_nearest_stations, get_waveforms_and_pga,
                   get_event_details, get_mechanism, generate_bbp_src,
                   generate_bbp_stl, generate_bbp_input_text, run_bbp_simulation,
                   select_1d_velocity_model, get_simulated_pgas)

# --- CONFIGURATION AND CONSTANTS ---
SEARCH_RADIUS_KM = 300.0  # Wide search for comprehensive display map
RADIUS_DEG = SEARCH_RADIUS_KM / 111.0
MIN_MAGNITUDE = 3.0
CHECK_INTERVAL_SECONDS = 900

# TARGET_EVENT_ID is set for the current test mode (La Habra)
TARGET_EVENT_ID = "ci15481673"


# ----------------------------------------------------------------------
# 1. CORE WORKFLOW FUNCTION (The code you provided)
# ----------------------------------------------------------------------

def run_test_workflow(event_id):
    """
    Runs the display agent for a single, known event ID (La Habra),
    generating both OBSERVED and SIMULATED PGA maps.
    """
    print(f"--- DISPLAY AGENT TEST MODE: Event {event_id} ---")

    # 1. Get Event Details and Stations List (CRITICAL: Stations must be filtered for data first)
    event = get_event_details(event_id)
    if not event: return

    # Get OBSERVED data first, as this gives us the final, validated station list.
    # The inner call to get_nearest_stations is used to generate the initial list of candidates.
    observed_pga_data = get_waveforms_and_pga(event,
                                              get_nearest_stations(event, max_radius_deg=RADIUS_DEG, max_stations=100))

    if not observed_pga_data:
        print("Aborting: Failed to retrieve any OBSERVED PGA data. Cannot run simulation.")
        return

    # ------------------------------------------------------------------
    # OBSERVED MAP GENERATION
    # ------------------------------------------------------------------
    generate_display_map(observed_pga_data, "Observed", "display_obs_map_pga.png")

    # ------------------------------------------------------------------
    # BBP SIMULATION PIPELINE (To generate the simulated data)
    # ------------------------------------------------------------------

    print("\n--- BBP File Generation and Execution ---")

    # 2. Generate Source File (.src)
    mechanism = get_mechanism(event_id)
    src_file = f"event_{event_id}.src"
    generate_bbp_src(event, mechanism, output_file=src_file)

    # 3. Safety Check: Velocity Model
    vel_model_id = select_1d_velocity_model(event['latitude'])
    if vel_model_id == "2":
        print("Simulation aborted: BBP is optimized for LA_BASIN, not Northern California.")
        return

    # 4. Generate Station List File (.stl)
    stl_file = f"event_{event_id}.stl"
    # CRITICAL: This .stl uses the exact locations (lat/lon) of the observed stations.
    generate_bbp_stl(observed_pga_data, output_file=stl_file)

    # 5. Generate BBP Input File (bbp_input.txt)
    input_file = "bbp_input.txt"
    generate_bbp_input_text(event, src_file, stl_file, output_file=input_file)

    # 6. Run Docker Simulation (Execution)
    print(f"\n    >> Launching BBP Physics Engine for Simulation...")
    success = run_bbp_simulation(input_file)

    if not success:
        print("❌ ABORTING: BBP Simulation failed. Cannot generate simulated map.")
        return

    # 7. Get SIMULATED PGA Data (Parsing the newly created BBP output)
    simulated_pga_list = get_simulated_pgas(observed_pga_data)

    if not simulated_pga_list:
        print("Aborting: Failed to parse simulated PGA data.")
        return

    # ------------------------------------------------------------------
    # SIMULATED MAP GENERATION
    # ------------------------------------------------------------------

    # 8. Generate Simulated Map
    generate_display_map(simulated_pga_list, "Simulated", "display_sim_map_pga.png")
    print(f"\n✅ FULL TEST COMPLETE. Two maps generated: display_obs_map_pga.png and display_sim_map_pga.png.")


# ----------------------------------------------------------------------
# 2. SUPPORT FUNCTIONS (Must also be in the file)
# ----------------------------------------------------------------------

def generate_display_map(all_pga_data, data_type, output_file):
    """
    Generates a map showing observed or simulated PGA values. (Reusable map generator)
    """
    import matplotlib.pyplot as plt
    import contextily as cx

    print(f"\n--- TOOL: Generating {data_type} PGA Map ---")

    lats, lons, pga_vals, sizes = [], [], [], []

    if not all_pga_data:
        print(f"    No {data_type} PGA data collected to plot.")
        return

    for pga_record in all_pga_data:
        pga_cm_s2 = pga_record['pga_m_s2'] * 100.0
        pga_g_val = pga_record['pga_g']

        lats.append(pga_record['latitude'])
        lons.append(pga_record['longitude'])
        pga_vals.append(pga_cm_s2)
        sizes.append(max(10, pga_g_val * 1500.0))

    fig, ax = plt.subplots(figsize=(14, 14))

    scatter = ax.scatter(lons, lats, c=pga_vals, cmap='plasma',
                         s=sizes, edgecolors='k', zorder=10,
                         norm=plt.Normalize(vmin=0.5, vmax=10))

    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.PositronNoLabels, alpha=0.8)
    except:
        print("    Warning: Basemap failed to load.")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PGA (cm/s²)', fontsize=12)

    plt.title(f"{data_type} Ground Motion Map (La Habra Event)", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.set_aspect('equal')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Map saved to {output_file}")
    plt.close()


def start_display_monitor():
    # Placeholder for live monitoring loop (not used in test mode)
    pass


# ----------------------------------------------------------------------
# 3. EXECUTION BLOCK
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TARGET_EVENT_ID:
        # Run the single-event test mode
        run_test_workflow(TARGET_EVENT_ID)
    else:
        # Run the continuous live monitoring loop
        start_display_monitor()