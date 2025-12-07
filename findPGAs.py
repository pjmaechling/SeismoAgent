# findPGAs.py
import setuptools
import importlib_metadata
import time

from tools import (get_event_details, get_nearest_stations, get_waveforms_and_pga,
                   get_mechanism, generate_bbp_src, generate_bbp_stl,
                   generate_bbp_input_text, run_bbp_simulation)  # <--- Corrected Import

# TARGET: La Habra Earthquake (March 29, 2014) - M5.1
TARGET_EVENT_ID = "ci15481673"


def run_agent_workflow():
    print(f"Agent started (La Habra Test Mode: {TARGET_EVENT_ID}).")

    # Step 1: Event Details
    print(f"\n--- STEP 1: Get Details for Event {TARGET_EVENT_ID} ---")
    target_event = get_event_details(TARGET_EVENT_ID)
    if not target_event: return

    # Step 2: Source File (SRC)
    print(f"\n--- STEP 2: Generate Source File ---")
    mechanism = get_mechanism(TARGET_EVENT_ID)
    src_filename = f"event_{TARGET_EVENT_ID}.src"
    generate_bbp_src(target_event, mechanism, output_file=src_filename)

    # Step 3: Find Stations
    print("\n--- STEP 3: Find Nearest Stations ---")

    # IMPRESSIVE UPDATE:
    # 1. Radius: 1.5 degrees (~165 km) covers the whole LA Basin
    # 2. Max Stations: 40 gives you a dense cloud of data points
    stations = get_nearest_stations(target_event, max_radius_deg=1.5, max_stations=40)

    if not stations: return

    print(f"Found {len(stations)} stations.")

    # Step 4: Get Data (PGA)
    print("\n--- STEP 4: Analyze Ground Motion ---")
    pga_data = get_waveforms_and_pga(target_event, stations)

    if pga_data:
        # Step 5: Generate Station List (STL)
        print(f"\n--- STEP 5: Generate Station List ---")
        stl_filename = f"event_{TARGET_EVENT_ID}.stl"
        generate_bbp_stl(pga_data, output_file=stl_filename)

        # Step 6: Generate BBP Input Script
        print(f"\n--- STEP 6: Generate Input Script ---")
        input_filename = "bbp_input.txt"
        generate_bbp_input_text(target_event, src_filename, stl_filename, output_file=input_filename)

        # Step 7: RUN DOCKER
        print(f"\n--- STEP 7: RUNNING PHYSICS SIMULATION ---")
        print("Handing control to SCEC Broadband Platform (Docker)...")
        success = run_bbp_simulation(input_filename)

        if success:
            # ... inside run_agent_workflow, after run_bbp_simulation succeeds:
            if success:
                print("\n--- AGENT SUCCESS ---")
                print(f"La Habra simulation completed.")

                # CALL THE NEW COMPARISON TOOL
                from tools import compare_results
                compare_results(pga_data, TARGET_EVENT_ID,target_event)

    else:
        print("Could not retrieve waveforms. Workflow aborted.")


if __name__ == "__main__":
    run_agent_workflow()