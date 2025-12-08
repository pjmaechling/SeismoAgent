# display_monitor_live.py (Updated Imports)

import time
import datetime
import os
from tools import (get_recent_quakes, get_mechanism, generate_bbp_src,
                   get_nearest_stations, get_waveforms_and_pga,
                   generate_bbp_stl, generate_bbp_input_text,
                   run_bbp_simulation, select_1d_velocity_model,
                   get_simulated_pgas, generate_display_map, compare_results) # <-- ADD compare_results

# --- CONFIGURATION ---
MIN_MAGNITUDE = 3.0
SEARCH_RADIUS_KM = 300.0
RADIUS_DEG = SEARCH_RADIUS_KM / 111.0
CHECK_INTERVAL_SECONDS = 600  # Check every 10 minutes


# display_monitor_live.py (Replace the run_live_workflow function)

def run_live_workflow(event):
    """
    Executes the Dual-Map Display Pipeline for a single real-time event.
    """
    ev_id = event['id']

    # 1. Get Mechanism & Source File (generate_bbp_src assigns src_file)
    mechanism = get_mechanism(ev_id)

    # If you use the type-check guard (which is still good practice):
    if not isinstance(mechanism, dict):
        print("    >> Mechanism failed retrieval. Applying safe default.")
        mechanism = {'strike': 0.0, 'dip': 90.0, 'rake': 0.0, 'source_type': 'FALLBACK'}

    src_file = generate_bbp_src(event, mechanism)

    # 2. Get Stations & Velocity Model (select_1d_velocity_model assigns vm_file)
    #stations = get_nearest_stations(event, radius_deg=RADIUS_DEG)
    stations = get_nearest_stations(event, max_radius_deg=RADIUS_DEG)
    vm_file = select_1d_velocity_model(event)

    # 3. Get Observed PGAs (get_waveforms_and_pga assigns observed_pga_data)
    observed_pga_data = get_waveforms_and_pga(ev_id, event, stations)  # <-- ASSIGNS observed_pga_data

    if not observed_pga_data:
        print("❌ No valid observed PGA data could be retrieved. Halting workflow.")
        return

    valid_observed_data = []
    for item in observed_pga_data:
        # Only keep results that are dictionaries AND have the 'pga_g' key
        # (ensuring a successful calculation occurred)
        if isinstance(item, dict) and 'pga_g' in item:
            valid_observed_data.append(item)
        else:
            print(f"    Warning: Observed data contained invalid item: {item}. Discarding.")

    if not valid_observed_data:
        print("❌ All observed PGA data failed validation. Halting workflow.")
        return

    # --- Suggested addition to run_live_workflow before step 6 ---
    # Generate Map 1: Observed Data
    generate_display_map(
        pga_data=observed_pga_data,
        event_data=event,
        filename=f"display_obs_map_pga_{ev_id}.png",
        run_dir=run_dir,
        title=f"Observed PGA (RotD50)"
    )
    # -------------------------------------------------------------

    # 4. Generate Station List File (generate_bbp_stl assigns stl_file)
    stl_file = generate_bbp_stl(ev_id, stations)

    # 5. Generate BBP Input File (generate_bbp_input_text assigns input_file)
    input_file = generate_bbp_input_text(ev_id, event, src_file, stl_file, vm_file)  # <-- ASSIGNS input_file

    # ------------------------------------------------------------------
    # 6. Run Docker
    print(f"    >> Launching BBP Physics Engine...")
    # input_file is USED here:
    success = run_bbp_simulation(input_file)

    if not success:
        print("❌ BBP Simulation failed.")
        return

    # ------------------------------------------------------------------
    # 7. Compare Results and Generate Map 2
    # observed_pga_data is USED here:
    compare_results(observed_pga_data, ev_id, event)

    print(f"✅ COMPLETED: {ev_id}")
    print(f"   Maps saved to outdata/Event_{ev_id}/")


def start_live_monitor():
    print(f"--- SEISMO AGENT: LIVE DISPLAY MONITOR ---")
    # ... (other print statements) ...

    # Memory to prevent re-running events
    processed_ids = set()

    while True:
        try:
            # 1. Scan USGS
            recent_events = get_recent_quakes(min_magnitude=MIN_MAGNITUDE)

            # 2. Find NEW events
            new_events = [e for e in recent_events if e['id'] not in processed_ids]

            if new_events:
                print(f"\n>> Found {len(new_events)} new event(s). Processing...")

                # Process the largest new event first
                target = new_events[0]

                # --- NEW TRY/EXCEPT BLOCK FOR EVENT PROCESSING ---
                try:
                    run_live_workflow(target)  # <-- THIS IS THE CRITICAL CALL

                    # Mark ALL as processed ONLY IF the workflow ran without crashing
                    for e in new_events:
                        processed_ids.add(e['id'])

                except Exception as e:
                    # Log the specific event and crash details, then continue the main loop
                    print(f"\n🚨 EVENT PROCESSING FAILED for {target.get('id', 'Unknown ID')}: {e}")
                    print("    >> Skipping this event and continuing monitoring.")
                # ----------------------------------------------------

            else:
                pass  # Silent wait

        except KeyboardInterrupt:
            # Allow clean exit via Ctrl+C
            print("\nShutting down monitor...")
            break
        except Exception as e:
            # This catches errors in the scanning/sleep part of the loop (less common)
            print(f"CRASH PREVENTION: Main loop error: {e}")

        # 3. Sleep
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_live_monitor()