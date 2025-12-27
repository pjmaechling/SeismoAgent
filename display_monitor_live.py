# display_monitor_live.py (Updated to use new modular structure)

import time
import datetime
import os

# Import from new modular structure
from file_utils import ensure_run_directory
from event_retrieval import get_recent_quakes, get_mechanism
from station_operations import get_nearest_stations, get_waveforms_and_pga
from bbp_generation import generate_bbp_src, generate_bbp_stl, generate_bbp_input_text
from bbp_execution import run_bbp_simulation
from visualization import generate_display_map, compare_results

# --- CONFIGURATION ---
MIN_MAGNITUDE = 3.0
SEARCH_RADIUS_KM = 300.0
RADIUS_DEG = SEARCH_RADIUS_KM / 111.0
CHECK_INTERVAL_SECONDS = 600  # Check every 10 minutes


def run_live_workflow(event):
    """
    Executes the Dual-Map Display Pipeline for a single real-time event.
    """
    ev_id = event['id']

    # Define run directory using utility function
    run_dir = ensure_run_directory(ev_id)

    # 2. Get Stations & Velocity Model
    stations = get_nearest_stations(event, max_radius_deg=RADIUS_DEG)

    # 3. Get Observed PGAs
    observed_pga_data = get_waveforms_and_pga(event, stations)

    if not observed_pga_data:
        print(f"❌ No valid observed PGA data could be retrieved. Halting validation workflow for event {ev_id}.")
        return

    # --- Data Validation Filter ---
    valid_observed_data = []
    for item in observed_pga_data:
        # Only keep results that are dictionaries AND have the 'pga_g' key
        if isinstance(item, dict) and 'pga_g' in item:
            valid_observed_data.append(item)
        else:
            print(f"    Warning: Observed data contained invalid item: {item}. Discarding.")

    if not valid_observed_data:
        print("❌ All observed PGA data failed validation. Halting workflow.")
        return
    # ------------------------------

    # Note: Observed map will be generated in compare_results() after simulation
    # to ensure both maps use the same color scale

    # 4. Generate Station List File
    stl_file = generate_bbp_stl(event, stations)

    # Get Mechanism & Source File
    mechanism = get_mechanism(ev_id)

    # Mechanism guard (Good Practice)
    if not isinstance(mechanism, dict):
        print("    >> Mechanism failed retrieval. Applying safe default.")
        mechanism = {'strike': 0.0, 'dip': 90.0, 'rake': 0.0, 'source_type': 'FALLBACK'}

    src_file = generate_bbp_src(event, mechanism)

    # 5. Generate BBP Input File
    input_file = generate_bbp_input_text(event, src_file, stl_file)

    # ------------------------------------------------------------------
    # 6. Run Docker
    print(f"    >> Launching BBP Physics Engine...")
    success = run_bbp_simulation(input_file)

    if not success:
        print("❌ BBP Simulation failed.")

    # ------------------------------------------------------------------
    # 7. Compare Results and Generate Map 2
    # 🚨 FIX 2b: Pass the validated data set (valid_observed_data) to comparison
    compare_results(valid_observed_data, ev_id, event)

    print(f"✅ COMPLETED: {ev_id}")
    print(f"   Maps saved to {run_dir}/")

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