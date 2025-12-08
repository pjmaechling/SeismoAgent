import time
import datetime
import os
from tools import (get_recent_quakes, get_mechanism, generate_bbp_src,
                   get_nearest_stations, get_waveforms_and_pga,
                   generate_bbp_stl, generate_bbp_input_text,
                   run_bbp_simulation, compare_results, select_1d_velocity_model)

# CONFIGURATION
MIN_MAGNITUDE = 3.0
SEARCH_RADIUS_KM = 200.0
RADIUS_DEG = SEARCH_RADIUS_KM / 111.0
CHECK_INTERVAL_SECONDS = 600


# night_watch.py (Replace the run_pipeline_for_event function)

def run_pipeline_for_event(event):
    ev_id = event['id']
    print(f"\n==================================================")
    print(f"🚨 STARTING ANALYSIS: {ev_id} (M{event['magnitude']})")
    print(f"==================================================")

    # 1. Source Modeling
    mechanism = get_mechanism(ev_id)
    src_file = generate_bbp_src(event, mechanism)

    # 2. Find Stations
    stations = get_nearest_stations(event, max_radius_deg=RADIUS_DEG, max_stations=40)
    if not stations: return

    # 3. Safety Check
    if select_1d_velocity_model(event['latitude']) == "2":
        print("    NOTICE: Northern CA model selected. Aborting simulation.")
        return

    # 4. Get Data (Handles server busy and saves plots)
    pga_data = get_waveforms_and_pga(event, stations)
    if not pga_data: return

    # 5. Station List
    # --- FIX: Save to outdata/Event_ID/ ---
    stl_filename = f"event_{ev_id}.stl"
    stl_path = os.path.join("outdata", f"Event_{ev_id}", stl_filename)
    generate_bbp_stl(pga_data, output_file=stl_path)

    # 6. Input Script
    # Pass the full stl_path to the input generator
    input_file = generate_bbp_input_text(event, src_file, stl_path)

    # 7. Run Simulation
    success = run_bbp_simulation(input_file)

    if success:
        compare_results(pga_data, ev_id, event)
        print(f"✅ COMPLETED: {ev_id}")
    else:
        print(f"❌ FAILED: {ev_id}")

def start_night_watch():
    print(f"--- SEISMO AGENT: NIGHT WATCH MODE ACTIVATED ---")
    print(f"Trigger: M > {MIN_MAGNITUDE}")
    processed_ids = set()

    while True:
        try:
            recent_events = get_recent_quakes(min_magnitude=MIN_MAGNITUDE)
            new_events = [e for e in recent_events if e['id'] not in processed_ids]

            if new_events:
                print(f"\n>> Found {len(new_events)} new event(s).")
                run_pipeline_for_event(new_events[0])
                for e in new_events: processed_ids.add(e['id'])
            else:
                pass  # Silent sleep

        except Exception as e:
            print(f"CRASH PREVENTION: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_night_watch()