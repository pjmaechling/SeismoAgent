import os
from tools import (get_event_details, get_nearest_stations, get_waveforms_and_pga,
                   get_mechanism, generate_bbp_src, generate_bbp_stl,
                   generate_bbp_input_text, run_bbp_simulation, compare_results)

#TARGET_EVENT_ID = "ci15481673" # La Habra M5.1
TARGET_EVENT_ID = "ci38443303" # Example: Ridgecrest M7.1
#TARGET_EVENT_ID = "ci41134895" # Newport Inglewood M3.1

def run_agent_workflow():
    print(f"Agent started (Validation Mode: {TARGET_EVENT_ID}).")

    # 1. Get Event
    target_event = get_event_details(TARGET_EVENT_ID)
    if not target_event: return

    # 2. Source Model
    mechanism = get_mechanism(TARGET_EVENT_ID)
    src_file = generate_bbp_src(target_event, mechanism)

    # 3. Find Stations
    stations = get_nearest_stations(target_event, max_radius_deg=0.5, max_stations=8)
    if not stations: return

    # 4. Get Data
    pga_data = get_waveforms_and_pga(target_event, stations)
    if not pga_data: return

    # 5. Station List
    # --- FIX: Save to outdata/Event_ID/ ---
    stl_path = os.path.join("outdata", f"Event_{TARGET_EVENT_ID}", f"event_{TARGET_EVENT_ID}.stl")
    generate_bbp_stl(target_event,stations, output_file=stl_path)

    # 6. Input Script
    input_file = generate_bbp_input_text(target_event, src_file, stl_path)

    # 7. Run Simulation
    success = run_bbp_simulation(input_file)

    if success:
        print(f"\n✅ AGENT SUCCESS: {TARGET_EVENT_ID}")
    else:
        print("\n❌ AGENT FAILURE: Simulation did not complete.")

    compare_results(pga_data, TARGET_EVENT_ID, target_event)

if __name__ == "__main__":
    run_agent_workflow()