import time
import datetime
from tools import (get_recent_quakes, get_mechanism, generate_bbp_src,
                   get_nearest_stations, get_waveforms_and_pga,
                   generate_bbp_stl, generate_bbp_input_text,
                   run_bbp_simulation, compare_results, select_1d_velocity_model)  # <--- Added Import

# CONFIGURATION
MIN_MAGNITUDE = 3.0
SEARCH_RADIUS_KM = 200.0
RADIUS_DEG = SEARCH_RADIUS_KM / 111.0
CHECK_INTERVAL_SECONDS = 600  # Check every 10 minutes


def run_pipeline_for_event(event):
    """
    Runs the full SeismoAgent pipeline for a single discovered event.
    SAFETY CHECK: Aborts if Northern California model is selected.
    """
    ev_id = event['id']
    print(f"\n==================================================")
    print(f"🚨 STARTING ANALYSIS: {ev_id} (M{event['magnitude']})")
    print(f"==================================================")

    # 1. Source Modeling
    mechanism = get_mechanism(ev_id)
    src_file = f"event_{ev_id}.src"
    generate_bbp_src(event, mechanism, output_file=src_file)

    # 2. Find Stations (Prints the list automatically)
    stations = get_nearest_stations(event, max_radius_deg=RADIUS_DEG, max_stations=40)

    if not stations:
        print("    Aborting: No digital stations found nearby.")
        return

    # 3. SAFETY CHECK: Check Velocity Model
    # We call the tool to see what it WOULD select
    vel_model_id = select_1d_velocity_model(event['latitude'])

    if vel_model_id == "2":  # 2 = NOCAL
        print("\n    " + "!" * 50)
        print("    NOTICE! Northern California Velocity Model selected.")
        print("    Simulation aborted to prevent errors with LA_BASIN Docker image.")
        print("    " + "!" * 50)
        return

    # --- IF SOUTHERN CA, PROCEED ---

    # 4. Get Data & Plot Observations
    pga_data = get_waveforms_and_pga(event, stations)
    if not pga_data:
        print("    Aborting: Could not retrieve waveforms.")
        return

    # 5. Generate Station List
    stl_file = f"event_{ev_id}.stl"
    generate_bbp_stl(pga_data, output_file=stl_file)

    # 6. Generate BBP Input
    input_file = "bbp_input.txt"
    generate_bbp_input_text(event, src_file, stl_file, output_file=input_file)

    # 7. Run Docker Simulation
    print(f"    >> Launching BBP Physics Engine...")
    success = run_bbp_simulation(input_file)

    if success:
        # 8. Validation
        compare_results(pga_data, ev_id, event)
        print(f"✅ COMPLETED: {ev_id}")
    else:
        print(f"❌ FAILED: {ev_id}")


def start_night_watch():
    print(f"--- SEISMO AGENT: NIGHT WATCH MODE ACTIVATED ---")
    print(f"Trigger: M > {MIN_MAGNITUDE}")
    print(f"Radius:  {SEARCH_RADIUS_KM} km")
    print(f"Status:  Scanning California...")

    processed_ids = set()

    while True:
        try:
            # 1. Scan for recent events
            recent_events = get_recent_quakes(min_magnitude=MIN_MAGNITUDE)

            # 2. Filter for NEW events
            new_events = [e for e in recent_events if e['id'] not in processed_ids]

            if new_events:
                print(f"\n>> Found {len(new_events)} new event(s). Processing the largest...")
                target = new_events[0]

                # Run the job
                run_pipeline_for_event(target)

                # Mark all as processed
                for e in new_events:
                    processed_ids.add(e['id'])

            else:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                # print(f"[{timestamp}] No new events. Sleeping...")

        except Exception as e:
            print(f"CRASH PREVENTION: {e}")

        # 3. Sleep
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_night_watch()