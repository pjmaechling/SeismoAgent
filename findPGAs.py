# findPGAs.py
import json
from tools import get_event_details, get_nearest_stations, get_waveforms_and_pga

# HARDCODED TARGET: The M6.4 Ferndale Earthquake (Dec 20, 2022)
# This event has excellent strong motion data.
TARGET_EVENT_ID = "nc73821036"


def run_agent_workflow():
    print("Agent started (Historic Mode).")

    # Step 1: Get Historic Quake Details
    print(f"\n--- STEP 1: Get Details for Historic Event {TARGET_EVENT_ID} ---")
    target_event = get_event_details(TARGET_EVENT_ID)

    if not target_event:
        print("Could not find event details.")
        return

    print(f"Target Event: M{target_event['magnitude']} at {target_event['location']}")
    print(f"Time: {target_event['time']}")

    # Step 2: Find Stations
    print("\n--- STEP 2: Find Nearest Stations ---")
    # We increase radius to 1.0 degree (~111km) to ensure we get good regional coverage
    stations = get_nearest_stations(target_event, max_radius_deg=1.0)

    if not stations:
        print("No stations found.")
        return

    print(f"Found {len(stations)} stations to query.")
    for s in stations:
        print(f" - {s['network']}.{s['station']} ({s['distance_km']:.1f} km)")

    # Step 3: Get Data & Calculate PGA
    print("\n--- STEP 3: Analyze Ground Motion ---")
    pga_data = get_waveforms_and_pga(target_event, stations)

    if pga_data:
        print("\n--- FINAL REPORT: Largest Ground Motions ---")
        # Sort by strongest shaking (PGA)
        pga_data.sort(key=lambda x: x['pga_m_s2'], reverse=True)

        for entry in pga_data:
            print(f"Station {entry['station']} ({entry['distance_km']:.1f} km): "
                  f"{entry['pga_m_s2']:.4f} m/s² ({entry['pga_g'] * 100:.2f}% g)")

        # Optional: Save to file
        with open("earthquake_report.json", "w") as f:
            json.dump(pga_data, f, indent=2)
            print("\nSaved full analysis to earthquake_report.json")
    else:
        print("Could not retrieve waveforms for any station.")


if __name__ == "__main__":
    run_agent_workflow()