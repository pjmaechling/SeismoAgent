# # findPGAs.py
# import json
# from tools import get_event_details, get_nearest_stations, get_waveforms_and_pga
# # Import new tools
# from tools import get_mechanism, generate_bbp_src
#
# # M6.4 Ferndale Earthquake
# TARGET_EVENT_ID = "nc73821036"
#
# def run_agent_workflow():
#     print("Agent started (Source Generation Mode).")
#
#     # Step 1: Get Basic Details
#     print(f"\n--- STEP 1: Get Details for Event {TARGET_EVENT_ID} ---")
#     target_event = get_event_details(TARGET_EVENT_ID)
#     if not target_event: return
#
#     print(f"Target Event: M{target_event['magnitude']} at {target_event['location']}")
#
#     # Step 2: Get Physics (Mechanism)
#     print(f"\n--- STEP 2: Retrieve Focal Mechanism ---")
#     mechanism = get_mechanism(TARGET_EVENT_ID)
#
#     # Step 3: Generate SCEC BBP Source File
#     print(f"\n--- STEP 3: Generate BBP Input File ---")
#     src_filename = f"event_{TARGET_EVENT_ID}.src"
#
#     generate_bbp_src(target_event, mechanism, output_file=src_filename)
#
#     # Validation
#     print(f"\nSUCCESS: Source file '{src_filename}' is ready for simulation.")
#     print("You can now upload this file to the Broadband Platform.")
#
# if __name__ == "__main__":
#     run_agent_workflow()