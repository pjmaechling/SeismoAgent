"""
Station finding and waveform processing functions.
"""
import os
import time
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from file_utils import ensure_run_directory


def get_nearest_stations(event, max_radius_deg=1.5, max_stations=40):
    """Finds nearest stations (State-Aware: NCEDC vs SCEDC)."""
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
    else:
        client_name = "SCEDC"

    client = Client(client_name)
    event_time = UTCDateTime(event['time'])

    print(f"--- TOOL: Searching stations ({max_radius_deg:.2f} deg) using {client_name} ---")
    print("Event:", event)
    print("RADIUS_Deg:", max_radius_deg)
    try:
        inventory = client.get_stations(
            latitude=event['latitude'], longitude=event['longitude'],
            maxradius=max_radius_deg, starttime=event_time, endtime=event_time + 600,
            network="CI,NC,BK", channel="HN*,BH*,EH*", level="station"
        )
    except Exception as e:
        print(f"    Station search warning: {e}")
        return []

    station_list = []
    for network in inventory:
        for station in network:
            distance_m, _, _ = gps2dist_azimuth(event['latitude'], event['longitude'], station.latitude,
                                                station.longitude)
            station_list.append({
                "network": network.code,
                "station": station.code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "distance_km": distance_m / 1000.0,
            })

    sorted_stations = sorted(station_list, key=lambda x: x['distance_km'])
    unique_stations = []
    seen = set()
    for s in sorted_stations:
        uid = f"{s['network']}.{s['station']}"
        if uid not in seen:
            unique_stations.append(s)
            seen.add(uid)

    final_list = unique_stations[:max_stations]
    if final_list:
        print(f"    Found {len(final_list)} stations.")
    return final_list


def get_waveforms_and_pga(event, stations):
    """
    Downloads waveforms once per station, calculates PGA, and plots the processed acceleration.
    (REVERTED: Local caching has been removed.)
    """
    event_time = UTCDateTime(event['time'])
    results = []
    # Configuration for this function
    MAX_ATTEMPTS = 1
    BASE_SLEEP = 2.0
    MIN_TRACE_COUNT = 2  # Required for RotD50 calculation

    # Setup directories
    run_dir = ensure_run_directory(event['id'])
    plot_dir = os.path.join(run_dir, "observed_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Client Selection logic
    if event['latitude'] > 36.0:
        client_name = "NCEDC"
    else:
        client_name = "SCEDC"
    client = Client(client_name)

    print(f"--- TOOL: Downloading Waveforms & Plotting to {plot_dir} ---")

    for i, sta in enumerate(stations):
        net = sta['network']
        code = sta['station']

        # Politeness Delay between stations
        time.sleep(BASE_SLEEP)

        for attempt in range(MAX_ATTEMPTS):
            try:
                # 1. Inventory Check
                inventory = client.get_stations(
                    network=net, station=code, starttime=event_time, endtime=event_time + 60,
                    level="channel", channel="HN*,BH*"
                )

                # --- Channel Selection Logic (Find suitable pair) ---
                valid_pair = None
                used_instr = None

                for n in inventory:
                    for s in n:
                        if s.channels:
                            used_instr = s.channels[0].code[:2]
                            valid_pair = (s.channels[0], s.channels[1])
                            break
                    if valid_pair:
                        break

                if not valid_pair:
                    print(f"    -> Warning: No valid channel pair found for {net}.{code}. Skipping.")
                    break  # Move to next station

                chan1, chan2 = valid_pair[0].code, valid_pair[1].code
                location_code = valid_pair[0].location_code
                print(f"  - Attempting: {net}.{code} -> {chan1}/{chan2} (Single Request)")

                # 2. Download Waveforms
                st_filtered = client.get_waveforms(
                    net, code, location_code, f"{used_instr}*",  # Use location_code
                    event_time - 10, event_time + 60, attach_response=True
                )

                # Select only the specific two channels to avoid processing extra traces
                st_filtered = st_filtered.select(channel=chan1) + st_filtered.select(channel=chan2)

                # --- CRITICAL CHECK: Ensure we have both required traces ---
                if len(st_filtered) < MIN_TRACE_COUNT:
                    print(f"    -> Warning: Received only {len(st_filtered)} trace(s) for {net}.{code}. Skipping.")
                    break  # Move to next station

                # 3. Response Removal (Conversion to m/s²)
                st_accel = st_filtered.copy()
                pre_filt = [0.05, 0.1, 35, 40]
                # If removal fails, it will raise an Exception handled below
                st_accel.remove_response(output="ACC", pre_filt=pre_filt, water_level=60)

                # 4. Calculate PGA (using m/s² data)
                # Ensure tr.data is used only if it's a valid numpy array
                try:
                    pgas_m_s2 = [np.max(np.abs(tr.data)) for tr in st_accel if len(tr.data) > 0]
                    if len(pgas_m_s2) < MIN_TRACE_COUNT:
                        print(f"    -> Warning: Trace data failed processing after filtering. Skipping.")
                        break  # Failed data integrity check

                    rotd50_m_s2 = np.sqrt(pgas_m_s2[0] * pgas_m_s2[1])
                    rotd50_g = rotd50_m_s2 / 9.81
                except Exception as pga_e:
                    print(f"    -> Error during PGA calculation for {net}.{code}: {pga_e}. Skipping.")
                    break  # Move to next station

                # 5. Scaling and Plotting (cm/s²)
                plot_filename = os.path.join(plot_dir, f"{net}.{code}_{event['id']}_ACCEL_cms2.png")

                st_plot = st_accel.copy()
                for tr in st_plot:
                    tr.data *= 100.0  # m/s^2 -> cm/s^2
                    tr.stats.unit = "cm/s²"
                    tr.stats.channel = f"ACCEL ({tr.stats.channel.split('_')[0]})"

                st_plot.plot(outfile=plot_filename, show=False)

                # 6. Save PGA Result
                print(f"    -> Success! RotD50={rotd50_m_s2:.4f} m/s²")

                # The crash fix (implicit): Ensure a valid dictionary is ALWAYS appended on success
                results.append({
                    "station": f"{net}.{code}",
                    "latitude": sta['latitude'],
                    "longitude": sta['longitude'],
                    "distance_km": sta['distance_km'],
                    "pga_m_s2": rotd50_m_s2,  # This is a float
                    "pga_g": rotd50_g  # This is a float
                })
                break  # Success break

            except Exception as e:
                # Generalized failure handler for Obspy/Network issues
                if "500" in str(e):
                    print(f"    -> Server busy, skipping station {net}.{code}.")
                else:
                    print(f"    -> Error processing station {net}.{code}: {e}")

                break  # Stop trying and move to next station

    # The returned 'results' list contains only valid dictionaries with float values,
    # preventing the 'dict > float' error in subsequent functions.
    return results

