# tools.py
import datetime
from libcomcat.search import search
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from libcomcat.search import get_event_by_id

# tools.py (Append this to the bottom)

import numpy as np

# tools.py
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

import math
from libcomcat.search import get_event_by_id


# ... (keep existing imports and functions) ...

def get_mechanism(event_id):
    """
    Fetches the best available Focal Mechanism (Strike/Dip/Rake) from USGS.
    Returns a dictionary or defaults if none found.
    """
    print(f"--- TOOL: Searching for Focal Mechanism/Moment Tensor ---")
    try:
        detail = get_event_by_id(event_id)

        # Try to get the preferred moment tensor product
        mt = detail.getProducts('moment-tensor')
        if not mt:
            mt = detail.getProducts('focal-mechanism')

        if mt:
            # Parse the first available tensor
            tensor = mt[0]
            # libcomcat usually hides these deep in properties,
            # but often 'np1_strike' etc are in the product properties
            props = tensor.properties

            # Extract Nodal Plane 1 (Standard convention)
            s = float(props.get('nodal-plane-1-strike', 0))
            d = float(props.get('nodal-plane-1-dip', 90))
            r = float(props.get('nodal-plane-1-rake', 0))

            print(f"    Found Mechanism: Strike={s}, Dip={d}, Rake={r}")
            return {"strike": s, "dip": d, "rake": r}

    except Exception as e:
        print(f"    Warning: Could not retrieve mechanism ({e}).")

    print("    No mechanism found. Using generic California Strike-Slip defaults.")
    return {"strike": 0, "dip": 90, "rake": 0}


def calculate_fault_dims(magnitude):
    """
    Estimates Fault Length and Width using Wells & Coppersmith (1994)
    Empirical relationships for 'All' slip types.
    """
    # Rupture Area (A) = 10 ^ (-3.49 + 0.91 * Mw)
    area = 10 ** (-3.49 + 0.91 * magnitude)

    # Rupture Width (W) = 10 ^ (-1.01 + 0.32 * Mw)
    width = 10 ** (-1.01 + 0.32 * magnitude)

    # Ensure Width doesn't exceed seismogenic thickness (e.g. 15-20km)
    if width > 20.0:
        width = 20.0

    length = area / width

    return length, width


def generate_bbp_src(event_data, mechanism, output_file="event.src"):
    """
    Generates a SCEC Broadband Platform .src file.
    Assumes the USGS hypocenter is the CENTROID of the rupture.
    """
    print(f"--- TOOL: Generating BBP Source File: {output_file} ---")

    mag = event_data['magnitude']
    hypo_lat = event_data['latitude']
    hypo_lon = event_data['longitude']
    hypo_depth = event_data['depth_km']

    strike = mechanism['strike']
    dip = mechanism['dip']
    rake = mechanism['rake']

    # 1. Calculate Dimensions (Scaling Law)
    length, width = calculate_fault_dims(mag)

    # 2. Geometry Calculations
    # We assume the Hypocenter is the Centroid (Center of the fault plane)
    # So Hypo is at 0.0 along strike, and W/2 down dip
    hypo_along_strike = 0.0  # Center
    hypo_down_dip = width / 2.0

    # Calculate Depth to Top of Fault
    # Z_top = Z_hypo - (Dist_Up_Dip * sin(dip))
    # Dist_Up_Dip is half the width
    rad_dip = math.radians(dip)
    depth_to_top = hypo_depth - (hypo_down_dip * math.sin(rad_dip))

    # Safety: Don't let the fault float above the ground
    if depth_to_top < 0:
        print("    Adjustment: Fault top projected above ground. Clamping to 0.")
        depth_to_top = 0.0
        # Re-adjust hypocenter down dip to fit
        hypo_down_dip = hypo_depth / math.sin(rad_dip)

    # Calculate Top Center Lat/Lon
    # If the fault dips, the top edge is horizontally offset from the hypocenter
    # Horizontal offset = (Width/2) * cos(dip)
    # Direction = Strike - 90 degrees (Up dip direction is perpendicular to strike)

    offset_dist_km = (width / 2.0) * math.cos(rad_dip)

    # Simple Flat Earth approximation for small offsets (sufficient for src generation)
    # 1 deg lat ~= 111 km
    # 1 deg lon ~= 111 km * cos(lat)
    if offset_dist_km > 0.001:
        azimuth_rad = math.radians(strike - 90)
        d_lat = (offset_dist_km * math.cos(azimuth_rad)) / 111.0
        d_lon = (offset_dist_km * math.sin(azimuth_rad)) / (111.0 * math.cos(math.radians(hypo_lat)))

        lat_top_center = hypo_lat + d_lat
        lon_top_center = hypo_lon + d_lon
    else:
        lat_top_center = hypo_lat
        lon_top_center = hypo_lon

    # 3. Write File
    with open(output_file, "w") as f:
        f.write(f"MAGNITUDE = {mag:.2f}\n")
        f.write(f"FAULT_LENGTH = {length:.2f}\n")
        f.write(f"FAULT_WIDTH = {width:.2f}\n")
        f.write(f"DEPTH_TO_TOP = {depth_to_top:.2f}\n")
        f.write(f"STRIKE = {strike:.1f}\n")
        f.write(f"DIP = {dip:.1f}\n")
        f.write(f"RAKE = {rake:.1f}\n")
        f.write(f"LAT_TOP_CENTER = {lat_top_center:.5f}\n")
        f.write(f"LON_TOP_CENTER = {lon_top_center:.5f}\n")
        f.write(f"HYPO_ALONG_STRIKE = {hypo_along_strike:.1f}\n")
        f.write(f"HYPO_DOWN_DIP = {hypo_down_dip:.2f}\n")
        f.write(f"SEED = 12345\n")  # Random seed for stochastic parts

    print(f"    File written. L={length:.1f}km, W={width:.1f}km, Ztop={depth_to_top:.1f}km")
    return output_file

# ... (keep other functions like get_event_details, get_nearest_stations as is) ...

# tools.py
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
# FIX for NCEDC Redirect Issue (common in older ObsPy versions)
from obspy.clients.fdsn.header import URL_MAPPINGS

URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"


def get_waveforms_and_pga(event, stations):
    """
    Robust waveform downloader that tries multiple Data Centers and
    Specific Location Codes to bypass 'No Data' errors.
    """
    event_time = UTCDateTime(event['time'])
    results = []

    # PRIORITY LIST:
    # 1. NCEDC (Home of Northern CA data)
    # 2. IRIS (Global backup)
    # 3. SCEDC (Sometimes holds border data)
    client_names = ["NCEDC", "IRIS", "SCEDC"]

    print(f"\n--- TOOL: Downloading Waveforms (Robust Mode) ---")

    for sta in stations:
        net = sta['network']
        code = sta['station']

        found_trace = None
        used_provider = None

        # 1. Loop through Data Centers
        for provider in client_names:
            if found_trace: break

            try:
                client = Client(provider)

                # 2. ASK FOR METADATA FIRST (The "Menu")
                # We specifically look for Strong Motion (HN*) and Broadband (BH*)
                inventory = client.get_stations(
                    network=net, station=code,
                    starttime=event_time, endtime=event_time + 60,
                    level="response", channel="HN*,BH*"
                )

                # 3. ITERATE THROUGH EXACT MENU ITEMS
                # Instead of guessing "*", we request the EXACT item on the menu.
                for net_obj in inventory:
                    for sta_obj in net_obj:
                        for chan in sta_obj:

                            # Skip if we already found data
                            if found_trace: break

                            loc_code = chan.location_code  # e.g., "10", "00", ""
                            chan_code = chan.code  # e.g., "HNZ"

                            try:
                                # Download specific stream
                                st = client.get_waveforms(
                                    net, code, loc_code, chan_code,
                                    event_time - 10, event_time + 60,
                                    attach_response=True
                                )

                                # Check if trace actually has data points
                                if len(st) > 0 and st[0].stats.npts > 10:
                                    found_trace = st[0]
                                    used_provider = provider
                                    # print(f"    -> Got it from {provider}: {loc_code}.{chan_code}")
                            except Exception:
                                continue  # Try next channel

            except Exception:
                continue  # Try next provider

        if not found_trace:
            print(f"  - {net}.{code}: Failed to retrieve data from any center.")
            continue

        # 4. PHYSICS PROCESSING (Calculate G-Force)
        try:
            tr = found_trace

            # Remove Instrument Response (Counts -> m/s^2)
            # We use a conservative filter to avoid blowing up noise
            pre_filt = [0.05, 0.1, 40, 45]

            tr.remove_response(output="ACC", pre_filt=pre_filt, water_level=60)

            pga_m_s2 = np.max(np.abs(tr.data))
            pga_g = pga_m_s2 / 9.81

            print(f"  - {tr.id} [{used_provider}] | PGA={pga_m_s2:.4f} m/s² ({pga_g * 100:.2f}% g)")

            results.append({
                "station": tr.id,
                "distance_km": sta['distance_km'],
                "pga_m_s2": pga_m_s2,
                "pga_g": pga_g,
                "provider": used_provider
            })

        except Exception as e:
            print(f"  - {net}.{code}: Error processing physics ({e})")

    return results

# tools.py
from libcomcat.search import get_event_by_id  # <--- Import this


def get_event_details(event_id):
    """
    Retrieves details for a specific historic earthquake by its USGS ID.
    """
    print(f"--- TOOL: Retrieving details for Event ID: {event_id} ---")

    try:
        event = get_event_by_id(event_id)

        return {
            "id": event.id,
            "magnitude": event.magnitude,
            # Use isoformat() for ObsPy compatibility
            "time": event.time.isoformat(),
            "location": event.location,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth
        }
    except Exception as e:
        print(f"Error finding event {event_id}: {e}")
        return None

def get_recent_quakes(days_back=1, min_magnitude=2.0):
    """
    Searches USGS ComCat for recent earthquakes in the California region.
    Returns a list of dictionaries with key event details.
    """
    # Use utcnow() to get a naive datetime (no timezone info)
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=days_back)

    print(f"--- TOOL: Searching USGS for M{min_magnitude}+ events since {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        events = search(
            starttime=start_time,
            minmagnitude=min_magnitude,
            maxlatitude=42.0, minlatitude=32.0,
            maxlongitude=-114.0, minlongitude=-125.0
        )
    except Exception as e:
        print(f"Error during USGS search: {e}")
        return []

    structured_events = []

    if not events:
        return []

    # Sort by magnitude
    sorted_events = sorted(events, key=lambda x: x.magnitude, reverse=True)

    for event in sorted_events[:5]:
        structured_events.append({
            "id": event.id,
            "magnitude": event.magnitude,
            # FIX: Use isoformat() so ObsPy can read it later
            "time": event.time.isoformat(),
            "location": event.location,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth
        })

    return structured_events


def get_nearest_stations(event, max_radius_deg=1.5, max_stations=5):
    """
    Finds the nearest seismic stations to the event using IRIS.
    """
    client = Client("IRIS")

    event_time = UTCDateTime(event['time'])
    lat = event['latitude']
    lon = event['longitude']

    print(f"--- TOOL: Searching for stations within {max_radius_deg} deg of {lat:.3f}, {lon:.3f} ---")

    try:
        # WIDENED SEARCH:
        # 1. Radius increased to max_radius_deg (default 1.5)
        # 2. Channels added: EHZ (Short Period), SHZ (Short Period)
        # 3. REMOVED: matchtimeseries=True (This is often too strict for real-time)
        inventory = client.get_stations(
            latitude=lat, longitude=lon, maxradius=max_radius_deg,
            starttime=event_time, endtime=event_time + 600,
            channel="BHZ,HHZ,HNZ,EHZ,SHZ", level="station"
        )
    except Exception as e:
        # If it still fails, print the error but don't crash
        print(f"Station search warning: {e}")
        return []

    station_list = []

    for network in inventory:
        for station in network:
            distance_m, azimuth, _ = gps2dist_azimuth(lat, lon, station.latitude, station.longitude)

            station_list.append({
                "network": network.code,
                "station": station.code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "distance_km": distance_m / 1000.0,
                # We save the channel info so we know what to download later
                "channel": station[0].code if station else "UNK"
            })

    # Sort by distance
    sorted_stations = sorted(station_list, key=lambda x: x['distance_km'])

    # Simple deduplication (sometimes networks list the same station twice)
    unique_stations = []
    seen = set()
    for s in sorted_stations:
        uid = f"{s['network']}.{s['station']}"
        if uid not in seen:
            unique_stations.append(s)
            seen.add(uid)

    return unique_stations[:max_stations]