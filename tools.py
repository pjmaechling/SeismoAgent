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