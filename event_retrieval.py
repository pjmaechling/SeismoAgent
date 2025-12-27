"""
USGS/ComCat event data retrieval functions.
"""
import datetime
from obspy.clients.fdsn.header import URL_MAPPINGS
from libcomcat.search import search, get_event_by_id

# Fix for NCEDC redirects
URL_MAPPINGS['SCEDC'] = "https://service.scedc.caltech.edu"
URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"


def get_recent_quakes(min_magnitude=3.0, lookback_hours=24):
    """
    Searches USGS ComCat for recent earthquakes in California.
    """
    start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=lookback_hours)
    print(f"--- TOOL: Scanning USGS for M{min_magnitude}+ CA events since {start_time.strftime('%H:%M')} UTC ---")

    try:
        events = search(starttime=start_time, minmagnitude=min_magnitude,
                        maxlatitude=42.0, minlatitude=32.0,
                        maxlongitude=-114.0, minlongitude=-125.0)
    except Exception as e:
        print(f"    USGS Search Error: {e}")
        return []

    structured_events = []
    if events:
        sorted_events = sorted(events, key=lambda x: x.magnitude, reverse=True)
        for event in sorted_events:
            structured_events.append({
                "id": event.id,
                "magnitude": event.magnitude,
                "time": event.time.isoformat(),
                "location": event.location,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth
            })
    return structured_events


def get_event_details(event_id):
    """Retrieves details for a specific earthquake ID."""
    print(f"--- TOOL: Retrieving details for Event ID: {event_id} ---")
    try:
        event = get_event_by_id(event_id)
        return {
            "id": event.id,
            "magnitude": event.magnitude,
            "time": event.time.isoformat(),
            "location": event.location,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth
        }
    except Exception as e:
        print(f"Error finding event {event_id}: {e}")
        return None


def get_mechanism(event_id):
    """Fetches Focal Mechanism (Strike/Dip/Rake)."""
    print(f"--- TOOL: Searching for Focal Mechanism/Moment Tensor ---")
    try:
        detail = get_event_by_id(event_id)
        try:
            mt = detail.getProducts('moment-tensor')
        except Exception:
            mt = None
        if not mt:
            try:
                mt = detail.getProducts('focal-mechanism')
            except Exception:
                mt = None

        if not mt:
            print("    No mechanism product found. Using generic defaults.")
            return {"strike": 0, "dip": 90, "rake": 0}

        tensor = mt[0]
        props = tensor.properties
        s = float(props.get('nodal-plane-1-strike') or props.get('np1_strike') or 0)
        d = float(props.get('nodal-plane-1-dip') or props.get('np1_dip') or 90)
        r = float(props.get('nodal-plane-1-rake') or props.get('np1_rake') or 0)

        print(f"    Found Mechanism: Strike={s}, Dip={d}, Rake={r}")
        return {"strike": s, "dip": d, "rake": r}
    except Exception as e:
        print(f"    Warning: Could not retrieve mechanism ({e}). Using defaults.")
        return {"strike": 0, "dip": 90, "rake": 0}

