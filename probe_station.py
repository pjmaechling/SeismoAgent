from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# SWITCH to the local California Data Center
client = Client("NCEDC")

t = UTCDateTime("2022-12-20T10:34:25")
net = "NP"
sta = "1725"

print(f"--- Probing {net}.{sta} at NCEDC ---")

# 1. Ask for availability (Level='response' tells us exactly what channels exist)
try:
    inventory = client.get_stations(network=net, station=sta,
                                    starttime=t, endtime=t+60,
                                    level="response")
    print("Found channels:")
    for net_obj in inventory:
        for sta_obj in net_obj:
            for chan in sta_obj:
                print(f" - {chan.location_code}.{chan.code} (Sample Rate: {chan.sample_rate})")
except Exception as e:
    print(f"Error: {e}")

# 2. Try a test download of the first available channel
print("\nAttempting download...")
try:
    st = client.get_waveforms(net, sta, "*", "HNZ", t, t+10)
    print("SUCCESS! Downloaded trace:")
    print(st)
except Exception as e:
    print(f"Download failed: {e}")