"""
Backward compatibility module - re-exports all functions from the new modular structure.

This module maintains backward compatibility for code that imports from 'tools'.
All functions are now organized into separate modules:
- file_utils: Directory management
- event_retrieval: USGS/ComCat data retrieval
- station_operations: Station finding and waveform processing
- bbp_generation: BBP file generation
- bbp_execution: BBP simulation execution
- visualization: Map generation and result comparison
"""

# Import all functions from the new modules
from file_utils import ensure_run_directory
from event_retrieval import (
    get_recent_quakes,
    get_event_details,
    get_mechanism
)
from station_operations import (
    get_nearest_stations,
    get_waveforms_and_pga
)
from bbp_generation import (
    select_1d_velocity_model,
    calculate_fault_dims,
    generate_bbp_src,
    generate_bbp_stl,
    generate_bbp_input_text
)
from bbp_execution import (
    run_bbp_simulation,
    get_simulated_pgas
)
from visualization import (
    generate_display_map,
    compare_results
)

# Re-export URL_MAPPINGS for backward compatibility
from obspy.clients.fdsn.header import URL_MAPPINGS
URL_MAPPINGS['SCEDC'] = "https://service.scedc.caltech.edu"
URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"

# Export all functions for backward compatibility
__all__ = [
    'ensure_run_directory',
    'get_recent_quakes',
    'get_event_details',
    'get_mechanism',
    'get_nearest_stations',
    'get_waveforms_and_pga',
    'select_1d_velocity_model',
    'calculate_fault_dims',
    'generate_bbp_src',
    'generate_bbp_stl',
    'generate_bbp_input_text',
    'run_bbp_simulation',
    'get_simulated_pgas',
    'generate_display_map',
    'compare_results',
]
