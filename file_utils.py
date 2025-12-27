"""
File and directory utility functions.
"""
import os


def ensure_run_directory(event_id, base_dir="outdata"):
    """
    Creates a unique directory for the event run if it doesn't exist.
    Returns the path to that directory.
    """
    # Structure: outdata/Event_ci12345678
    run_dir = os.path.join(base_dir, f"Event_{event_id}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir

