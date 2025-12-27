"""
Visualization and result comparison functions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextily as cx
from bbp_execution import get_simulated_pgas


def generate_display_map(pga_data, event_data, filename, run_dir, title, vmax=None):
    """
    Generates a map showing PGA data (either observed or simulated)
    at station locations, with added geographic context (basemap).
    
    Parameters:
    -----------
    pga_data : list of dicts
        List containing PGA data dictionaries with keys 'pga_g', 'latitude', 'longitude'
    event_data : dict
        Event metadata dictionary with keys 'latitude', 'longitude', 'magnitude', 'title'
    filename : str
        Output filename for the map
    run_dir : str
        Directory to save the map
    title : str
        Title for the map
    vmax : float, optional
        Maximum value for the color scale. If None, uses the maximum value in pga_data.
    """
    # 1. Setup Data for Plotting
    # Convert list of dicts to DataFrame for easy manipulation
    df = pd.DataFrame(pga_data)

    # Calculate PGA in cm/s² for visualization
    df['pgas_cm_s2'] = df['pga_g'] * 981.0

    # 2. Create the Plot and Axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate map extent with small padding
    pad = 0.05
    lat_min, lat_max = df['latitude'].min() - pad, df['latitude'].max() + pad
    lon_min, lon_max = df['longitude'].min() - pad, df['longitude'].max() + pad

    # Determine vmax if not provided (use data max as default)
    if vmax is None:
        vmax = df['pgas_cm_s2'].max()
    # If vmax is provided, use it directly (assumed to be in cm/s² units)

    # 3. Plot Data (in native Lat/Lon: EPSG:4326)
    scatter = ax.scatter(
        df['longitude'],
        df['latitude'],
        c=df['pgas_cm_s2'],
        cmap='viridis',
        s=150,  # Slightly larger for visibility over basemap
        edgecolor='black',
        alpha=0.8,
        vmin=0,
        vmax=vmax
    )

    # Plot Earthquake Epicenter
    ax.plot(
        event_data['longitude'],
        event_data['latitude'],
        'r*',
        markersize=25,
        markeredgecolor='black',
        label=f"Epicenter (M{event_data['magnitude']})"
    )

    # Set Axes Limits (Crucial before adding basemap)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # 4. Add Geographic Context Layer (Basemap)
    # This automatically converts the axis to Web Mercator (EPSG:3857) and overlays the tiles.
    cx.add_basemap(
        ax,
        crs='EPSG:4326',  # Tell contextily that the data/limits are WGS84 (Lat/Lon)
        source=cx.providers.OpenStreetMap.Mapnik,  # Use a standard, reliable OSM provider
        zoom='auto'
    )

    # 5. Final Touches
    # Use the 'title' from event_data for a full description (assuming you fixed the crash)
    full_title = f"{event_data.get('title', 'Unknown Event')}\n{title}"

    plt.colorbar(scatter, label=f"PGA (cm/s²)", orientation='vertical')
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(full_title)
    ax.legend()
    # The basemap provides context, so the grid is less critical and removed here

    # 6. Save the File
    output_path = os.path.join(run_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Increased DPI for map quality
    plt.close(fig)
    print(f"    Map saved to: {output_path}")

    return output_path


def compare_results(pga_data, event_id, event_data, output_dir="outdata"):
    """
    Compares observed PGA vs simulated PGA, generates both maps with consistent color scales,
    and calculates validation residuals.
    """
    # CRITICAL: Define the run directory once at the top
    run_dir = os.path.join(output_dir, f"Event_{event_id}")

    print("\n--- TOOL: Generating Simulated Maps & Calculating Residuals ---", run_dir)

    # 1. Get Simulated PGAs
    sim_data = get_simulated_pgas(pga_data, event_id, output_dir=run_dir)

    if not sim_data:
        print("    No simulated PGA data found to compare. Cannot proceed.")
        return False

    # 2. Calculate maximum PGA value from both observed and simulated data
    # Convert to cm/s² for consistent comparison
    obs_pgas_cm_s2 = [d['pga_g'] * 981.0 for d in pga_data]
    sim_pgas_cm_s2 = [d['pga_g'] * 981.0 for d in sim_data]
    
    max_obs = max(obs_pgas_cm_s2) if obs_pgas_cm_s2 else 0
    max_sim = max(sim_pgas_cm_s2) if sim_pgas_cm_s2 else 0
    vmax_unified = max(max_obs, max_sim)
    
    print(f"    Unified color scale: max PGA = {vmax_unified:.2f} cm/s²")
    print(f"      (Observed max: {max_obs:.2f} cm/s², Simulated max: {max_sim:.2f} cm/s²)")

    # 3. Generate the map for OBSERVED data with unified color scale
    obs_map_filename = f"display_obs_map_pga_{event_id}.png"
    
    generate_display_map(
        pga_data=pga_data,
        event_data=event_data,
        filename=obs_map_filename,
        run_dir=run_dir,
        title=f"Observed PGA (RotD50)",
        vmax=vmax_unified
    )

    # 4. Generate the map for SIMULATED data with same unified color scale
    sim_map_filename = f"display_sim_map_pga_{event_id}.png"

    generate_display_map(
        pga_data=sim_data,
        event_data=event_data,  # <-- CORRECT: Pass the main event metadata dict
        filename=sim_map_filename,
        run_dir=run_dir,
        title=f"BBP Simulated PGA (GP Method)",
        vmax=vmax_unified
    )

    # 5. Calculate Residuals (Validation Metrics)

    # Ensure both lists are sorted/aligned based on station code
    obs_data_dict = {d['station'].split('.')[1]: d['pga_g'] for d in pga_data}
    sim_data_dict = {d['station']: d['pga_g'] for d in sim_data}

    stations_to_compare = sorted(list(obs_data_dict.keys() & sim_data_dict.keys()))

    residuals = []

    # Calculate Residual: log10(Simulated) - log10(Observed)
    print("\n| Station | Observed PGA (g) | Simulated PGA (g) | Residual (log10)|")
    print("|:--- |:---:|:---:|:---:|")

    for station in stations_to_compare:
        obs_pga = obs_data_dict[station]
        sim_pga = sim_data_dict[station]

        # Calculate log residual only if both values are greater than zero
        if obs_pga > 0 and sim_pga > 0:
            residual = np.log10(sim_pga) - np.log10(obs_pga)
            residuals.append(residual)
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | {residual:.3f} |")
        else:
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | N/A (Zero Value) |")

    # 6. Final Summary
    if residuals:
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        print(f"\n✅ Validation Summary:")
        print(f"   Mean Residual (Bias): {mean_residual:.3f} (Lower bias is better)")
        print(f"   Std. Dev. Residual: {std_residual:.3f} (Lower scatter is better)")

    return True

