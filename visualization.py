"""
Visualization and result comparison functions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextily as cx
from scipy import stats
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


def generate_residual_map(residual_data, event_data, filename, run_dir):
    """
    Generates a geographic map showing residuals at station locations.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing residual data with keys: 'station', 'residual', 'latitude', 'longitude'
    event_data : dict
        Event metadata dictionary
    filename : str
        Output filename
    run_dir : str
        Directory to save the map
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate map extent
    pad = 0.05
    lat_min, lat_max = df['latitude'].min() - pad, df['latitude'].max() + pad
    lon_min, lon_max = df['longitude'].min() - pad, df['longitude'].max() + pad
    
    # Use diverging colormap centered at zero
    vmax = max(abs(df['residual'].min()), abs(df['residual'].max()))
    vmin = -vmax
    
    scatter = ax.scatter(
        df['longitude'],
        df['latitude'],
        c=df['residual'],
        cmap='RdBu_r',
        s=200,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax
    )
    
    # Plot epicenter
    ax.plot(
        event_data['longitude'],
        event_data['latitude'],
        'r*',
        markersize=25,
        markeredgecolor='black',
        label=f"Epicenter (M{event_data['magnitude']})"
    )
    
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # Add basemap
    cx.add_basemap(
        ax,
        crs='EPSG:4326',
        source=cx.providers.OpenStreetMap.Mapnik,
        zoom='auto'
    )
    
    cbar = plt.colorbar(scatter, label="Residual (log₁₀)", orientation='vertical')
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"{event_data.get('title', 'Unknown Event')}\nResidual Map (log₁₀(Sim) - log₁₀(Obs))")
    ax.legend()
    
    output_path = os.path.join(run_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Residual map saved to: {output_path}")
    
    return output_path


def generate_site_specific_residual_chart(residual_data, event_id, event_data, run_dir):
    """
    Generates a horizontal bar chart showing residuals for each station.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing residual data with keys: 'station', 'residual'
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the chart
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    df = df.sort_values('residual')
    
    # Calculate statistics
    mean_res = df['residual'].mean()
    std_res = df['residual'].std()
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
    
    # Color bars based on residual sign
    colors = ['red' if r > 0 else 'blue' for r in df['residual']]
    
    bars = ax.barh(df['station'], df['residual'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add reference lines
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Residual')
    ax.axvline(mean_res, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.3f}')
    ax.axvline(mean_res + std_res, color='orange', linestyle=':', linewidth=1.5, label=f'Mean + 1σ: {mean_res + std_res:.3f}')
    ax.axvline(mean_res - std_res, color='orange', linestyle=':', linewidth=1.5, label=f'Mean - 1σ: {mean_res - std_res:.3f}')
    ax.axvline(mean_res + 2*std_res, color='purple', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean + 2σ: {mean_res + 2*std_res:.3f}')
    ax.axvline(mean_res - 2*std_res, color='purple', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean - 2σ: {mean_res - 2*std_res:.3f}')
    
    ax.set_xlabel('Residual (log₁₀(Sim) - log₁₀(Obs))')
    ax.set_ylabel('Station')
    ax.set_title(f"{event_data.get('title', 'Unknown Event')}\nSite-Specific Residuals")
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    output_path = os.path.join(run_dir, f"residual_bar_chart_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Site-specific residual chart saved to: {output_path}")
    
    return output_path


def generate_observed_vs_simulated_scatter(residual_data, event_id, event_data, run_dir):
    """
    Generates a scatter plot of observed vs simulated PGA with 1:1 line.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing data with keys: 'station', 'obs_pga', 'sim_pga', 'residual'
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the plot
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    df = df[(df['obs_pga'] > 0) & (df['sim_pga'] > 0)]
    
    if len(df) == 0:
        return None
    
    obs_log = np.log10(df['obs_pga'])
    sim_log = np.log10(df['sim_pga'])
    
    # Calculate statistics
    r_squared = np.corrcoef(obs_log, sim_log)[0, 1]**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs_log, sim_log)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by residual magnitude
    scatter = ax.scatter(
        obs_log,
        sim_log,
        c=df['residual'],
        cmap='RdBu_r',
        s=100,
        edgecolor='black',
        alpha=0.7,
        vmin=-max(abs(df['residual'])),
        vmax=max(abs(df['residual']))
    )
    
    # 1:1 line
    min_val = min(obs_log.min(), sim_log.min())
    max_val = max(obs_log.max(), sim_log.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line (Perfect Match)')
    
    # Regression line
    x_reg = np.array([min_val, max_val])
    y_reg = slope * x_reg + intercept
    ax.plot(x_reg, y_reg, 'g-', linewidth=2, alpha=0.7, label=f'Regression: y={slope:.3f}x+{intercept:.3f}')
    
    cbar = plt.colorbar(scatter, label="Residual (log₁₀)")
    ax.set_xlabel('Observed PGA (log₁₀(g))')
    ax.set_ylabel('Simulated PGA (log₁₀(g))')
    ax.set_title(f"{event_data.get('title', 'Unknown Event')}\nObserved vs Simulated PGA\nR² = {r_squared:.3f}, Slope = {slope:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    output_path = os.path.join(run_dir, f"obs_vs_sim_scatter_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Observed vs Simulated scatter plot saved to: {output_path}")
    
    return output_path


def generate_residual_distribution_plots(residual_data, event_id, event_data, run_dir):
    """
    Generates histogram, box plot, and Q-Q plot of residuals.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing residual data with keys: 'residual'
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the plots
    """
    if not residual_data:
        return None
    
    residuals = [d['residual'] for d in residual_data]
    residuals = np.array(residuals)
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    n = len(residuals)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Histogram
    ax1 = plt.subplot(1, 3, 1)
    n_bins = min(20, max(5, int(np.sqrt(n))))
    ax1.hist(residuals, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Overlay normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mean_res, std_res) * n * (residuals.max() - residuals.min()) / n_bins
    ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal(μ={mean_res:.3f}, σ={std_res:.3f})')
    
    ax1.axvline(mean_res, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.3f}')
    ax1.set_xlabel('Residual (log₁₀)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Residual Histogram')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2 = plt.subplot(1, 3, 2)
    bp = ax2.boxplot(residuals, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    ax2.axhline(mean_res, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.3f}')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, label='Zero')
    ax2.set_ylabel('Residual (log₁₀)')
    ax2.set_title('Residual Box Plot')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Q-Q plot
    ax3 = plt.subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    ax3.grid(alpha=0.3)
    
    plt.suptitle(f"{event_data.get('title', 'Unknown Event')}\nResidual Distribution (N={n}, μ={mean_res:.3f}, σ={std_res:.3f})", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(run_dir, f"residual_distribution_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Residual distribution plots saved to: {output_path}")
    
    return output_path


def generate_residual_vs_distance_plot(residual_data, event_id, event_data, run_dir):
    """
    Generates a plot of residuals vs distance from epicenter.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing data with keys: 'residual', 'distance_km', 'obs_pga'
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the plot
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by observed PGA magnitude
    scatter = ax.scatter(
        df['distance_km'],
        df['residual'],
        c=np.log10(df['obs_pga']),
        cmap='viridis',
        s=100,
        edgecolor='black',
        alpha=0.7
    )
    
    # Add trend line if enough points
    if len(df) > 2:
        z = np.polyfit(df['distance_km'], df['residual'], 1)
        p = np.poly1d(z)
        ax.plot(df['distance_km'], p(df['distance_km']), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.6f}x+{z[1]:.3f}')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, label='Zero Residual')
    ax.axhline(df['residual'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["residual"].mean():.3f}')
    
    cbar = plt.colorbar(scatter, label="log₁₀(Observed PGA)")
    ax.set_xlabel('Distance from Epicenter (km)')
    ax.set_ylabel('Residual (log₁₀(Sim) - log₁₀(Obs))')
    ax.set_title(f"{event_data.get('title', 'Unknown Event')}\nResidual vs Distance from Epicenter")
    ax.grid(alpha=0.3)
    ax.legend()
    
    output_path = os.path.join(run_dir, f"residual_vs_distance_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Residual vs Distance plot saved to: {output_path}")
    
    return output_path


def generate_residual_vs_observed_pga_plot(residual_data, event_id, event_data, run_dir):
    """
    Generates a plot of residuals vs observed PGA magnitude.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing data with keys: 'residual', 'obs_pga', 'distance_km'
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the plot
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    df = df[df['obs_pga'] > 0]  # Filter out zeros
    
    if len(df) == 0:
        return None
    
    df['obs_pga_log'] = np.log10(df['obs_pga'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by distance
    scatter = ax.scatter(
        df['obs_pga_log'],
        df['residual'],
        c=df['distance_km'],
        cmap='plasma',
        s=100,
        edgecolor='black',
        alpha=0.7
    )
    
    # Add trend line if enough points
    if len(df) > 2:
        z = np.polyfit(df['obs_pga_log'], df['residual'], 1)
        p = np.poly1d(z)
        ax.plot(df['obs_pga_log'], p(df['obs_pga_log']), "r--", alpha=0.8, linewidth=2,
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, label='Zero Residual')
    ax.axhline(df['residual'].mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {df["residual"].mean():.3f}')
    
    cbar = plt.colorbar(scatter, label="Distance from Epicenter (km)")
    ax.set_xlabel('Observed PGA (log₁₀(g))')
    ax.set_ylabel('Residual (log₁₀(Sim) - log₁₀(Obs))')
    ax.set_title(f"{event_data.get('title', 'Unknown Event')}\nResidual vs Observed PGA")
    ax.grid(alpha=0.3)
    ax.legend()
    
    output_path = os.path.join(run_dir, f"residual_vs_observed_pga_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Residual vs Observed PGA plot saved to: {output_path}")
    
    return output_path


def generate_residual_summary_dashboard(residual_data, event_id, event_data, run_dir):
    """
    Generates a multi-panel dashboard combining residual map, histogram, and scatter plot.
    
    Parameters:
    -----------
    residual_data : list of dicts
        List containing all residual data with full metadata
    event_id : str
        Event identifier
    event_data : dict
        Event metadata
    run_dir : str
        Directory to save the dashboard
    """
    if not residual_data:
        return None
    
    df = pd.DataFrame(residual_data)
    residuals = df['residual'].values
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top: Residual Map
    ax1 = fig.add_subplot(gs[0, :])
    pad = 0.05
    lat_min, lat_max = df['latitude'].min() - pad, df['latitude'].max() + pad
    lon_min, lon_max = df['longitude'].min() - pad, df['longitude'].max() + pad
    
    vmax = max(abs(df['residual'].min()), abs(df['residual'].max()))
    vmin = -vmax
    
    scatter1 = ax1.scatter(
        df['longitude'],
        df['latitude'],
        c=df['residual'],
        cmap='RdBu_r',
        s=200,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax
    )
    
    ax1.plot(
        event_data['longitude'],
        event_data['latitude'],
        'r*',
        markersize=25,
        markeredgecolor='black',
        label=f"Epicenter (M{event_data['magnitude']})"
    )
    
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    
    try:
        cx.add_basemap(
            ax1,
            crs='EPSG:4326',
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom='auto'
        )
    except:
        pass  # If basemap fails, continue without it
    
    plt.colorbar(scatter1, ax=ax1, label="Residual (log₁₀)", orientation='vertical')
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title("Residual Map")
    ax1.legend()
    
    # Bottom Left: Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    n_bins = min(20, max(5, int(np.sqrt(len(residuals)))))
    ax2.hist(residuals, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mean_res, std_res) * len(residuals) * (residuals.max() - residuals.min()) / n_bins
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal(μ={mean_res:.3f}, σ={std_res:.3f})')
    ax2.axvline(mean_res, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (log₁₀)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Histogram')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Bottom Middle: Observed vs Simulated Scatter
    ax3 = fig.add_subplot(gs[1, 1])
    df_valid = df[(df['obs_pga'] > 0) & (df['sim_pga'] > 0)]
    if len(df_valid) > 0:
        obs_log = np.log10(df_valid['obs_pga'])
        sim_log = np.log10(df_valid['sim_pga'])
        scatter3 = ax3.scatter(
            obs_log,
            sim_log,
            c=df_valid['residual'],
            cmap='RdBu_r',
            s=80,
            edgecolor='black',
            alpha=0.7
        )
        min_val = min(obs_log.min(), sim_log.min())
        max_val = max(obs_log.max(), sim_log.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')
        plt.colorbar(scatter3, ax=ax3, label="Residual")
        ax3.set_xlabel('Observed PGA (log₁₀(g))')
        ax3.set_ylabel('Simulated PGA (log₁₀(g))')
        ax3.set_title('Observed vs Simulated')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
    
    # Bottom Right: Residual vs Distance
    ax4 = fig.add_subplot(gs[1, 2])
    scatter4 = ax4.scatter(
        df['distance_km'],
        df['residual'],
        c=np.log10(df['obs_pga']),
        cmap='viridis',
        s=80,
        edgecolor='black',
        alpha=0.7
    )
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.axhline(mean_res, color='green', linestyle='--', linewidth=2)
    plt.colorbar(scatter4, ax=ax4, label="log₁₀(Obs PGA)")
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel('Residual (log₁₀)')
    ax4.set_title('Residual vs Distance')
    ax4.grid(alpha=0.3)
    
    # Bottom row: Box plot and Q-Q plot
    ax5 = fig.add_subplot(gs[2, 0])
    bp = ax5.boxplot(residuals, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax5.axhline(mean_res, color='green', linestyle='--', linewidth=2)
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Residual (log₁₀)')
    ax5.set_title('Box Plot')
    ax5.grid(axis='y', alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot')
    ax6.grid(alpha=0.3)
    
    # Summary statistics text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    stats_text = f"""
    Summary Statistics:
    
    Number of Stations: {len(residuals)}
    Mean Residual: {mean_res:.4f}
    Std. Deviation: {std_res:.4f}
    Min Residual: {residuals.min():.4f}
    Max Residual: {residuals.max():.4f}
    Median Residual: {np.median(residuals):.4f}
    
    Interpretation:
    • Positive residual: Sim > Obs (over-prediction)
    • Negative residual: Sim < Obs (under-prediction)
    • Mean near 0: Low bias
    • Low std dev: Low scatter
    """
    ax7.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"{event_data.get('title', 'Unknown Event')}\nResidual Analysis Dashboard", 
                 fontsize=14, y=0.98)
    
    output_path = os.path.join(run_dir, f"residual_dashboard_{event_id}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Residual summary dashboard saved to: {output_path}")
    
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

    # Create dictionaries with full station info for alignment
    obs_data_full = {}
    for d in pga_data:
        station_id = d['station'].split('.')[1] if '.' in d['station'] else d['station']
        obs_data_full[station_id] = d
    
    sim_data_full = {}
    for d in sim_data:
        station_id = d['station']
        sim_data_full[station_id] = d

    stations_to_compare = sorted(list(set(obs_data_full.keys()) & set(sim_data_full.keys())))

    residuals_list = []
    residuals_values = []

    # Calculate Residual: log10(Simulated) - log10(Observed)
    print("\n| Station | Observed PGA (g) | Simulated PGA (g) | Residual (log10)|")
    print("|:--- |:---:|:---:|:---:|")

    for station in stations_to_compare:
        obs_data = obs_data_full[station]
        sim_data_item = sim_data_full[station]
        
        obs_pga = obs_data['pga_g']
        sim_pga = sim_data_item['pga_g']

        # Calculate log residual only if both values are greater than zero
        if obs_pga > 0 and sim_pga > 0:
            residual = np.log10(sim_pga) - np.log10(obs_pga)
            residuals_values.append(residual)
            
            # Store full residual data for visualizations
            residuals_list.append({
                'station': station,
                'residual': residual,
                'obs_pga': obs_pga,
                'sim_pga': sim_pga,
                'latitude': obs_data.get('latitude', sim_data_item.get('latitude')),
                'longitude': obs_data.get('longitude', sim_data_item.get('longitude')),
                'distance_km': obs_data.get('distance_km', sim_data_item.get('distance_km', 0))
            })
            
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | {residual:.3f} |")
        else:
            print(f"| {station} | {obs_pga:.3e} | {sim_pga:.3e} | N/A (Zero Value) |")

    # 6. Final Summary
    if residuals_values:
        mean_residual = np.mean(residuals_values)
        std_residual = np.std(residuals_values)
        print(f"\n✅ Validation Summary:")
        print(f"   Mean Residual (Bias): {mean_residual:.3f} (Lower bias is better)")
        print(f"   Std. Dev. Residual: {std_residual:.3f} (Lower scatter is better)")
        
        # 7. Generate All Residual Visualizations
        print("\n--- Generating Residual Visualizations ---")
        
        # Residual Map
        if residuals_list:
            generate_residual_map(
                residuals_list,
                event_data,
                f"residual_map_{event_id}.png",
                run_dir
            )
            
            # Site-Specific Residual Chart
            generate_site_specific_residual_chart(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            # Observed vs Simulated Scatter Plot
            generate_observed_vs_simulated_scatter(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            # Residual Distribution Plots
            generate_residual_distribution_plots(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            # Residual vs Distance Plot
            generate_residual_vs_distance_plot(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            # Residual vs Observed PGA Plot
            generate_residual_vs_observed_pga_plot(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            # Multi-Panel Summary Dashboard
            generate_residual_summary_dashboard(
                residuals_list,
                event_id,
                event_data,
                run_dir
            )
            
            print(f"\n✅ All residual visualizations saved to: {run_dir}/")

    return True

