import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from config import Config

def plot_tracking_windows(space_file="ml_space_dataset.csv", ground_file="traditional_ground_dataset.csv"):
    print("Loading Space and Ground tracking datasets...")
    try:
        df_space = pd.read_csv(space_file)
        df_ground = pd.read_csv(ground_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure you have run main.py first.")
        return

    # Check for the multi-episode structure
    if 'episode_id' not in df_space.columns or 'episode_id' not in df_ground.columns:
        print("Error: Data is missing 'episode_id'. Please run the updated main.py.")
        return
        
    # Find episodes that exist in both datasets to ensure a good visual comparison
    episodes_space = set(df_space['episode_id'].unique())
    episodes_ground = set(df_ground['episode_id'].unique())
    shared_episodes = list(episodes_space.intersection(episodes_ground))
    
    if not shared_episodes:
        print("No episodes found that have both Space and Ground tracking data.")
        # Fall back to picking from whichever dataset has data
        selected_ep = random.choice(list(episodes_ground) if episodes_ground else list(episodes_space))
    else:
        selected_ep = random.choice(shared_episodes)
        
    print(f"Randomly selected Episode {selected_ep} for visualization.")
    
    # Filter to the selected episode
    plot_space = df_space[df_space['episode_id'] == selected_ep].copy()
    plot_ground = df_ground[df_ground['episode_id'] == selected_ep].copy()

    # Convert coordinates to km
    if not plot_space.empty:
        plot_space['obs_x_km'] = plot_space['obs_x'] / 1000
        plot_space['obs_y_km'] = plot_space['obs_y'] / 1000
        plot_space['obs_z_km'] = plot_space['obs_z'] / 1000
        plot_space['true_deb_x_km'] = plot_space['true_deb_x'] / 1000
        plot_space['true_deb_y_km'] = plot_space['true_deb_y'] / 1000
        plot_space['true_deb_z_km'] = plot_space['true_deb_z'] / 1000

    if not plot_ground.empty:
        plot_ground['true_deb_x_km'] = plot_ground['true_deb_x'] / 1000
        plot_ground['true_deb_y_km'] = plot_ground['true_deb_y'] / 1000
        plot_ground['true_deb_z_km'] = plot_ground['true_deb_z'] / 1000

    # Set up the figure dashboard
    fig = plt.figure(figsize=(16, 8))
    
    # --- Plot 1: 3D Orbital Trajectories ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_radius = 6371 # km
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax1.plot_wireframe(x, y, z, color="lightblue", alpha=0.3)

    if not plot_space.empty:
        ax1.plot(plot_space['obs_x_km'], plot_space['obs_y_km'], plot_space['obs_z_km'], 
                 label='Observer Satellite (Space Radar)', color='blue', linewidth=2)
        ax1.plot(plot_space['true_deb_x_km'], plot_space['true_deb_y_km'], plot_space['true_deb_z_km'], 
                 label='Debris Orbit (Space Window)', color='red', linestyle='dashed', alpha=0.7)
                 
    if not plot_ground.empty:
        ax1.plot(plot_ground['true_deb_x_km'], plot_ground['true_deb_y_km'], plot_ground['true_deb_z_km'], 
                 label='Debris Orbit (Ground Window)', color='green', linewidth=2)
        station_lat = np.radians(plot_ground['station_lat_deg'].iloc[0])
        station_lon = np.radians(plot_ground['station_lon_deg'].iloc[0])
        
        # Convert Geodetic (Lat/Lon) to 3D Cartesian (km)
        earth_radius_km = 6371.0
        stn_x = earth_radius_km * np.cos(station_lat) * np.cos(station_lon)
        stn_y = earth_radius_km * np.cos(station_lat) * np.sin(station_lon)
        stn_z = earth_radius_km * np.sin(station_lat)
        
        # Plot the station as a bright green triangle on the surface
        ax1.scatter(stn_x, stn_y, stn_z, color='lime', marker='^', s=150, 
                    edgecolor='black', label='Ground Station', zorder=5)

    ax1.set_title(f'Absolute Orbital Trajectories (Episode {selected_ep})')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.legend()

    # --- Plot 2: Space Radar vs Ground Radar Tracking Windows ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    if not plot_space.empty:
        ax2.plot(plot_space['time_elapsed_s'] / 60, plot_space['noisy_range_m'] / 1000, 
                 color='purple', marker='o', markersize=4, label='Space Radar Range (Max 35km)')
    
    if not plot_ground.empty:
        ax2.plot(plot_ground['time_elapsed_s'] / 60, plot_ground['noisy_ground_range_m'] / 1000, 
                 color='orange', marker='x', markersize=4, label='Ground Radar Range (Max 3000km)')

    ax2.set_title(f'Tracking Windows: Space vs Ground (Episode {selected_ep})')
    ax2.set_xlabel('Time Since Episode Start (Minutes)')
    ax2.set_ylabel('Observed Range (km)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_tracking_windows(Config.OUTPUT_FILE, Config.OUTPUT_FILE_GROUND)