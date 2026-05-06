import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from config import Config

def plot_random_episode(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run main.py first.")
        return

    # Check for the multi-episode structure
    if 'episode_id' in df.columns:
        episodes = df['episode_id'].unique()
        selected_ep = random.choice(episodes)
        print(f"Dataset contains {len(episodes)} episodes.")
        print(f"Randomly selected Episode {selected_ep} for visualization.")
        
        # Filter the dataframe to only include the randomly selected episode
        plot_df = df[df['episode_id'] == selected_ep].copy()
        title_suffix = f" (Episode {selected_ep})"
    else:
        print("No 'episode_id' column found. Visualizing the entire dataset.")
        plot_df = df.copy()
        title_suffix = ""

    if plot_df.empty:
        print("Error: The selected episode contains no data.")
        return

    # Convert coordinates from meters to kilometers for easier viewing
    plot_df['obs_x_km'] = plot_df['obs_x'] / 1000
    plot_df['obs_y_km'] = plot_df['obs_y'] / 1000
    plot_df['obs_z_km'] = plot_df['obs_z'] / 1000
    
    plot_df['true_deb_x_km'] = plot_df['true_deb_x'] / 1000
    plot_df['true_deb_y_km'] = plot_df['true_deb_y'] / 1000
    plot_df['true_deb_z_km'] = plot_df['true_deb_z'] / 1000

    # Set up the figure dashboard
    fig = plt.figure(figsize=(16, 8))
    
    # --- Plot 1: 3D Orbital Trajectories ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot Earth (approximate sphere)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_radius = 6371 # km
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax1.plot_wireframe(x, y, z, color="lightblue", alpha=0.3)

    # Plot Observer and Debris (using plot_df instead of df)
    ax1.plot(plot_df['obs_x_km'], plot_df['obs_y_km'], plot_df['obs_z_km'], 
             label='Observer Satellite (Radar)', color='blue', linewidth=2)
    ax1.plot(plot_df['true_deb_x_km'], plot_df['true_deb_y_km'], plot_df['true_deb_z_km'], 
             label='True Debris Orbit', color='red', linestyle='dashed')
    
    ax1.set_title(f'Absolute Orbital Trajectories{title_suffix}')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.legend()

    # --- Plot 2: Noisy Radar Measurements (ML Inputs) ---
    ax2 = fig.add_subplot(2, 2, 2)
    # Using a scatter/line combo to clearly see the sparse data points
    ax2.plot(plot_df['time_elapsed_s'] / 60, plot_df['noisy_range_m'] / 1000, 
             color='purple', alpha=0.7, marker='o', markersize=4)
    ax2.set_title(f'Sensor Reality: Noisy Range{title_suffix}')
    ax2.set_ylabel('Range (km)')
    ax2.grid(True)

    # --- Plot 3: Sub-State Vector Target (Eta-Dot) ---
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(plot_df['time_elapsed_s'] / 60, plot_df['eta_dot'], 
             color='orange', alpha=0.7, marker='x', markersize=4)
    ax3.set_title(f'Feature Engineering: $\dot{{\eta}}${title_suffix}')
    ax3.set_xlabel('Time Since Episode Start (Minutes)')
    ax3.set_ylabel('$\dot{\eta}$')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_random_episode(Config.OUTPUT_FILE)