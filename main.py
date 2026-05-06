import csv
import numpy as np
from config import Config
from truth_model import TruthGenerator
from sensor_model import RadarSensor 

def main():
    print(f"Starting Multi-Episode Data Generation: {Config.NUM_EPISODES} encounters...")
    
    # Safely handle Astropy units from config
    sim_dur_s = float(Config.SIMULATION_DURATION.to('s').value) if hasattr(Config.SIMULATION_DURATION, 'value') else float(Config.SIMULATION_DURATION * 60)
    time_step_s = float(Config.TIME_STEP.value) if hasattr(Config.TIME_STEP, 'value') else float(Config.TIME_STEP)
    total_steps = int(sim_dur_s / time_step_s)

    dataset = []
    dataset_ground = []
    window_durations = [] # Track the visibility duration for each episode
    window_durations_ground = [] # Track visibility for ground radar as well

    for episode in range(Config.NUM_EPISODES):
        print(f"Simulating Encounter {episode + 1}/{Config.NUM_EPISODES}...")
        
        truth_gen = TruthGenerator(time_step_s)
        # Note: FOV was removed in the previous step, so only passing MAX_RANGE_KM
        radar = RadarSensor(Config.SIGMA_RANGE, Config.SIGMA_DOPPLER, Config.MAX_RANGE_KM)
        
        visible_steps = 0
        visible_steps_ground = 0
        
        for step in range(total_steps):
            time_delta = step * time_step_s
            
            # Propagate and Observe
            r_obs, v_obs, r_deb, v_deb, ground_visible, noisy_ground_range = truth_gen.propagate_step(time_delta)
            obs_data = radar.observe_realistic(r_obs, v_obs, r_deb, v_deb)
            
            # Only save data if debris is actually caught in the radar window
            if obs_data is not None and obs_data["eta_dot"] != 0.0:
                visible_steps += 1
                row = [
                    episode,
                    time_delta,
                    *r_obs, *v_obs, 
                    obs_data["noisy_range"],
                    obs_data["noisy_doppler"],
                    obs_data["eta_dot"],
                    *obs_data["true_state_deb"] 
                ]
                dataset.append(row)
            
            if ground_visible:
                visible_steps_ground += 1
                row_ground = [
                    episode, time_delta,
                    noisy_ground_range,
                    *r_deb, *v_deb # True states included to calculate RMSE later
                ]
                dataset_ground.append(row_ground)
                
        # If the radar saw the debris during this episode, record how long it was visible
        if visible_steps > 0:
            window_durations.append(visible_steps * time_step_s)
        if visible_steps_ground > 0:
            window_durations_ground.append(visible_steps_ground * time_step_s)

    # Export to CSV
    headers = [
        "episode_id", "time_elapsed_s", 
        "obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz",
        "noisy_range_m", "noisy_doppler_ms", "eta_dot", 
        "true_deb_x", "true_deb_y", "true_deb_z", "true_deb_vx", "true_deb_vy", "true_deb_vz"
    ]
               
    with open(Config.OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(dataset)

    # onboard radar visibility statistics

    print(f"\nComplete! Generated {len(dataset)} valid tracking data points across {Config.NUM_EPISODES} encounters.")
    print(f"Data saved to {Config.OUTPUT_FILE}")

    if window_durations:
        avg_window = np.mean(window_durations)
        min_window = np.min(window_durations)
        max_window = np.max(window_durations)
        
        print(f"\n--- Visibility Window Statistics ---")
        print(f"Average Tracking Window : {avg_window:.2f} seconds ({avg_window/60:.2f} minutes)")
        print(f"Minimum Tracking Window : {min_window:.2f} seconds ({min_window/60:.2f} minutes)")
        print(f"Maximum Tracking Window : {max_window:.2f} seconds ({max_window/60:.2f} minutes)")
        print(f"Total Successful Tracks : {len(window_durations)} / {Config.NUM_EPISODES}")
    else:
        print("\nNo onboard radar visibility windows found in any episode. Adjust your encounter parameters.")

    ground_headers = ["episode_id", "time_elapsed_s", "noisy_ground_range_m", 
                      "true_deb_x", "true_deb_y", "true_deb_z", "true_deb_vx", "true_deb_vy", "true_deb_vz"]
    with open(Config.OUTPUT_FILE_GROUND, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(ground_headers)
        writer.writerows(dataset_ground)

    # Ground radar visibility statistics
    print(f"\nComplete! Generated {len(dataset_ground)} valid tracking data points across {Config.NUM_EPISODES} encounters.")
    print(f"Data saved to {Config.OUTPUT_FILE_GROUND}")
    if window_durations_ground:
        avg_window_ground = np.mean(window_durations_ground)
        min_window_ground = np.min(window_durations_ground)
        max_window_ground = np.max(window_durations_ground)
        
        print(f"\n--- Ground Radar Visibility Window Statistics ---")
        print(f"Average Ground Tracking Window : {avg_window_ground:.2f} seconds ({avg_window_ground/60:.2f} minutes)")
        print(f"Minimum Ground Tracking Window : {min_window_ground:.2f} seconds ({min_window_ground/60:.2f} minutes)")
        print(f"Maximum Ground Tracking Window : {max_window_ground:.2f} seconds ({max_window_ground/60:.2f} minutes)")
        print(f"Total Successful Ground Tracks : {len(window_durations_ground)} / {Config.NUM_EPISODES}")
    else:
        print("\nNo ground radar visibility windows found in any episode. Adjust your encounter parameters.")
    

if __name__ == "__main__":
    main()