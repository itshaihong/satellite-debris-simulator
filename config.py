import numpy as np
from astropy import units as u

class Config:
    # Dataset Scaling
    NUM_EPISODES = 10               # Generate 100 unique encounters for training
    
    # Time settings for each encounter
    SIMULATION_DURATION = 120 * u.min # 120 minutes to ensure they pass each other
    TIME_STEP = 5 * u.s              # Radar ping frequency
    
    # Sensor Noise and Hardware Limits
    SIGMA_RANGE = 10.0               # High-fidelity radar (10 meters)
    SIGMA_DOPPLER = 0.1              # High-fidelity radar (0.1 m/s)
    FOV_DEGREES = 120.0
    MAX_RANGE_KM = 35.0
    
    # Output
    OUTPUT_FILE = "ml_training_dataset.csv"
    OUTPUT_FILE_GROUND = "ground_radar_dataset.csv"