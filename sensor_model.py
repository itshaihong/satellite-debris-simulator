import numpy as np
from config import Config

class RadarSensor:
    def __init__(self, sigma_range, sigma_doppler, time_step):
        self.sigma_r = sigma_range
        self.sigma_v = sigma_doppler
        self.dt = time_step

        # Hardware limits
        self.fov_rad = np.radians(Config.FOV_DEGREES)
        self.max_range = Config.MAX_RANGE_KM * 1000.0 # Convert to meters
        
        # Environmental constants
        self.reference_range = 10000.0 # 10 km reference for 1/R^4 noise scaling
        self.earth_radius = 6371000.0  # ~6371 km in meters
        
        self.prev_eta = None

    def observe(self, r_obs, v_obs, r_deb, v_deb):
        # 1. Calculate Relative True State
        rel_pos = r_deb - r_obs
        rel_vel = v_deb - v_obs
        
        # True Range (Euclidean distance)
        true_range = np.linalg.norm(rel_pos)
        
        # True Doppler (Range-rate: dot product of rel_pos and rel_vel / range)
        true_doppler = np.dot(rel_pos, rel_vel) / true_range
        
        # 2. Inject Gaussian Noise (Sensor Reality)
        
        noisy_range = true_range + np.random.normal(0, self.sigma_r)
        noisy_doppler = true_doppler + np.random.normal(0, self.sigma_v)
        
        # 3. Calculate Eta and Eta-dot
        current_eta = noisy_range * noisy_doppler
        
        if self.prev_eta is None:
            eta_dot = 0.0 # Cannot calculate derivative on first step
        else:
            # Finite difference for eta derivative
            eta_dot = (current_eta - self.prev_eta) / self.dt
            
        self.prev_eta = current_eta
        
        return {
            "true_range": true_range,
            "true_doppler": true_doppler,
            "noisy_range": noisy_range,
            "noisy_doppler": noisy_doppler,
            "eta_dot": eta_dot,
            "true_state_deb": np.concatenate([r_deb, v_deb]) # Clean Labels
        }
    
    def observe_realistic(self, r_obs, v_obs, r_deb, v_deb):
        # 1. Calculate Relative Vectors
        rel_pos = r_deb - r_obs
        rel_vel = v_deb - v_obs
        true_range = np.linalg.norm(rel_pos)
        
        if true_range == 0:
            return None # Prevent division by zero
            
        # --- CONSTRAINT 1: MAXIMUM RANGE CUT-OFF ---
        if true_range > self.max_range:
            self.prev_eta = None
            return None
            
        # --- CONSTRAINT 2: EARTH OCCLUSION (Line of Sight) ---
        # Calculate the closest point to Earth's center along the radar beam vector
        d_mag_sq = np.dot(rel_pos, rel_pos)
        
        # 't' represents the parameterized position along the line segment (0 = observer, 1 = debris)
        t_closest = -np.dot(r_obs, rel_pos) / d_mag_sq
        
        # If t is between 0 and 1, the closest point lies strictly between the two objects
        if 0 < t_closest < 1:
            closest_point = r_obs + t_closest * rel_pos
            closest_dist = np.linalg.norm(closest_point)
            
            # If that closest point is inside the Earth, the Earth is blocking the view
            if closest_dist < self.earth_radius:
                self.prev_eta = None
                return None
                

        # --- SIGNAL PROCESSING ---
        true_doppler = np.dot(rel_pos, rel_vel) / true_range
        
        # Dynamic noise based on Radar Equation (1/R^4 power drop translates to R^2 noise scaling)
        scale_factor = (true_range / self.reference_range)**2 
        dynamic_sigma_r = self.sigma_r * scale_factor
        dynamic_sigma_v = self.sigma_v * scale_factor

        # Inject Gaussian Noise
        noisy_range = true_range + np.random.normal(0, dynamic_sigma_r)
        noisy_doppler = true_doppler + np.random.normal(0, dynamic_sigma_v)
        
        # Calculate ML Sub-State feature (Eta and Eta-dot)
        current_eta = noisy_range * noisy_doppler
        
        if self.prev_eta is None:
            eta_dot = 0.0 
        else:
            eta_dot = (current_eta - self.prev_eta) / self.dt
            
        self.prev_eta = current_eta
        
        return {
            "true_range": true_range,
            "true_doppler": true_doppler,
            "noisy_range": noisy_range,
            "noisy_doppler": noisy_doppler,
            "eta_dot": eta_dot,
            "true_state_deb": np.concatenate([r_deb, v_deb])
        }