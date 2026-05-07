import pandas as pd
import numpy as np
import orekit
from scipy.optimize import least_squares
from config import Config

from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
import os

# Initialize Orekit (must be done in every script that uses it)
vm = orekit.initVM()
if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()
setup_orekit_curdir("orekit-data.zip")

from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngleType
from org.orekit.utils import PVCoordinates, Constants, IERSConventions
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import GeodeticPoint
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import TopocentricFrame


# --- CONFIG FOR BENCHMARK ---
EARTH_MU = Constants.WGS84_EARTH_MU
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

def run_simple_op(state_vector, dt, start_epoch, inertial_frame):
    """
    Simplified Propagator using Orekit.
    ONLY uses J2 (2x0) gravity. No drag, no sun/moon.
    Dynamically uses the start_epoch and inertial_frame passed from the dataset.
    """
    pos = Vector3D(float(state_vector[0]), float(state_vector[1]), float(state_vector[2]))
    vel = Vector3D(float(state_vector[3]), float(state_vector[4]), float(state_vector[5]))
    pv = PVCoordinates(pos, vel)
    
    # Create the orbit using the dynamic epoch and frame
    initial_orbit = CartesianOrbit(pv, inertial_frame, start_epoch, EARTH_MU)
    
    # Setup robust Numerical Propagator
    min_step = 0.1 
    max_step = 300.0
    pos_tolerance = 0.1 # meters
    
    tolerances = NumericalPropagator.tolerances(pos_tolerance, initial_orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(min_step, max_step, 
                                            orekit.JArray_double.cast_(tolerances[0]), 
                                            orekit.JArray_double.cast_(tolerances[1]))
    
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(initial_orbit))
    
    # Add ONLY J2 Force Model
    gravity_provider = GravityFieldFactory.getNormalizedProvider(2, 0)
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))
    
    # Propagate to target measurement time
    target_date = start_epoch.shiftedBy(float(dt))
    final_state = propagator.propagate(target_date).getPVCoordinates()
    
    return np.array([
        final_state.getPosition().getX(), final_state.getPosition().getY(), final_state.getPosition().getZ(),
        final_state.getVelocity().getX(), final_state.getVelocity().getY(), final_state.getVelocity().getZ()
    ])

def compute_residuals(state_guess, times, observed_ranges, station_frame, start_epoch, inertial_frame):
    """
    Cost Function: Difference between guessed orbit ranges and noisy ground data.
    Automatically accounts for Earth's rotation using TopocentricFrame.
    """
    residuals = []
    for i, t in enumerate(times):
        # 1. Propagate the guessed state to the measurement time
        pred_state = run_simple_op(state_guess, t, start_epoch, inertial_frame)
        pred_pos = Vector3D(float(pred_state[0]), float(pred_state[1]), float(pred_state[2]))
        
        # 2. Calculate the exact target date for this specific measurement
        target_date = start_epoch.shiftedBy(float(t))
        
        # 3. Get exact Range from the Rotating Earth Station
        # This replaces the static .distance() calculation and fixes the rotation drift!
        computed_range = station_frame.getRange(pred_pos, inertial_frame, target_date)
        
        # 4. Calculate Residual (Error)
        residuals.append(observed_ranges[i] - computed_range)
        
    return np.array(residuals)

def main():
    df = pd.read_csv(Config.OUTPUT_FILE_GROUND)
    episodes = df['episode_id'].unique()
    if not len(episodes):
        print("No data found in CSV.")
        return
        
    ep = episodes[2] 
    episode_data = df[df['episode_id'] == ep].reset_index(drop=True)

    # --- TRACKING ARC SPLITTER ---
    # Look at the time steps to see if there is a massive gap (e.g., > 60 seconds)
    times_all = episode_data['time_elapsed_s'].values
    time_diffs = np.diff(times_all)
    
    # Find indices where the gap between measurements is huge (indicating the debris went below horizon)
    gap_indices = np.where(time_diffs > 60.0)[0]
    
    if len(gap_indices) > 0:
        first_pass_end = gap_indices[0]
        print(f"\n[!] Multiple passes detected. Splitting arc at t={times_all[first_pass_end]}s.")
        # Slice the dataframe to ONLY include the first continuous pass
        episode_data = episode_data.iloc[:first_pass_end + 1].copy()
    else:
        print("\nSingle continuous pass detected.")
    
    print(f"--- Starting Classical OD Benchmark (Episode {ep}) ---")
    
    # --- Build station directly from Dataset Metadata ---
    lat_rad = np.radians(episode_data['station_lat_deg'].iloc[0])
    lon_rad = np.radians(episode_data['station_lon_deg'].iloc[0])
    alt_m = episode_data['station_alt_m'].iloc[0]
    
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, earth_frame)
    geo_point = GeodeticPoint(float(lat_rad), float(lon_rad), float(alt_m))
    station_frame = TopocentricFrame(earth, geo_point, "Dataset_Station")
    
    # Define start epoch and inertial frame (MUST match truth_model)
    utc = TimeScalesFactory.getUTC()
    collision_epoch = AbsoluteDate(2026, 5, 6, 12, 10, 0.0, utc)
    start_epoch = collision_epoch.shiftedBy(-5400.0)
    inertial_frame = FramesFactory.getEME2000()
    
    print(f"Loaded Station from CSV: Lat {np.degrees(lat_rad):.2f}°, Lon {np.degrees(lon_rad):.2f}°")
    
    # Extract Measurements
    times_all = episode_data['time_elapsed_s'].values
    ranges_all = episode_data['noisy_ground_range_m'].values
    
    # --- SPEED OPTIMIZATION 1: DOWNSAMPLE DATA ---
    # Take every 5th measurement (e.g., 25s intervals instead of 5s)
    # This cuts the propagation workload by 80%!
    times = times_all[::5]
    ranges = ranges_all[::5]
    print(f"Processing tracking arc of {len(times)} measurements over {times[-1] - times[0]} seconds.")
    
    # Setup Initial Guess
    true_initial = episode_data.iloc[0][['true_deb_x', 'true_deb_y', 'true_deb_z', 'true_deb_vx', 'true_deb_vy', 'true_deb_vz']].values
    x0 = true_initial.copy()
    x0[:3] += 5000.0  
    
    print("Running Non-Linear Least Squares Optimizer (J2-Only Model)...")
    
    res = least_squares(
        compute_residuals, 
        x0=x0, 
        args=(times, ranges, station_frame, start_epoch, inertial_frame),
        method='trf',      
        ftol=1e-2 ,         # Stop when cost function changes by less than 1% (faster) # --- SPEED OPTIMIZATION 2: LOOSER TOLERANCES ---
        # xtol=1e-2,         # Stop when state variables stop moving much
        # diff_step=1e-3,    # Use larger finite difference nudges to prevent getting stuck
        max_nfev=30
    )
    
    print(f"\nOptimization Complete (Success: {res.success})")
    od_rmse = np.sqrt(np.mean((res.x[:3] - true_initial[:3])**2))
    print(f"Initial Catalog Position Error : ~5000.00 meters")
    print(f"Post-OD Position RMSE          : {od_rmse:.2f} meters")

    # =====================================================================
    # --- ORBIT PREDICTION (OP) PHASE ---
    # =====================================================================
    print("\n--- Starting Classical Orbit Prediction (OP) Benchmark ---")
    
    try:
        # 1. Load the Space Dataset to find the exact True State at the collision
        df_space = pd.read_csv(Config.OUTPUT_FILE)
        ep_space_data = df_space[df_space['episode_id'] == ep]
        
        if not ep_space_data.empty:
            # 2. Find the exact moment of closest approach (Target Time ~5400s)
            target_row = ep_space_data.iloc[(ep_space_data['time_elapsed_s'] - 5400.0).abs().argsort()[:1]]
            
            true_target_time = target_row['time_elapsed_s'].values[0]
            true_future_pos = target_row[['true_deb_x', 'true_deb_y', 'true_deb_z']].values[0]
            
            print(f"Targeting Encounter Time: {true_target_time:.1f} seconds...")
            
            # 3. Propagate the optimized OD state forward using our simplified J2 engine
            predicted_state = run_simple_op(res.x, true_target_time, start_epoch, inertial_frame)
            pred_future_pos = predicted_state[:3]
            
            # 4. Calculate Final Prediction Error
            op_rmse = np.sqrt(np.mean((pred_future_pos - true_future_pos)**2))
            
            print(f"Post-OP Prediction RMSE : {op_rmse:.2f} meters")
            
            print("\n================ BENCHMARK RESULT ================")
            print(f"Over a {(true_target_time)/60:.1f}-minute prediction window, the classical")
            print(f"J2-only physics engine drifted by {op_rmse/1000:.2f} kilometers.")
            print("This is the baseline error your ML model needs to beat!")
            print("==================================================")
            
        else:
            print(f"No space tracking data found for Episode {ep}. Cannot calculate OP RMSE.")
            
    except FileNotFoundError:
        print("Error: ml_space_dataset.csv not found. Cannot compare OP to ground truth.")

if __name__ == "__main__":
    main()