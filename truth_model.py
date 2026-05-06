import os
import random
import numpy as np
import orekit
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir

# Initialize Orekit
vm = orekit.initVM()
if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()
setup_orekit_curdir("orekit-data.zip")

from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, PositionAngleType, OrbitType
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import GeodeticPoint
from org.orekit.frames import TopocentricFrame

class TruthGenerator:
    def __init__(self, time_step):
        self.time_step = float(time_step) if not hasattr(time_step, 'value') else time_step.value
        self.utc = TimeScalesFactory.getUTC()
        
        # 1. Define the Encounter Epoch (The moment they cross paths)
        self.collision_epoch = AbsoluteDate(2026, 5, 6, 12, 10, 0.0, self.utc)
        
        # 2. Define the Start Epoch (10 minutes before encounter) # MODIFIABLE
        self.start_epoch = self.collision_epoch.shiftedBy(-5400.0)
        
        self.inertial_frame = FramesFactory.getEME2000()
        self.earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                                      Constants.WGS84_EARTH_FLATTENING, self.earth_frame)
        
        # # Define a ground radar station (Latitude, Longitude, Altitude) [Abandoned]
        # station_lat = float(np.radians(40.0))
        # station_lon = float(np.radians(-100.0))
        # station_alt = 0.0
        
        # station_point = GeodeticPoint(station_lat, station_lon, station_alt)
        # self.ground_station = TopocentricFrame(self.earth, station_point, "SSN_Ground_Radar")
        
        # Ground radar elevation mask (cannot track below 10 degrees due to horizon/clutter)
        self.elevation_mask = float(np.radians(5.0))
        # Ground radar max slant range (e.g., 3000 km)
        self.ground_max_range = 3000000.0

        # 3. Randomize Observer Satellite
        self.obs_alt = random.uniform(500000.0, 600000.0)
        self.obs_inc = random.uniform(50.0, 60.0)
        
        # 4. Generate Orbits using Velocity Sampling
        obs_orbit_at_collision, deb_orbit_at_collision = self._generate_encounter_orbits()
        
        # 5. "Rewind" the orbits analytically by 10 minutes to set the start state
        obs_start_orbit = obs_orbit_at_collision.shiftedBy(-5400.0) # MODIFIABLE
        deb_start_orbit = deb_orbit_at_collision.shiftedBy(-5400.0) # MODIFIABLE
        
        self.obs_propagator = self._build_propagator(obs_start_orbit)
        self.deb_propagator = self._build_propagator(deb_start_orbit)

        # 1. Get the debris position at collision in inertial frame
        mid_date = self.start_epoch.shiftedBy(2700.0)
        temp_state = self.deb_propagator.propagate(mid_date)
        true_mid_pos = temp_state.getPVCoordinates().getPosition()
        geo_point = self.earth.transform(true_mid_pos, self.inertial_frame, mid_date)
        
        # 3. Place the Ground Station exactly under the collision point
        self.ground_station = TopocentricFrame(self.earth, geo_point, "Synchronized_Radar")

    def _generate_encounter_orbits(self):
        """Generates Observer and Debris orbits that perfectly intersect at collision_epoch"""
        
        # A. Create Observer Orbit at collision epoch
        a_obs = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + self.obs_alt
        obs_orbit = KeplerianOrbit(a_obs, 0.0001, float(np.radians(self.obs_inc)), 
                                   0.0, 0.0, 0.0, PositionAngleType.MEAN, 
                                   self.inertial_frame, self.collision_epoch, Constants.WGS84_EARTH_MU)
        
        # B. Get Observer's Cartesian Position and Velocity at collision
        obs_pv = obs_orbit.getPVCoordinates()
        pos = obs_pv.getPosition()
        vel = obs_pv.getVelocity()
        
        # C. Construct RSW Local Frame Unit Vectors
        u_r = pos.scalarMultiply(1.0 / pos.getNorm())                          # Radial
        
        w_vec = pos.crossProduct(vel)
        u_w = w_vec.scalarMultiply(1.0 / w_vec.getNorm())                      # Cross-track (Normal)
        
        s_vec = u_w.crossProduct(u_r)
        u_s = s_vec.scalarMultiply(1.0 / s_vec.getNorm())                      # Along-track          # Along-track
        
        # D. Velocity Sampling: Generate a random relative encounter speed (e.g., 0.5 to 1.5 km/s)
        # Keep sampling velocities until we find an orbit that stays in space!
        valid_orbit = False
        while not valid_orbit:
            # D. Velocity Sampling
            dv_r = random.uniform(-0.5, 0.5) * 1000.0  
            dv_s = random.uniform(-1.0, 1.0) * 1000.0  
            dv_w = random.uniform(-0.5, 0.5) * 1000.0  
            
            dv_inertial = Vector3D(float(dv_r), u_r).add(Vector3D(float(dv_s), u_s)).add(Vector3D(float(dv_w), u_w))
            deb_vel = vel.add(dv_inertial)
            
            # E. Create Potential Debris Orbit
            deb_pv = PVCoordinates(pos, deb_vel)
            deb_orbit = CartesianOrbit(deb_pv, self.inertial_frame, self.collision_epoch, Constants.WGS84_EARTH_MU)
            
            # F. Physics Check: Must be elliptical (e < 1) and stay above the atmosphere
            eccentricity = deb_orbit.getE()
            if eccentricity < 1.0:
                semi_major_axis = deb_orbit.getA()
                perigee_altitude = semi_major_axis * (1.0 - eccentricity) - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
                
                # Minimum altitude boundary for Harris-Priester is 100km, we use 150km for safety
                if perigee_altitude > 150000.0: 
                    valid_orbit = True
        
        return obs_orbit, deb_orbit

    def _build_propagator(self, initial_orbit):
        min_step, max_step = 0.001, 300.0
        position_tolerance = 10.0
        tolerances = NumericalPropagator.tolerances(position_tolerance, initial_orbit, initial_orbit.getType())
        integrator = DormandPrince853Integrator(min_step, max_step, 
                                                orekit.JArray_double.cast_(tolerances[0]), 
                                                orekit.JArray_double.cast_(tolerances[1]))

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(OrbitType.EQUINOCTIAL) 
        propagator.setInitialState(SpacecraftState(initial_orbit))

        # High-Fidelity Physics
        gravity_provider = GravityFieldFactory.getNormalizedProvider(40, 40)
        propagator.addForceModel(HolmesFeatherstoneAttractionModel(self.earth_frame, gravity_provider))
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
        atmosphere = HarrisPriester(CelestialBodyFactory.getSun(), self.earth)
        propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(1.0, 2.2)))

        return propagator

    def propagate_step(self, time_delta):
        """Propagates both orbits to the target epoch."""
        if hasattr(time_delta, 'value'):
            delta_seconds = float(time_delta.value)
        else:
            delta_seconds = float(time_delta)
            
        target_date = self.start_epoch.shiftedBy(delta_seconds)
        
        obs_state = self.obs_propagator.propagate(target_date)
        deb_state = self.deb_propagator.propagate(target_date)
        
        # 1. Standard Cartesian Vectors (for your ML model)
        obs_pv = obs_state.getPVCoordinates()
        deb_pv = deb_state.getPVCoordinates()
        
        r_obs = np.array([obs_pv.getPosition().getX(), obs_pv.getPosition().getY(), obs_pv.getPosition().getZ()])
        v_obs = np.array([obs_pv.getVelocity().getX(), obs_pv.getVelocity().getY(), obs_pv.getVelocity().getZ()])
        
        r_deb = np.array([deb_pv.getPosition().getX(), deb_pv.getPosition().getY(), deb_pv.getPosition().getZ()])
        v_deb = np.array([deb_pv.getVelocity().getX(), deb_pv.getVelocity().getY(), deb_pv.getVelocity().getZ()])

        # 2. TRADITIONAL GROUND STATION MATH ---
        # Get the debris position in the inertial frame
        deb_pos_inertial = deb_pv.getPosition()
        
        # Calculate true elevation and range from the specific ground station on Earth
        elevation = self.ground_station.getElevation(deb_pos_inertial, self.inertial_frame, target_date)
        slant_range = self.ground_station.getRange(deb_pos_inertial, self.inertial_frame, target_date)
        
        ground_visible = False
        noisy_ground_range = None
        
        # Check against hardware and geometric constraints
        if elevation > self.elevation_mask and slant_range < self.ground_max_range:
            ground_visible = True
            # Traditional Ground Radar Noise (e.g., 15m standard deviation)
            noisy_ground_range = slant_range + np.random.normal(0, 15.0)
        
        return r_obs, v_obs, r_deb, v_deb, ground_visible, noisy_ground_range