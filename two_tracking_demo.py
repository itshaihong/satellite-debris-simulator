import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# --- CONFIGURATION ---
duration_frames = 200
earth_radius = 6371.0

# Orbital parameters (Simplified 2D for visual clarity)
obs_alt = earth_radius + 550.0
deb_alt = earth_radius + 550.0

# --- SETUP FIGURE ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-8000, 8000)
ax.set_ylim(-1000, 8000)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')

# 1. Draw Earth
earth = plt.Circle((0, 0), earth_radius, color='dodgerblue', alpha=0.6)
ax.add_patch(earth)

# 2. Draw Ground Station and its massive Elevation Mask (Tracking Window)
gs_x, gs_y = 0, earth_radius
ax.plot(gs_x, gs_y, marker='^', color='lime', markersize=10, label='Ground Station')
gs_window = patches.Wedge((gs_x, gs_y), 3000, 0, 180, color='lime', alpha=0.15, label='Ground Radar Window')
ax.add_patch(gs_window)

# 3. Setup Observer Satellite, Debris, and Space Radar Window
obs_dot, = ax.plot([], [], 'bo', markersize=8, label='Observer Satellite')
deb_dot, = ax.plot([], [], 'ro', markersize=6, label='Space Debris')
space_window = patches.Wedge((0,0), 300, 0, 45, color='cyan', alpha=0.4, label='Space Radar Window')
ax.add_patch(space_window)

# Dynamic UI Text
time_text = ax.text(-7500, 7500, '', color='white', fontsize=12)
gs_text = ax.text(-7500, 7000, '', color='lime', fontsize=12, fontweight='bold')
space_text = ax.text(-7500, 6500, '', color='cyan', fontsize=12, fontweight='bold')

ax.legend(loc='upper right', facecolor='white', framealpha=0.8)
plt.title("Tracking Window Comparison", color='white', fontsize=16, pad=20)

# --- ANIMATION LOGIC ---
def update(frame):
    # Time progression
    t = frame / duration_frames
    
    # Observer orbits left to right over the pole
    obs_angle = np.pi * 0.7 - (t * np.pi * 0.4)
    ox = obs_alt * np.cos(obs_angle)
    oy = obs_alt * np.sin(obs_angle)
    obs_dot.set_data([ox], [oy])
    
    # Observer's tiny forward-facing radar cone
    cone_angle_deg = np.degrees(obs_angle) - 90
    space_window.set_center((ox, oy))
    space_window.set_theta1(cone_angle_deg - 20)
    space_window.set_theta2(cone_angle_deg + 20)
    
    # Debris approaches from the right (counter-orbit or highly elliptical)
    deb_angle = np.pi * 0.3 + (t * np.pi * 0.5)
    dx = deb_alt * np.cos(deb_angle)
    dy = deb_alt * np.sin(deb_angle)
    deb_dot.set_data([dx], [dy])
    
    # Check Intersection (Ground)
    dist_to_gs = np.sqrt((dx - gs_x)**2 + (dy - gs_y)**2)
    if dist_to_gs < 3000 and dy > earth_radius:
        gs_text.set_text("GROUND TRACKING: ACTIVE")
    else:
        gs_text.set_text("GROUND TRACKING: INACTIVE")
        
    # Check Intersection (Space)
    dist_to_obs = np.sqrt((dx - ox)**2 + (dy - oy)**2)
    # Simple proximity check for visualizer
    if dist_to_obs < 400: 
        space_text.set_text("SPACE TRACKING: ACTIVE")
        space_text.set_color('yellow')
    else:
        space_text.set_text("SPACE TRACKING: INACTIVE")
        space_text.set_color('cyan')

    time_text.set_text(f"Simulation Time: {int(t*100)}%")
    return obs_dot, deb_dot, space_window, time_text, gs_text, space_text

# Create Animation
anim = FuncAnimation(fig, update, frames=duration_frames, interval=50, blit=True)

# Save as MP4 or GIF
print("Saving animation to tracking_comparison.mp4...")
try:
    anim.save('tracking_comparison.mp4', writer='ffmpeg', fps=30)
    print("Saved successfully!")
except Exception as e:
    print(f"Could not save MP4 (ensure ffmpeg is installed): {e}")
    print("Attempting to save as GIF instead...")
    anim.save('tracking_comparison.gif', writer='pillow', fps=30)
    print("GIF saved successfully!")

plt.close()