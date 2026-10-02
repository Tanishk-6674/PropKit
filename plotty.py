import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

def generate_separated_animations(data_path="simulated_data/flight_telemetry_gravity_turn_3d_displacement.npz", step_interval=15):
    gif_dir = "simulated_data/GIF"
    mpeg_dir = "simulated_data/mpeg"
    
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(mpeg_dir, exist_ok=True)
    
    # 1. Load data payload from disk
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: The database file '{data_path}' was not found. Please run your simulation.py script first.")
        return

    time = data["time"]
    disp = data["displacement"]  
    vel = data["velocity"]        
    thrust = data["thrust"]      
    drag = data["drag"]          
    
    velocity_mag = np.linalg.norm(vel, axis=1)
    thrust_mag = np.linalg.norm(thrust, axis=1)
    drag_mag = np.linalg.norm(drag, axis=1)
    
    gravity_loss_mag = np.linalg.norm(data["gravity_loss"], axis=1)
    drag_loss_mag = np.linalg.norm(data["drag_loss"], axis=1)
    alt_km = disp[:, 2] / 1000.0  

    plt.style.use('dark_background')
    frames_range = list(range(1, len(time), step_interval))

    # PURE PYTHON CORE SAVER FUNCTION WITH GRAPHICS MEMORY FLUSHING
    def compile_and_save_dual(fig, update_fn, base_filename):
        gif_path = os.path.join(gif_dir, f"{base_filename}.gif")
        mp4_path = os.path.join(mpeg_dir, f"{base_filename}.mp4")
        
        frames = []
        print(f" -> Rendering frames memory buffers for '{base_filename}'...")
        
        for f in frames_range:
            update_fn(f)
            
            # CRITICAL FIX: Forces Matplotlib to completely redraw the graphics elements 
            # and flush event buffers before capturing the pixels.
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Read from the updated buffer interface safely
            rgba_buffer = fig.canvas.buffer_rgba()
            image = np.asarray(rgba_buffer, dtype='uint8')
            
            # Slice off alpha transparency layer to form clean RGB frames
            frames.append(image[:, :, :3].copy())
        
        # Save output assets using imageio
        print(f" -> Writing GIF asset: {gif_path}")
        imageio.mimsave(gif_path, frames, duration=1.0/15.0, loop=0)
        
        print(f" -> Writing MP4 asset: {mp4_path}")
        try:
            imageio.mimsave(mp4_path, frames, fps=15, codec='libx264', quality=8)
        except Exception as err:
            print(f"   [Notice]: MP4 render error: {err}")

    # =========================================================================
    # ANIMATION 1: 3D SPATIAL DISPLACEMENT PROFILE (TRAJECTORY)
    # =========================================================================
    print("\nProcessing 1/3: 3D Spatial Trajectory Tracking Channels...")
    # FIXED: Aspect ratio sizing to prevent divisible-by-16 macro_block warnings
    fig1 = plt.figure(figsize=(8, 6.4))
    ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
    fig1.suptitle("3D FLIGHT SPATIAL TRAJECTORY TRACKER", fontsize=12, fontweight='bold')

    line3d, = ax1.plot([], [], [], color="#00F5FF", linewidth=3, label="Flight Path")
    point3d, = ax1.plot([], [], [], color='white', marker='o', markersize=6)

    ax1.set_xlabel("Downrange X (m)")
    ax1.set_ylabel("Crossrange Y (m)")
    ax1.set_zlabel("Altitude Z (m)")
    ax1.grid(True, linestyle=":", alpha=0.15)

    x_min, x_max = np.min(disp[:, 0]), np.max(disp[:, 0])
    y_min, y_max = np.min(disp[:, 1]), np.max(disp[:, 1])
    z_min, z_max = np.min(disp[:, 2]), np.max(disp[:, 2])

    if x_min == x_max: x_min, x_max = -10.0, 10.0
    if y_min == y_max: y_min, y_max = -10.0, 10.0
    if z_min == z_max: z_min, z_max = 0.0, 100.0

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.view_init(elev=20, azim=-45)

    def update_3d(frame):
        if frame > 0:
            line3d.set_data(disp[:frame, 0], disp[:frame, 1])
            line3d.set_3d_properties(disp[:frame, 2])
            point3d.set_data([disp[frame-1, 0]], [disp[frame-1, 1]])
            point3d.set_3d_properties([disp[frame-1, 2]])

    compile_and_save_dual(fig1, update_3d, "flight_trajectory_3d")
    plt.close(fig1)

    # =========================================================================
    # ANIMATION 2: VELOCITY DECOMPOSITION OVER TIME
    # =========================================================================
    print("\nProcessing 2/3: Time-Domain Velocity Dynamics...")
    # FIXED: Aspect ratio sizing to prevent divisible-by-16 macro_block warnings
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    fig2.suptitle("VELOCITY DECOMPOSITION OVER TIME", fontsize=12, fontweight='bold')

    line_v_t, = ax2.plot([], [], color="#00F5FF", linewidth=2, label=r"$V_{actual}$")
    line_drag_t, = ax2.plot([], [], color="#FF00FF", linestyle="-.", label=r"$V_{drag\_loss}$")
    line_grav_t, = ax2.plot([], [], color="#FF5555", linestyle=":", label=r"$V_{grav\_loss}$")
    
    ax2_thrust = ax2.twinx()
    fill_thrust = ax2_thrust.fill_between([], [], color="#FFCC00", alpha=0.15, label=r"$F_{thrust}$")
    ax2_thrust.set_ylabel("Thrust Force (N)", color="#FFCC00")
    ax2_thrust.tick_params(colors="#FFCC00")

    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Velocity Dynamics (m/s)")
    ax2.set_xlim(np.min(time), np.max(time))
    ax2.set_ylim(0, np.max(velocity_mag) * 1.1)
    ax2_thrust.set_ylim(0, np.max(thrust_mag) * 1.1)
    ax2.grid(True, linestyle=":", alpha=0.2)
    ax2.legend(loc="upper left", framealpha=0.1)

    def update_time(frame):
        # Declare nonlocal to handle the clean replacement of fill_between polygons
        # inside an independent drawing script step loop
        nonlocal fill_thrust
        t_slice = time[:frame]
        line_v_t.set_data(t_slice, velocity_mag[:frame])
        line_drag_t.set_data(t_slice, drag_loss_mag[:frame])
        line_grav_t.set_data(t_slice, gravity_loss_mag[:frame])
        
        fill_thrust.remove()
        fill_thrust = ax2_thrust.fill_between(t_slice, thrust_mag[:frame], color="#FFCC00", alpha=0.12)

    compile_and_save_dual(fig2, update_time, "velocity_vs_time")
    plt.close(fig2)

    # =========================================================================
    # ANIMATION 3: VELOCITY TRAJECTORY VS GEOPOTENTIAL ALTITUDE
    # =========================================================================
    print("\nProcessing 3/3: Altitude-Domain Vector Profile...")
    # FIXED: Aspect ratio sizing to prevent divisible-by-16 macro_block warnings
    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    fig3.suptitle("VELOCITY TRAJECTORY PROFILE VS GEOPOTENTIAL ALTITUDE", fontsize=12, fontweight='bold')

    line_v_a, = ax3.plot([], [], color="#00F5FF", linewidth=2, label=r"$V_{actual}$")
    line_drag_a, = ax3.plot([], [], color="#FF00FF", linestyle="-.", label=r"$V_{drag\_loss}$")
    line_grav_a, = ax3.plot([], [], color="#FF5555", linestyle=":", label=r"$V_{grav\_loss}$")
    
    ax3_drag = ax3.twinx()
    line_f_drag, = ax3_drag.plot([], [], color="#FF6600", linestyle="--", linewidth=1.2, label=r"$F_{drag}$")
    ax3_drag.set_ylabel("Aerodynamic Drag Force (N)", color="#FF6600")
    ax3_drag.tick_params(colors="#FF6600")

    ax3.set_xlabel("Altitude Profile (km)")
    ax3.set_ylabel("Velocity Dynamics (m/s)")
    ax3.set_xlim(0, np.max(alt_km) * 1.1)
    ax3.set_ylim(0, np.max(velocity_mag) * 1.1)
    ax3_drag.set_ylim(0, np.max(drag_mag) * 1.1)
    ax3.grid(True, linestyle=":", alpha=0.2)
    ax3.legend(loc="lower left", framealpha=0.1)

    def update_alt(frame):
        alt_slice = alt_km[:frame]
        line_v_a.set_data(alt_slice, velocity_mag[:frame])
        line_drag_a.set_data(alt_slice, drag_loss_mag[:frame])
        line_grav_a.set_data(alt_slice, gravity_loss_mag[:frame])
        line_f_drag.set_data(alt_slice, drag_mag[:frame])

    compile_and_save_dual(fig3, update_alt, "velocity_vs_altitude")
    plt.close(fig3)

    print("\n========================================================")
    print("SUCCESS: SYSTEM DUAL EXECUTION FILE COMPILATION COMPLETE")
    print("========================================================")
    print(" -> GIFs saved to: simulated_data/GIF/")
    print(" -> MP4s saved to: simulated_data/mpeg/")

if __name__ == "__main__":
    generate_separated_animations(step_interval=15)
