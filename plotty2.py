import numpy as np
import matplotlib.pyplot as plt

def generate_individual_page_plots(data_path="flight_telemetry.npz", burn_durations=[5.0, 5.0]):
    # 1. Load the binary array matrix from disk
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: The database file '{data_path}' was not found. Run your simulation script first.")
        return

    # Extract all standalone numpy arrays
    time = data["time"]
    altitude = data["altitude"]
    velocity = data["velocity"]
    thrust = data["thrust"]
    drag = data["drag"]
    gravity_loss = data["gravity_loss"]
    drag_loss = data["drag_loss"]
    velocity_margin = data["velocity_margin"]
    thrust_margin = data["thrust_margin"]
    specific_impulse = data["specific_impulse"]

    # Establish light theme properties globally
    plt.style.use('default')
    c_actual = "#0066CC"  # Professional blue 
    c_tsio = "#008837"    # Dark forest green 
    c_grav = "#D9381E"    # Clear red 
    c_drag = "#7B2CBF"    # Deep purple 

    # Calculate dynamic milestone indices for stage handoffs
    total_intervals = len(time)
    num_stages = len(burn_durations)
    intervals_per_stage = total_intervals // num_stages

    # =========================================================================
    # PAGE 1: GEOPOTENTIAL ALTITUDE VS TIME TRAJECTORY (WITH BOXES & MILESTONES)
    # =========================================================================
    fig0, ax0 = plt.subplots(figsize=(10, 6.5))
    ax0.plot(time, altitude / 1000.0, color=c_actual, linewidth=2.5, label=r"Trajectory Profile ($h$)")
    ax0.set_title("PAGE 1: GEOPOTENTIAL ALTITUDE TRAJECTORY PROFILE", fontsize=12, fontweight='bold', pad=15)
    ax0.set_xlabel("Elapsed Mission Time (seconds)", fontsize=10)
    ax0.set_ylabel("Altitude (kilometers)", fontsize=10)
    ax0.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")

    # Dynamic annotation engine for staging metrics
    for idx in range(num_stages):
        start_idx = idx * intervals_per_stage
        end_idx = total_intervals if idx == num_stages - 1 else (idx + 1) * intervals_per_stage
        
        stage_time_slice = time[start_idx:end_idx]
        stage_start_t = stage_time_slice[0]
        
        # 1. Detect and pinpoint engine burnout milestone
        burn_duration = burn_durations[idx]
        burnout_target_t = stage_start_t + burn_duration
        bo_idx = start_idx + np.abs(stage_time_slice - burnout_target_t).argmin()
        
        bo_t = time[bo_idx]
        bo_h = altitude[bo_idx] / 1000.0
        
        # Plot physical dot marker for burnout points
        ax0.scatter(bo_t, bo_h, color="#FF9900", edgecolor="black", s=60, zorder=5)
        ax0.annotate(f"Stage {idx+1}\nBurnout\nt={bo_t:.1f}s\nh={bo_h:.1f}km",
                     xy=(bo_t, bo_h),
                     xytext=(bo_t - 2.5, bo_h + 3.5),
                     arrowprops=dict(arrowstyle="->", color="#FF9900", lw=1.2),
                     fontsize=9, color="#B36B00", fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDF0", ec="#FF9900", alpha=0.9))

        # 2. Detect stage separation milestone right before sequence transitions
        if idx < num_stages - 1:
            sep_idx = end_idx - 1
            sep_t = time[sep_idx]
            sep_h = altitude[sep_idx] / 1000.0
            sep_v = velocity[sep_idx]
            sep_mass = data["inst_mass"][sep_idx] if "inst_mass" in data.files else 0.0
            
            # Plot marker cross for separation point
            ax0.scatter(sep_t, sep_h, color="#D9381E", edgecolor="black", s=70, marker="X", zorder=6)
            
            # Formulate full stage data package text layout inside the bounding box
            stage_data_box = (
                f"STAGE {idx+1} SEPARATION\n"
                f"-------------------\n"
                f"Time: {sep_t:.2f} s\n"
                f"Alt: {sep_h:.2f} km\n"
                f"Vel: {sep_v:.1f} m/s\n"
                f"Mass: {sep_mass:.1f} kg"
            )
            
            ax0.annotate(stage_data_box,
                         xy=(sep_t, sep_h),
                         xytext=(sep_t + 1.5, sep_h - 4.5),
                         arrowprops=dict(arrowstyle="->", color="#D9381E", lw=1.2),
                         fontsize=9, family='monospace',
                         bbox=dict(boxstyle="square,pad=0.5", fc="#FFF0F0", ec="#D9381E", lw=1.5, alpha=0.95))

    plt.tight_layout()
    fig0.savefig("report_page_1_altitude.png", dpi=300, facecolor="white")
    print("Page 1 generated: 'report_page_1_altitude.png'")


    # =========================================================================
    # PAGE 2: VELOCITY PROPULSION COMPONENTS VS TIME
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6.5))
    ax1.plot(time, velocity, color=c_actual, linewidth=2.5, label=r"Net Velocity ($V_{actual}$)")
    ax1.plot(time, drag_loss, color=c_drag, linestyle="-.", alpha=0.8, label=r"Cumulative Drag Loss ($V_{drag}$)")
    ax1.plot(time, gravity_loss, color=c_grav, linestyle=":", alpha=0.8, label=r"Cumulative Gravity Loss ($V_{grav}$)")
    ax1.plot(time, velocity_margin, color=c_tsio, linestyle="--", alpha=0.7, label=r"Ideal Capability ($V_{margin}$)")
    
    # Overlay dynamic secondary axis tracking active engine thrust force
    ax1_thrust = ax1.twinx()
    ax1_thrust.fill_between(time, thrust, color="#FFCC00", alpha=0.15, label="Thrust Force")
    ax1_thrust.set_ylabel("Engine Thrust Force (N)", color="#B38F00", fontsize=10)
    ax1_thrust.tick_params(colors="#B38F00")
    
    ax1.set_title("PAGE 2: VELOCITY COMPONENT DISTRIBUTIONS VS TIME DOMAIN", fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Elapsed Mission Time (seconds)", fontsize=10)
    ax1.set_ylabel("Velocity Vectors Dynamics (m/s)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper right", framealpha=0.9, facecolor="#F8F9FA")

    plt.tight_layout()
    fig1.savefig("report_page_2_velocity_time.png", dpi=300, facecolor="white")
    print("Page 2 generated: 'report_page_2_velocity_time.png'")


    # =========================================================================
    # PAGE 3: VELOCITY ATTACHMENT SPECTRUM VS ALTITUDE PROFILE
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    alt_km = altitude / 1000.0
    ax2.plot(alt_km, velocity, color=c_actual, linewidth=2.5, label=r"Net Velocity ($V_{actual}$)")
    ax2.plot(alt_km, drag_loss, color=c_drag, linestyle="-.", alpha=0.8, label=r"Cumulative Drag Loss ($V_{drag}$)")
    ax2.plot(alt_km, gravity_loss, color=c_grav, linestyle=":", alpha=0.8, label=r"Cumulative Gravity Loss ($V_{grav}$)")
    
    # Overlay dynamic aerodynamic force metrics vs altitude
    ax2_drag = ax2.twinx()
    ax2_drag.plot(alt_km, drag, color="#E65C00", linestyle="--", linewidth=1.2, label="Dynamic Drag")
    ax2_drag.set_ylabel("Aerodynamic Drag Force (N)", color="#E65C00", fontsize=10)
    ax2_drag.tick_params(colors="#E65C00")

    ax2.set_title("PAGE 3: VELOCITY TRAJECTORY PROFILE VS GEOPOTENTIAL ALTITUDE", fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel("Altitude Profile (kilometers)", fontsize=10)
    ax2.set_ylabel("Velocity Vectors Dynamics (m/s)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc="lower center", framealpha=0.9, facecolor="#F8F9FA")

    plt.tight_layout()
    fig2.savefig("report_page_3_velocity_altitude.png", dpi=300, facecolor="white")
    print("Page 3 generated: 'report_page_3_velocity_altitude.png'")

    # Show all pages sequentially in separate desktop plot viewports
    plt.show()

if __name__ == "__main__":
    generate_individual_page_plots(burn_durations=[5.0, 5.0])
