import numpy as np
import matplotlib.pyplot as plt

def generate_immersive_plots(data_path="flight_telemetry.npz"):
    # 1. Load the binary array matrix from disk
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: The database file '{data_path}' was not found. Please run your simulation first.")
        return

    # Extract standalone numpy arrays
    time = data["time"]
    altitude = data["altitude"]
    velocity = data["velocity"]
    thrust = data["thrust"]
    drag = data["drag"]
    gravity_loss = data["gravity_loss"]
    drag_loss = data["drag_loss"]

    # 2. Establish a high-fidelity dark-mode aerospace dashboard style
    plt.style.use('dark_background')
    
    # Custom neon hex codes for structural visibility
    c_actual = "#00F5FF"  # Cyan neon for net actual velocity
    c_tsio = "#00FF87"    # Bright green for ideal engine gain
    c_grav = "#FF5555"    # Muted red for gravity penalty
    c_drag = "#FF00FF"    # Neon magenta for aerodynamic drag loss

    # Create a side-by-side 1x2 subplot layout grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("MULTI-STAGE AEROSPACE TELEMETRY ANALYSIS DASHBOARD", 
                 fontsize=14, fontweight='bold', color='white', y=0.96)

    # ----------------------------------------------------
    # LEFT PANEL: COMPLETE VELOCITY PROPULSION PROFILE VS TIME
    # ----------------------------------------------------
    ax1.plot(time, velocity, color=c_actual, linewidth=2.5, label="Resultant Net Velocity ($V_{actual}$)")
    ax1.plot(time, drag_loss, color=c_drag, linestyle="-.", alpha=0.8, label="Cumulative Drag Loss ($V_{drag\_loss}$)")
    ax1.plot(time, gravity_loss, color=c_grav, linestyle=":", alpha=0.8, label="Cumulative Gravity Loss ($V_{grav\_loss}$)")
    
    # Add a secondary axis to overlay dynamic engine thrust force
    ax1_thrust = ax1.twinx()
    ax1_thrust.fill_between(time, thrust, color="#FFCC00", alpha=0.12, label="Engine Thrust Force ($F_{thrust}$)")
    ax1_thrust.set_ylabel("Engine Thrust Force (N)", color="#FFCC00", fontsize=10)
    ax1_thrust.tick_params(colors="#FFCC00")
    
    ax1.set_title("VELOCITY DECOMPOSITION OVER THE TIME DOMAIN", fontsize=11, color='white', pad=12)
    ax1.set_xlabel("Elapsed Mission Time (seconds)", fontsize=10)
    ax1.set_ylabel("Velocity Dynamics (m/s)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.25, color="gray")
    
    # Combine legend handles from both primary and secondary y-axes safely
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_t, labels1_t = ax1_thrust.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_t, labels1 + labels1_t, loc="upper right", framealpha=0.2)

    # ----------------------------------------------------
    # RIGHT PANEL: VELOCITY ATTENUATION SPECTRUM VS ALTITUDE
    # ----------------------------------------------------
    # Convert altitude to kilometers for cleaner labeling layout metrics
    alt_km = altitude / 1000.0

    ax2.plot(alt_km, velocity, color=c_actual, linewidth=2.5, label="Resultant Net Velocity ($V_{actual}$)")
    ax2.plot(alt_km, drag_loss, color=c_drag, linestyle="-.", alpha=0.8, label="Cumulative Drag Loss ($V_{drag\_loss}$)")
    ax2.plot(alt_km, gravity_loss, color=c_grav, linestyle=":", alpha=0.8, label="Cumulative Gravity Loss ($V_{grav\_loss}$)")
    
    # Add a secondary axis to show atmospheric aerodynamic drag force vs altitude
    ax2_drag = ax2.twinx()
    ax2_drag.plot(alt_km, drag, color="#FF6600", linestyle="--", linewidth=1.2, label="Aerodynamic Drag Force ($F_{drag}$)")
    ax2_drag.set_ylabel("Aerodynamic Drag Force (N)", color="#FF6600", fontsize=10)
    ax2_drag.tick_params(colors="#FF6600")

    ax2.set_title("VELOCITY TRAJECTORY PROFILE VS GEOPOTENTIAL ALTITUDE", fontsize=11, color='white', pad=12)
    ax2.set_xlabel("Altitude Profile (kilometers)", fontsize=10)
    ax2.set_ylabel("Velocity Dynamics (m/s)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.25, color="gray")
    
    # Combine legend handles for the second panel
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_d, labels2_d = ax2_drag.get_legend_handles_labels()
    ax2.legend(lines2 + lines2_d, labels2 + labels2_d, loc="lower center", framealpha=0.2)

    # 3. Clean alignment margins and display window interface
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save a high-resolution snapshot image to local workspace before popping window interface open
    plt.savefig("flight_telemetry_dashboard.png", dpi=300)
    print("Telemetry visualization successfully saved to: 'flight_telemetry_dashboard.png'")
    
    plt.show()

if __name__ == "__main__":
    generate_immersive_plots()
