import os
import json
import numpy as np
from rocket_physics import consts, Vectors, Time, Mass, Vel, Thrust, Displacement

# JSON multi-stage configuration profile updated with a stronger crossrange lateral push on the Y-axis
json_input_data = """
{
    "initial_time": 0.0,
    "runtime": 60.0,
    "intervals": 6000,
    "pitchover_time": 3.0,
    "pitchover_kick_vector": [0.25, 0.20, 0.93],
    "elements": {
        "stage_1": {
            "wet_mass": 450.0,
            "dry_mass": 150.0,
            "payload": 100.0,
            "burn_duration": 15.0,
            "Vj": 2900.0,
            "Cd": 0.22,
            "Area": 0.25,
            "apogee_config": 80000.0
        },
        "stage_2": {
            "wet_mass": 150.0,
            "dry_mass": 50.0,
            "payload": 20.0,
            "burn_duration": 20.0,
            "Vj": 3400.0,
            "Cd": 0.15,
            "Area": 0.12,
            "apogee_config": 250000.0
        }
    }
}
"""

if __name__ == "__main__":
    # Load and parse the setup JSON profile configuration matrix
    config = json.loads(json_input_data)
    
    total_intervals = config["intervals"]
    t_start = config["initial_time"]
    t_end = config["runtime"]
    pitchover_t = config["pitchover_time"]
    
    # Extracting the explicit 3-element array to drive authentic 3D spatial curves
    kick_dir = np.array(config["pitchover_kick_vector"], dtype=float)
    
    global_time_vec = np.linspace(t_start, t_end, total_intervals)
    
    # 3D trackers (shape initialized to (N, 3) for multi-dimensional telemetry channels)
    results = {
        "time": global_time_vec,
        "displacement": np.zeros((total_intervals, 3)),
        "velocity": np.zeros((total_intervals, 3)),
        "orientation_n": np.zeros((total_intervals, 3)), 
        "tsio_dv_gain": np.zeros((total_intervals, 3)),
        "grav_dv_loss": np.zeros((total_intervals, 3)),
        "drag_dv_loss": np.zeros((total_intervals, 3)),
        "inst_mass": np.zeros(total_intervals),
        "f_thrust": np.zeros((total_intervals, 3)),
        "f_drag": np.zeros((total_intervals, 3)),
        "f_grav": np.zeros((total_intervals, 3))
    }
    
    stages_data = config["elements"]
    stage_keys = list(stages_data.keys())
    num_stages = len(stage_keys)
    
    intervals_per_stage = total_intervals // num_stages
    
    # Initialize the rocket displacement vector system instance at origin
    rocket_displacement = Displacement(np.array([0.0, 0.0, 0.0]))
    current_vel = np.array([0.0, 0.0, 0.0]) 
    
    # Initialize 3D telemetry tracking vectors
    accumulated_grav_loss_vec = np.array([0.0, 0.0, 0.0])
    accumulated_drag_loss_vec = np.array([0.0, 0.0, 0.0])
    
    thrust_calculator = Thrust()
    vec_manager = Vectors() 
    
    for stage_idx, stage_name in enumerate(stage_keys):
        stage_cfg = stages_data[stage_name]
        
        start_idx = stage_idx * intervals_per_stage
        end_idx = total_intervals if stage_idx == num_stages - 1 else (stage_idx + 1) * intervals_per_stage
            
        stage_mass = Mass(dry_mass=stage_cfg["dry_mass"], wet_mass=stage_cfg["wet_mass"], payload=stage_cfg["payload"])
        stage_mass.calc_m_dot(stage_cfg["burn_duration"])
        
        stage_time_obj = Time(intervals=(end_idx - start_idx), end_time=global_time_vec[end_idx - 1], start_time=global_time_vec[start_idx])
        stage_vel_obj = Vel(initial_velocity=current_vel, obj_time=stage_time_obj)
        
        stage_start_time = global_time_vec[start_idx]
        
        for t in range(start_idx, end_idx):
            current_time = global_time_vec[t]
            current_alt = rocket_displacement.get_altitude()
            
            # Continuous mass flux step calculation
            inst_m = stage_mass.get_inst_mass(current_time, stage_start_time, stage_cfg["burn_duration"])
            
            # Dynamically compute orientation unit vector (n) for this frame step block
            n_hat = vec_manager.get_pointing_vector(current_time, pitchover_t, kick_dir, current_vel)
            
            # Vector forces calculations mapping orientations rules
            f_thrust = thrust_calculator.calc_time_varying_inertial(stage_mass, current_time, stage_start_time, stage_cfg["burn_duration"], stage_cfg["Vj"], n_hat)
            f_drag = thrust_calculator.calc_drag_force(stage_cfg["Cd"], stage_cfg["Area"], current_vel, current_alt, n_hat)
            f_grav = thrust_calculator.calc_grav_force(current_alt, inst_m, vec_manager.k)
            
            # Log frames into multidimensional dictionary arrays configuration cleanly
            results["displacement"][t] = rocket_displacement.vector
            results["velocity"][t] = current_vel
            results["orientation_n"][t] = n_hat
            results["inst_mass"][t] = inst_m
            results["f_thrust"][t] = f_thrust
            results["f_drag"][t] = f_drag
            results["f_grav"][t] = f_grav
            
            # Submitting vector parameters to the upgraded loss tracking methods
            results["tsio_dv_gain"][t] = stage_vel_obj.calc_tsio_del_vel(stage_cfg["Vj"], inst_m, stage_cfg["wet_mass"], n_hat)
            results["grav_dv_loss"][t] = stage_vel_obj.calc_grav_del_vel(accumulated_grav_loss_vec)
            results["drag_dv_loss"][t] = stage_vel_obj.calc_drag_del_vel(accumulated_drag_loss_vec)
            
            # Integrate kinematics forward into the next step frame boundaries
            if t < total_intervals - 1:
                step_dt = float(global_time_vec[t+1] - current_time)
                
                # Check launch threshold parameters using vertical axis and thrust elements
                if current_alt == 0.0 and np.linalg.norm(f_thrust) <= np.linalg.norm(f_grav):
                    acceleration = np.zeros(3)
                    current_vel = np.zeros(3)
                else:
                    # 3D Vector summing forces expression
                    net_force = f_thrust + f_drag + f_grav
                    acceleration = net_force / inst_m
                    
                    # Accumulate 3D loss integrations over time frames
                    accumulated_grav_loss_vec += (f_grav / inst_m) * step_dt
                    accumulated_drag_loss_vec += (f_drag / inst_m) * step_dt
                    
                    current_vel += acceleration * step_dt
                
                # Dynamic vector displacement state shift
                rocket_displacement.update_position(current_vel, step_dt)
                
                # Clamp ground boundaries immediately after updating spatial positions
                if rocket_displacement.get_altitude() < 0.0:
                    rocket_displacement.reset_ground_limit()
                    if current_vel[2] < 0.0:
                        current_vel[2] = 0.0  # Arrest downward velocity on vertical Z component channel

    # Enforce safe folder handling using the os module wrapper
    os.makedirs("simulated_data", exist_ok=True)
    
    # Save the complete multi-dimensional array matrix package payload to disk
    np.savez(
        "simulated_data/flight_telemetry_gravity_turn_3d_displacement.npz", 
        time=results["time"], 
        displacement=results["displacement"], 
        velocity=results["velocity"], 
        orientation=results["orientation_n"],
        thrust=results["f_thrust"], 
        drag=results["f_drag"],
        gravity=results["f_grav"],
        tsio_dv_gain=results["tsio_dv_gain"], 
        gravity_loss=results["grav_dv_loss"], 
        drag_loss=results["drag_dv_loss"],
    )
    
    print("--- 3D GRAVITY TURN KINEMATIC INTEGRATION COMPLETE ---")
    print(f"Max Altitude Achieved (Z component): {np.max(results['displacement'][:, 2]):.2f} meters")
    print(f"Final Downrange (X axis): {results['displacement'][-1, 0]:.2f} meters")
    print(f"Final Crossrange (Y axis): {results['displacement'][-1, 1]:.2f} meters")
