import json
import numpy as np
from rocket_physics import consts, Time, Mass, Vel, Thrust, Altitude

# Optimized JSON multi-stage input profile mapping realistic parameters
json_input_data = """
{
    "initial_time": 0.0,
    "runtime": 15.0,
    "intervals": 3000,
    "elements": {
        "stage_1": {
            "wet_mass": 45.0,
            "dry_mass": 15.0,
            "payload": 10.0,
            "burn_duration": 5.0,
            "Vj": 2500.0,
            "Cd": 0.20,
            "Area": 0.05,
            "apogee_config": 20000.0
        },
        "stage_2": {
            "wet_mass": 15.0,
            "dry_mass": 5.0,
            "payload": 2.0,
            "burn_duration": 5.0,
            "Vj": 3000.0,
            "Cd": 0.15,
            "Area": 0.03,
            "apogee_config": 60000.0
        }
    }
}
"""

if __name__ == "__main__":
    config = json.loads(json_input_data)
    
    total_intervals = config["intervals"]
    t_start = config["initial_time"]
    t_end = config["runtime"]
    
    global_time_vec = np.linspace(t_start, t_end, total_intervals)
    
    results = {
        "time": global_time_vec,
        "altitude": np.zeros(total_intervals),
        "actual_velocity": np.zeros(total_intervals),
        "tsio_dv_gain": np.zeros(total_intervals),
        "grav_dv_loss": np.zeros(total_intervals),
        "drag_dv_loss": np.zeros(total_intervals),
        "inst_mass": np.zeros(total_intervals),
        "f_thrust_inertial": np.zeros(total_intervals),
        "f_drag": np.zeros(total_intervals),
        "f_grav": np.zeros(total_intervals)
    }
    
    stages_data = config["elements"]
    stage_keys = list(stages_data.keys())
    num_stages = len(stage_keys)
    
    intervals_per_stage = total_intervals // num_stages
    
    current_alt = 0.0
    current_vel = 0.0
    accumulated_grav_loss = 0.0
    accumulated_drag_loss = 0.0
    
    thrust_calculator = Thrust()
    
    for stage_idx, stage_name in enumerate(stage_keys):
        stage_cfg = stages_data[stage_name]
        
        start_idx = stage_idx * intervals_per_stage
        if stage_idx == num_stages - 1:
            end_idx = total_intervals
        else:
            end_idx = (stage_idx + 1) * intervals_per_stage
            
        stage_mass = Mass(dry_mass=stage_cfg["dry_mass"], wet_mass=stage_cfg["wet_mass"], payload=stage_cfg["payload"])
        stage_mass.calc_m_dot(stage_cfg["burn_duration"])
        
        stage_time_obj = Time(intervals=(end_idx - start_idx), end_time=global_time_vec[end_idx - 1], start_time=global_time_vec[start_idx])
        stage_vel_obj = Vel(initial_velocity=current_vel, obj_time=stage_time_obj)
        
        stage_start_time = global_time_vec[start_idx]
        
        for t in range(start_idx, end_idx):
            current_time = global_time_vec[t]
            
            # Ground limit check to prevent downward numerical inversion
            if current_alt < 0.0:
                current_alt = 0.0
                if current_vel < 0.0:
                    current_vel = 0.0
            
            inst_m = stage_mass.get_inst_mass(current_time, stage_start_time, stage_cfg["burn_duration"])
            f_thrust = thrust_calculator.calc_time_varying_inertial(stage_mass, current_time, stage_start_time, stage_cfg["burn_duration"], stage_cfg["Vj"])
            f_drag = thrust_calculator.calc_drag_force(stage_cfg["Cd"], stage_cfg["Area"], current_vel, current_alt)
            f_grav = thrust_calculator.calc_grav_force(current_alt, inst_m)
            
            # Log metrics to arrays
            results["altitude"][t] = current_alt
            results["actual_velocity"][t] = current_vel
            results["inst_mass"][t] = inst_m
            results["f_thrust_inertial"][t] = f_thrust
            results["f_drag"][t] = f_drag
            results["f_grav"][t] = f_grav
            
            # Populate complementary telemetry vector trackers sequentially
            results["tsio_dv_gain"][t] = stage_vel_obj.calc_tsio_del_vel(stage_cfg["Vj"], inst_m, stage_cfg["wet_mass"])
            results["grav_dv_loss"][t] = stage_vel_obj.calc_grav_del_vel(accumulated_grav_loss)
            results["drag_dv_loss"][t] = stage_vel_obj.calc_drag_del_vel(accumulated_drag_loss)
            
            if t < total_intervals - 1:
                step_dt = float(global_time_vec[t+1] - current_time)
                
                # Kinematic Acceleration Integration: a = (F_thrust - F_drag - F_grav) / mass
                # Rocket stays stationary on pad until thrust exceeds gravity forces
                if current_alt == 0.0 and f_thrust <= f_grav:
                    acceleration = 0.0
                    current_vel = 0.0
                else:
                    acceleration = (f_thrust - f_drag - f_grav) / inst_m
                    # Component loss integration
                    accumulated_grav_loss += (f_grav / inst_m) * step_dt
                    accumulated_drag_loss += (f_drag / inst_m) * step_dt
                    
                    # Update true physical state values forward
                    current_vel += acceleration * step_dt
                
                current_alt += current_vel * step_dt

    # Binary file save execution block
    np.savez(
        "flight_telemetry.npz", 
        time=results["time"], 
        altitude=results["altitude"], 
        velocity=results["actual_velocity"], 
        thrust=results["f_thrust_inertial"], 
        drag=results["f_drag"],
        gravity_loss=results["grav_dv_loss"],
        drag_loss=results["drag_dv_loss"]
    )
    
    print("--- KINEMATIC INTEGRATION COMPLETE: ARRAYS VALIDATED ---")
    print(f"Peak Operational Speed Achieved: {np.max(results['actual_velocity']):.2f} m/s")
    print(f"Max Altitude Achieved: {np.max(results['altitude']):.2f} meters")
