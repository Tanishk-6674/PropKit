# simulation.py - PART 1
import os
import json
import numpy as np
from rocket_physics import consts, Vectors, Time, Mass, Vel, Thrust, Displacement

# JSON configuration structured with high-frequency chugging profiles
json_input_data = """
{
    "initial_time": 0.0,
    "runtime": 120.0,
    "intervals": 12000,
    "pitchover_time": 3.0,
    "pitchover_kick_vector": [0.25, 0.20, 0.93],
    "elements": {
        "stage_1": {
            "wet_mass": 350.0,
            "dry_mass": 100.0,
            "motor_firing_delay": 1.5,
            "ignition_coefficient": 0.06,
            "burn_duration": 15.0,
            "burning_type": {
                "type": "expression",
                "rate": "(1.0 + 0.02 * t) + 0.12 * sin(2.0 * pi * 8.0 * t)"
            },
            "sliver_coefficient": 0.05,
            "separation_delay": 5.0,
            "Vj": 2900.0,
            "Cd": 0.22,
            "Area": 0.25
        },
        "stage_2": {
            "wet_mass": 120.0,
            "dry_mass": 40.0,
            "motor_firing_delay": 1.0,
            "ignition_coefficient": 0.04,
            "burn_duration": 20.0,
            "burning_type": {
                "type": "expression",
                "rate": "1.0 + 0.08 * sin(2.0 * pi * 12.0 * t)"
            },
            "sliver_coefficient": 0.08,
            "separation_delay": 2.0,
            "Vj": 3400.0,
            "Cd": 0.15,
            "Area": 0.12
        },
        "payload_bus": {
            "wet_mass": 30.0,
            "dry_mass": 30.0,
            "motor_firing_delay": 0.0,
            "burn_duration": 0.0,
            "separation_delay": 0.0,
            "Vj": 0.0,
            "Cd": 0.10,
            "Area": 0.05
        }
    }
}
"""

def calculate_remaining_mass(current_stage_idx: int, stage_keys: list, stages_data: dict) -> float:
    """Safely calculates collective mass of upper stages sitting above active index."""
    remaining_mass = 0.0
    for upper_idx in range(current_stage_idx + 1, len(stage_keys)):
        upper_name = stage_keys[upper_idx]
        remaining_mass += stages_data[upper_name]["wet_mass"]
    return remaining_mass

def evaluate_stage_transitions(current_time: float, current_stage_idx: int, stage_start_time: float, 
                               stage_keys: list, stages_data: dict) -> tuple:
    """Manages the staging timeline clock windows and advances active stage pointer."""
    if current_stage_idx >= len(stage_keys):
        return current_stage_idx, stage_start_time, None
    current_stage_name = stage_keys[current_stage_idx]
    stage_cfg = stages_data[current_stage_name]
    t_delay = stage_cfg.get("motor_firing_delay", 0.0)
    t_burn = stage_cfg["burn_duration"]
    t_sep = stage_cfg.get("separation_delay", 0.0)
    total_stage_window = t_delay + t_burn + t_sep
    elapsed_stage_time = current_time - stage_start_time
    if elapsed_stage_time > total_stage_window:
        if current_stage_idx + 1 < len(stage_keys):
            current_stage_idx += 1
            stage_start_time = current_time
            current_stage_name = stage_keys[current_stage_idx]
            stage_cfg = stages_data[current_stage_name]
        else:
            stage_cfg = None
    return current_stage_idx, stage_start_time, stage_cfg
# simulation.py - PART 2
def compute_flight_physics(current_time: float, current_alt: float, current_vel: np.ndarray, stage_cfg: dict, 
                           stage_mass: Mass, global_remaining_mass: float, stage_start_time: float, 
                           pitchover_t: float, kick_dir: np.ndarray, stages_data: dict, stage_keys: list,
                           thrust_calculator: Thrust, vec_manager: Vectors, vel_logger: Vel) -> tuple:
    """Routes state data directly to rocket_physics methods to generate force vectors."""
    if stage_cfg is not None and stage_cfg["burn_duration"] > 0.0:
        elapsed_stage_time = current_time - stage_start_time
        n_hat = vec_manager.get_pointing_vector(current_time, pitchover_t, kick_dir, current_vel)
        inst_m, current_m_flux = stage_mass.get_advanced_mass_and_flux(elapsed_stage_time, stage_cfg, global_remaining_mass)
        f_thrust = thrust_calculator.calc_time_varying_inertial(current_m_flux, stage_cfg["Vj"], n_hat)
        f_drag = thrust_calculator.calc_drag_force(stage_cfg["Cd"], stage_cfg["Area"], current_vel, current_alt, n_hat)
        dv_gain = vel_logger.calc_tsio_del_vel(stage_cfg["Vj"], inst_m, stage_cfg["wet_mass"] + global_remaining_mass, n_hat)
    else:
        final_stage_name = stage_keys[-1]
        inst_m = stages_data[final_stage_name]["dry_mass"]
        n_hat = current_vel / np.linalg.norm(current_vel) if np.linalg.norm(current_vel) > 0 else vec_manager.k
        f_thrust = np.zeros(3)
        f_drag = thrust_calculator.calc_drag_force(stages_data[final_stage_name]["Cd"], stages_data[final_stage_name]["Area"], current_vel, current_alt, n_hat)
        dv_gain = np.zeros(3)
    f_grav = thrust_calculator.calc_grav_force(rocket_displacement.vector, inst_m)
    return inst_m, n_hat, f_thrust, f_drag, f_grav, dv_gain

def log_telemetry_frame(t: int, results: dict, displacement_vec: np.ndarray, velocity_vec: np.ndarray, 
                        n_hat: np.ndarray, inst_m: float, f_thrust: np.ndarray, f_drag: np.ndarray, 
                        f_grav: np.ndarray, dv_gain: np.ndarray, vel_logger: Vel, acc_grav: np.ndarray, acc_drag: np.ndarray):
    """Updates telemetry dictionary arrays for data storage."""
    results["displacement"][t], results["velocity"][t], results["orientation_n"][t] = displacement_vec, velocity_vec, n_hat
    results["inst_mass"][t], results["f_thrust"][t], results["f_drag"][t], results["f_grav"][t] = inst_m, f_thrust, f_drag, f_grav
    results["tsio_dv_gain"][t] = dv_gain
    results["grav_dv_loss"][t] = vel_logger.calc_grav_del_vel(acc_grav)
    results["drag_dv_loss"][t] = vel_logger.calc_drag_del_vel(acc_drag)

if __name__ == "__main__":
    config = json.loads(json_input_data)
    total_intervals, t_start, t_end, pitchover_t = config["intervals"], config["initial_time"], config["runtime"], config["pitchover_time"]
    kick_dir = np.array(config["pitchover_kick_vector"], dtype=float)
    global_time_vec = np.linspace(t_start, t_end, total_intervals)
    results = {
        "time": global_time_vec, "displacement": np.zeros((total_intervals, 3)), "velocity": np.zeros((total_intervals, 3)),
        "orientation_n": np.zeros((total_intervals, 3)), "tsio_dv_gain": np.zeros((total_intervals, 3)), "grav_dv_loss": np.zeros((total_intervals, 3)),
        "drag_dv_loss": np.zeros((total_intervals, 3)), "inst_mass": np.zeros(total_intervals), "f_thrust": np.zeros((total_intervals, 3)),
        "f_drag": np.zeros((total_intervals, 3)), "f_grav": np.zeros((total_intervals, 3))
    }
    stages_data = config["elements"]
    stage_keys = list(stages_data.keys())
    stage_objects = {name: Mass(dry_mass=stages_data[name]["dry_mass"], wet_mass=stages_data[name]["wet_mass"], payload=0.0) for name in stage_keys}
    current_stage_idx, stage_start_time = 0, 0.0
    rocket_displacement = Displacement(np.array([0.0, 0.0, 0.0]))
    current_vel, accumulated_grav_loss_vec, accumulated_drag_loss_vec = np.zeros(3), np.zeros(3), np.zeros(3)
    thrust_calculator, vec_manager, global_time_obj = Thrust(), Vectors(), Time(intervals=total_intervals, end_time=t_end, start_time=t_start)
    vel_logger = Vel(initial_velocity=current_vel, obj_time=global_time_obj)

    for t in range(total_intervals):
        current_time, current_alt = global_time_vec[t], rocket_displacement.get_altitude()
        global_remaining_mass = calculate_remaining_mass(current_stage_idx, stage_keys, stages_data)
        current_stage_idx, stage_start_time, stage_cfg = evaluate_stage_transitions(current_time, current_stage_idx, stage_start_time, stage_keys, stages_data)
        active_mass_obj = stage_objects[stage_keys[current_stage_idx]] if current_stage_idx < len(stage_keys) else None
        inst_m, n_hat, f_thrust, f_drag, f_grav, dv_gain = compute_flight_physics(current_time, current_alt, current_vel, stage_cfg, active_mass_obj, global_remaining_mass, stage_start_time, pitchover_t, kick_dir, stages_data, stage_keys, thrust_calculator, vec_manager, vel_logger)
        log_telemetry_frame(t, results, rocket_displacement.vector, current_vel, n_hat, inst_m, f_thrust, f_drag, f_grav, dv_gain, vel_logger, accumulated_grav_loss_vec, accumulated_drag_loss_vec)
        if t < total_intervals - 1:
            step_dt = float(global_time_vec[t+1] - current_time)
            if current_alt == 0.0 and np.linalg.norm(f_thrust) <= np.linalg.norm(f_grav):
                acceleration, current_vel = np.zeros(3), np.zeros(3)
            else:
                acceleration = (f_thrust + f_drag + f_grav) / inst_m
                accumulated_grav_loss_vec += (f_grav / inst_m) * step_dt
                accumulated_drag_loss_vec += (f_drag / inst_m) * step_dt
                current_vel += acceleration * step_dt
            rocket_displacement.update_position(current_vel, step_dt)
            if rocket_displacement.get_altitude() < 0.0:
                rocket_displacement.reset_ground_limit()
                if current_vel[2] < 0.0: current_vel[2] = 0.0

    os.makedirs("simulated_data", exist_ok=True)
    np.savez("simulated_data/flight_telemetry_gravity_turn_3d_displacement.npz", time=results["time"], displacement=results["displacement"], velocity=results["velocity"], orientation=results["orientation_n"], thrust=results["f_thrust"], drag=results["f_drag"], gravity=results["f_grav"], tsio_dv_gain=results["tsio_dv_gain"], gravity_loss=results["grav_dv_loss"], drag_loss=results["drag_dv_loss"], inst_mass=results["inst_mass"])
    print("--- DECOUPLED COMPONENT MODULAR SIMULATION DONE ---")
