# rocket_physics.py
import numpy as np

SAFE_MATH = {
    "np": np, "exp": np.exp, "log": np.log, "sin": np.sin, 
    "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi
}

class consts:
    G = 6.674e-11
    ME = 5.97e+24
    RE = 6.371e6
    g = 9.81
    rho_0 = 1.225     
    Zfactor = 8500.0  

class Vectors:
    def __init__(self):
        self.i = np.array([1.0, 0.0, 0.0])
        self.j = np.array([0.0, 1.0, 0.0])
        self.k = np.array([0.0, 0.0, 1.0])
        
    def get_pointing_vector(self, current_time: float, pitchover_time: float, kick_vector: np.ndarray, velocity_vec: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity_vec)
        if current_time < pitchover_time:
            return self.k
        elif current_time >= pitchover_time and speed < 1.0:
            norm_kick = np.linalg.norm(kick_vector)
            return kick_vector / norm_kick if norm_kick > 0 else self.k
        else:
            return velocity_vec / speed

class Time:
    def __init__(self, intervals: int, end_time: float, start_time: float = 0.0):
        self.intervals = intervals
        self.endtime = end_time
        self.start_time = start_time
        self.burn_out = 0.0
        self.ignition = 0.0
        self.time_vec = np.linspace(self.start_time, self.endtime, self.intervals)

class Mass:
    def __init__(self, dry_mass: float, wet_mass: float, payload: float):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.payload_mass = payload
        self.compiled_rate_code = None 

    def get_advanced_mass_and_flux(self, elapsed_stage_time: float, stage_cfg: dict, global_remaining_mass: float) -> tuple:
        t_delay = stage_cfg.get("motor_firing_delay", 0.0)
        t_burn = stage_cfg["burn_duration"]
        
        m_prop = self.wet_mass - self.dry_mass
        sliver_coeff = stage_cfg.get("sliver_coefficient", 0.0)
        m_sliver = m_prop * sliver_coeff
        m_nominal_prop = m_prop - m_sliver

        if elapsed_stage_time <= t_delay:
            return (self.wet_mass + global_remaining_mass), 0.0

        elif elapsed_stage_time <= (t_delay + t_burn):
            t_burn_elapsed = elapsed_stage_time - t_delay
            t_sliver_entry = t_burn * (1.0 - sliver_coeff)
            m_dot_base = m_nominal_prop / t_burn
            
            if self.compiled_rate_code is None:
                b_cfg = stage_cfg.get("burning_type", {"type": "expression", "rate": "1.0"})
                expr_string = str(b_cfg.get("rate", "1.0"))
                self.compiled_rate_code = compile(expr_string, "<string>", "eval")
            
            scope_vars = {"t": t_burn_elapsed, **SAFE_MATH}
            try:
                shape_multiplier = float(eval(self.compiled_rate_code, {"__builtins__": None}, scope_vars))
            except:
                shape_multiplier = 1.0
            
            ign_coeff = stage_cfg.get("ignition_coefficient", 0.0)
            ignition_ramp = (1.0 - np.exp(-t_burn_elapsed / (t_burn * ign_coeff))) if ign_coeff > 0.0 else 1.0
            
            if t_burn_elapsed <= t_sliver_entry:
                m_flux_current = m_dot_base * shape_multiplier * ignition_ramp
                consumed_mass = m_dot_base * t_burn_elapsed * ((1.0 + shape_multiplier) / 2.0) * ignition_ramp
                inst_m = (self.wet_mass - consumed_mass) + global_remaining_mass
                return inst_m, m_flux_current
            else:
                t_in_sliver = t_burn_elapsed - t_sliver_entry
                tau_sliver = t_burn * sliver_coeff * 0.4
                sliver_decay = np.exp(-t_in_sliver / tau_sliver)
                
                scope_vars_entry = {"t": t_sliver_entry, **SAFE_MATH}
                try:
                    shape_at_entry = float(eval(self.compiled_rate_code, {"__builtins__": None}, scope_vars_entry))
                except:
                    shape_at_entry = 1.0
                    
                m_flux_current = m_dot_base * shape_at_entry * sliver_decay * ignition_ramp
                mass_at_entry = self.wet_mass - (m_dot_base * t_sliver_entry * ((1.0 + shape_at_entry) / 2.0) * ignition_ramp)
                inst_m = self.dry_mass + (mass_at_entry - self.dry_mass) * sliver_decay + global_remaining_mass
                return inst_m, m_flux_current
        else:
            return global_remaining_mass, 0.0

class Vel:
    def __init__(self, initial_velocity: np.ndarray, obj_time: Time):
        self.run_time = obj_time.time_vec
        self.u_velocity = np.array(initial_velocity, dtype=float) 
        self.tsio_delta_V = np.zeros(3)
        self.grav_delta_V = np.zeros(3)
        self.drag_delta_v_val = np.zeros(3)

    def calc_tsio_del_vel(self, Vj: float, inst_mass: float, stage_wet_mass: float, n_vector: np.ndarray):
        magnitude = Vj * np.log(stage_wet_mass / max(1e-5, inst_mass))
        self.tsio_delta_V = magnitude * n_vector
        return self.tsio_delta_V

    def calc_grav_del_vel(self, accumulated_grav_loss_vec: np.ndarray):
        self.grav_delta_V = accumulated_grav_loss_vec
        return self.grav_delta_V

    def calc_drag_del_vel(self, accumulated_drag_loss_vec: np.ndarray):
        self.drag_delta_v_val = accumulated_drag_loss_vec
        return self.drag_delta_v_val

class Thrust:
    def calc_time_varying_inertial(self, current_m_flux: float, Vj: float, n_vector: np.ndarray):
        return current_m_flux * Vj * n_vector 

    def calc_drag_force(self, Cd_nose: float, Area_nose: float, velocity_vec: np.ndarray, 
                        current_alt: float, n_hat: np.ndarray) -> np.ndarray:
        """Advanced aerodynamic model with dynamic cross-section scaling and Mach wave drag spikes."""
        speed = np.linalg.norm(velocity_vec)
        if speed < 1e-5 or current_alt > 85000.0: 
            return np.zeros(3) # No atmosphere drag past the Karman boundary
            
        v_hat = velocity_vec / speed
        
        # 1. Dynamic Area & Cd scaling via Angle of Attack (AoA)
        dot_product = np.clip(np.dot(n_hat, v_hat), -1.0, 1.0)
        alpha = np.arccos(dot_product)
        
        Area_side = Area_nose * 15.0  
        Cd_side = 1.1  
        
        dynamic_Area = (Area_nose * np.cos(alpha)**2) + (Area_side * np.sin(alpha)**2)
        baseline_Cd = (Cd_nose * np.cos(alpha)**2) + (Cd_side * np.sin(alpha)**2)
        
        # 2. Dynamic Speed of Sound & Mach Transonic Wave Drag Scaling
        # Speed of sound roughly drops from 340 m/s at sea level to ~295 m/s in upper layers
        local_speed_of_sound = max(295.0, 340.29 - (0.004 * current_alt))
        mach = speed / local_speed_of_sound
        
        # Standard NASA transonic wave drag factor scaling curve approximation
        if 0.8 <= mach <= 1.2:
            mach_multiplier = 1.0 + 2.5 * (1.0 - np.cos(np.pi * (mach - 0.8) / 0.4))
        elif mach > 1.2:
            mach_multiplier = 1.0 + 1.8 / (mach ** 0.5)
        else:
            mach_multiplier = 1.0
            
        final_Cd = baseline_Cd * mach_multiplier
        
        rho = consts.rho_0 * np.exp(-current_alt / consts.Zfactor)
        magnitude = 0.5 * rho * (speed ** 2) * dynamic_Area * final_Cd
        
        return -magnitude * v_hat

    def calc_grav_force(self, current_displacement_vec: np.ndarray, inst_mass: float) -> np.ndarray:
        """Spherical Earth model mapping true central-pull gravitational vector fields."""
        # Calculate true center-to-center displacement vector
        earth_center_to_rocket = current_displacement_vec + np.array([0.0, 0.0, consts.RE])
        distance = np.linalg.norm(earth_center_to_rocket)
        
        if distance < 1e-5:
            return np.array([0.0, 0.0, -consts.g * inst_mass])
            
        magnitude = (consts.G * consts.ME * inst_mass) / (distance ** 2)
        radial_down_unit_vector = -earth_center_to_rocket / distance
        
        return magnitude * radial_down_unit_vector

class Displacement:
    def __init__(self, initial_displacement: np.ndarray = None):
        if initial_displacement is None:
            self.vector = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.vector = np.array(initial_displacement, dtype=float)
            
    def update_position(self, velocity_vec: np.ndarray, dt: float):
        self.vector += velocity_vec * dt
        return self.vector
        
    def get_altitude(self) -> float:
        return float(self.vector[2])
        
    def reset_ground_limit(self):
        self.vector[2] = max(0.0, self.vector[2])
