import numpy as np

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

    def set_motor_time(self, ignition: float, burn_out: float):
        self.ignition = ignition
        self.burn_out = burn_out
        
class Mass:
    def __init__(self, dry_mass: float, wet_mass: float, payload: float):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.payload_mass = payload
        self.alpha_fr = float((dry_mass - payload) / wet_mass) 
        self.beta_fr = float(payload / wet_mass)               
        self.gamma_fr = float((wet_mass - dry_mass) / wet_mass) 
        self.m_flux = 0.0

    def calc_m_dot(self, burn_time: float):
        if burn_time > 0:
            self.m_flux = (self.wet_mass - self.dry_mass) / burn_time
        else:
            self.m_flux = 0.0
        return self.m_flux

    def get_inst_mass(self, current_time: float, stage_start_time: float, burn_out_duration: float):
        elapsed_stage_time = current_time - stage_start_time
        if elapsed_stage_time > burn_out_duration:
            return self.dry_mass
        return self.wet_mass - (self.m_flux * elapsed_stage_time)
        
class Vel:
    def __init__(self, initial_velocity: np.ndarray, obj_time: Time):
        self.run_time = obj_time.time_vec
        self.u_velocity = np.array(initial_velocity, dtype=float) 
        
        self.tsio_delta_V = np.zeros(3)
        self.grav_delta_V = np.zeros(3)
        self.drag_delta_v_val = np.zeros(3)

    def calc_tsio_del_vel(self, Vj: float, inst_mass: float, stage_wet_mass: float, n_vector: np.ndarray):
        magnitude = Vj * np.log(stage_wet_mass / inst_mass)
        self.tsio_delta_V = magnitude * n_vector
        return self.tsio_delta_V

    def calc_grav_del_vel(self, accumulated_grav_loss_vec: np.ndarray):
        self.grav_delta_V = accumulated_grav_loss_vec
        return self.grav_delta_V

    def calc_drag_del_vel(self, accumulated_drag_loss_vec: np.ndarray):
        self.drag_delta_v_val = accumulated_drag_loss_vec
        return self.drag_delta_v_val

class Thrust:
    def calc_time_varying_inertial(self, obj_mass: Mass, current_time: float, stage_start_time: float, burn_out_duration: float, Vj: float, n_vector: np.ndarray):
        elapsed_stage_time = current_time - stage_start_time
        if elapsed_stage_time <= burn_out_duration:
            magnitude = obj_mass.m_flux * Vj  
            return magnitude * n_vector 
        return np.zeros(3)

    def calc_drag_force(self, Cd: float, Area: float, velocity_vec: np.ndarray, current_alt: float, n_vector: np.ndarray):
        rho = consts.rho_0 * np.exp(-current_alt / consts.Zfactor)
        speed = np.linalg.norm(velocity_vec)
        magnitude = 0.5 * rho * (speed ** 2) * Area * Cd
        return -magnitude * n_vector 

    def calc_grav_force(self, current_alt: float, inst_mass: float, k_vector: np.ndarray):
        magnitude = (consts.G * consts.ME * inst_mass) / ((consts.RE + current_alt) ** 2)
        return -magnitude * k_vector 

class Displacement:
    def __init__(self, initial_displacement: np.ndarray = None):
        # Default to origin [0,0,0] if no starting point vector is given
        if initial_displacement is None:
            self.vector = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.vector = np.array(initial_displacement, dtype=float)
            
    def update_position(self, velocity_vec: np.ndarray, dt: float):
        """Integrates velocity over time step to step displacement forward."""
        self.vector += velocity_vec * dt
        return self.vector
        
    def get_altitude(self) -> float:
        """Returns the Z component representing true altitude over ground."""
        return float(self.vector[2])
        
    def reset_ground_limit(self):
        """Enforces hard boundary so rocket doesn't phase below ground."""
        self.vector[2] = max(0.0, self.vector[2])
