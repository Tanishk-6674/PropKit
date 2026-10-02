import numpy as np

class consts:
    G = 6.674e-11
    ME = 5.97e+24
    RE = 6.371e6
    g = 9.81
    rho_0 = 1.225     
    Zfactor = 8500.0  

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
    def __init__(self, initial_velocity: float, obj_time: Time):
        self.run_time = obj_time.time_vec
        self.u_velocity = initial_velocity 
        
        # Component vectors for tracking telemetry losses explicitly
        self.tsio_delta_V = 0.0
        self.grav_delta_V = 0.0
        self.drag_delta_v_val = 0.0

    def calc_tsio_del_vel(self, Vj: float, inst_mass: float, stage_wet_mass: float):
        self.tsio_delta_V = Vj * np.log(stage_wet_mass / inst_mass)
        return self.tsio_delta_V

    def calc_grav_del_vel(self, total_grav_loss: float):
        self.grav_delta_V = total_grav_loss
        return self.grav_delta_V

    def calc_drag_del_vel(self, total_drag_loss: float):
        self.drag_delta_v_val = total_drag_loss
        return self.drag_delta_v_val

class Thrust:
    def calc_time_varying_inertial(self, obj_mass: Mass, current_time: float, stage_start_time: float, burn_out_duration: float, Vj: float):
        elapsed_stage_time = current_time - stage_start_time
        if elapsed_stage_time <= burn_out_duration:
            return obj_mass.m_flux * Vj  
        return 0.0

    def calc_drag_force(self, Cd: float, Area: float, velocity: float, current_alt: float):
        rho = consts.rho_0 * np.exp(-current_alt / consts.Zfactor)
        return 0.5 * rho * (velocity ** 2) * Area * Cd

    def calc_grav_force(self, current_alt: float, inst_mass: float):
        return (consts.G * consts.ME * inst_mass) / ((consts.RE + current_alt) ** 2)

class Altitude:
    def __init__(self, initial_altitude: float, apogee: float):
        self.x = initial_altitude
        self.apogee = apogee
