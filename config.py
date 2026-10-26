from dataclasses import dataclass

@dataclass
class EnergyConfig:
    interval_minutes: int = 60
    battery_capacity_kwh: float = 80.0
    battery_max_power_kw: float = 20.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc: float = 0.55
    min_soc: float = 0.10
    max_soc: float = 0.95
    desired_temperature_c: float = 24.0
    comfort_low_c: float = 22.0
    comfort_high_c: float = 26.0
    peak_threshold_kw: float = 45.0
    peak_penalty_per_kw: float = 0.18
    comfort_penalty_per_degree: float = 2.0
    battery_degradation_per_kwh: float = 0.035
    export_price_fraction: float = 0.35
    seed: int = 42
