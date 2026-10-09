import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- 1. SYNTHETIC DATASET GENERATOR ---
def generate_soc_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # Supported process nodes (nm)
    nodes = np.random.choice([3, 5, 7, 10, 14, 28], size=n_samples, p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.1])
    cores = np.random.randint(2, 33, size=n_samples)                 # 2 to 32 CPU cores
    freq_ghz = np.random.uniform(1.2, 4.2, size=n_samples)          # 1.2 to 4.2 GHz
    l3_cache_mb = np.random.choice([2, 4, 8, 16, 32, 64], size=n_samples)
    npu_tops = np.random.choice([0, 10, 20, 45, 80], size=n_samples) # Edge to AI TOPS

    # Physical scaling proxies:
    # Operating voltage scales down with node size
    vdd = 0.65 + 0.015 * nodes
    
    # Capacitance / density factor per gate decreases with node
    cap_factor = (nodes / 28.0) ** 0.8
    
    # 1. Area Estimation (mm²)
    # Logic area scales quadratically with node size; cache is SRAM dense
    core_area = (cores * 1.8) * (nodes / 14.0) ** 1.3
    cache_area = (l3_cache_mb * 0.45) * (nodes / 14.0) ** 1.1
    npu_area = (npu_tops * 0.35) * (nodes / 14.0) ** 1.2
    base_die_overhead = 8.0  # I/O, memory controllers, interconnects
    area_mm2 = core_area + cache_area + npu_area + base_die_overhead + np.random.normal(0, 1.5, n_samples)
    area_mm2 = np.clip(area_mm2, 8.0, 450.0)

    # 2. Power Estimation (Watts)
    # Dynamic Power: P_dyn ~ alpha * C * V^2 * f
    p_dyn = (cores * cap_factor * (vdd ** 2) * freq_ghz * 0.7) + (npu_tops * 0.08)
    # Leakage/Static Power: worse at advanced sub-7nm nodes due to quantum tunneling
    leakage_multiplier = np.where(nodes <= 5, 0.35, 0.15)
    p_leak = area_mm2 * leakage_multiplier * (vdd ** 1.2) * 0.05
    total_power_w = p_dyn + p_leak + np.random.normal(0, 0.5, n_samples)
    total_power_w = np.clip(total_power_w, 0.8, 180.0)

    # 3. Performance Metric (Normalized Synthetic Score)
    ipc_factor = 1.0 + (28.0 - nodes) * 0.015  # Modern nodes have microarchitectural advantages
    multi_core_efficiency = cores ** 0.85      # Amdahl's law scaling roll-off
    cpu_perf = multi_core_efficiency * freq_ghz * ipc_factor * 120
    ai_perf = npu_tops * 45
    perf_score = cpu_perf + ai_perf + np.random.normal(0, 50, n_samples)
    perf_score = np.clip(perf_score, 150.0, 9500.0)

    df = pd.DataFrame({
        "node_nm": nodes,
        "cores": cores,
        "freq_ghz": np.round(freq_ghz, 2),
        "l3_cache_mb": l3_cache_mb,
        "npu_tops": npu_tops,
        "power_w": np.round(total_power_w, 2),
        "area_mm2": np.round(area_mm2, 2),
        "perf_score": np.round(perf_score, 1)
    })
    return df


# --- 2. MULTI-OUTPUT REGRESSOR ---
class PPAEngine:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_cols = ["node_nm", "cores", "freq_ghz", "l3_cache_mb", "npu_tops"]
        self.target_cols = ["power_w", "area_mm2", "perf_score"]
        self.dataset = None

    def train(self):
        self.dataset = generate_soc_dataset()
        X = self.dataset[self.feature_cols]
        y = self.dataset[self.target_cols]
        self.model.fit(X, y)

    def predict(self, node_nm, cores, freq_ghz, l3_cache_mb, npu_tops):
        features = pd.DataFrame([[node_nm, cores, freq_ghz, l3_cache_mb, npu_tops]], columns=self.feature_cols)
        preds = self.model.predict(features)[0]
        return {
            "power_w": round(float(preds[0]), 2),
            "area_mm2": round(float(preds[1]), 2),
            "perf_score": round(float(preds[2]), 1)
        }

    # --- 3. PARETO OPTIMALITY SOLVER ---
    def find_pareto_designs(self, max_power=None, max_area=None, min_perf=None):
        """Filters dataset to non-dominated designs (minimize power, minimize area, maximize performance)."""
        df = self.dataset.copy()
        
        # Apply user constraints
        if max_power:
            df = df[df["power_w"] <= max_power]
        if max_area:
            df = df[df["area_mm2"] <= max_area]
        if min_perf:
            df = df[df["perf_score"] >= min_perf]
            
        if df.empty:
            return pd.DataFrame()

        # Vectorized Pareto filtration
        # Metrics to minimize: power, area. Metric to maximize: perf (invert to -perf)
        costs = np.column_stack((df["power_w"].values, df["area_mm2"].values, -df["perf_score"].values))
        
        is_pareto = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_pareto[i]:
                # Point is dominated if another point is <= in all costs and strictly < in at least one
                is_pareto[is_pareto] = ~(
                    np.all(costs[is_pareto] <= c, axis=1) & 
                    np.any(costs[is_pareto] < c, axis=1)
                )
                is_pareto[i] = True
                
        pareto_df = df.iloc[is_pareto].copy()
        return pareto_df.sort_values(by="perf_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Training PPA Engine...")
    engine = PPAEngine()
    engine.train()
    
    sample_spec = {"node_nm": 5, "cores": 8, "freq_ghz": 3.0, "l3_cache_mb": 16, "npu_tops": 20}
    prediction = engine.predict(**sample_spec)
    print("\nSample Design Prediction:", sample_spec)
    print("Predicted Metrics:", prediction)
    
    pareto_front = engine.find_pareto_designs(max_power=25.0, max_area=80.0)
    print(f"\nDiscovered {len(pareto_front)} Pareto-optimal configurations under 25W and 80mm²:")
    print(pareto_front.head(5))