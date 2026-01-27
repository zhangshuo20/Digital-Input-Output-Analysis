#%%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
file_path = Path("./digital_SUTs_IOTs/data_for_cal/Digital IO Tables 2000-2020 for China.xlsx")
out_dir = Path("./results")
out_dir.mkdir(exist_ok=True)

years = [2000,2001,2002,2003,2004,2005,2006,2007,
         2008,2009,2010,2011,2012,2013,2014,2017,2018,2020]

N = 94  # number of products/sectors in each year-sheet

# table layout (based on your Excel structure)
row0 = 1          # first sector row
col0 = 1          # first sector col
rows = slice(row0, row0 + N)
cols = slice(col0, col0 + N)

col_C = col0 + N + 0  # household consumption column
row_VA = row0 + N     # value added row
row_X  = row0 + N + 1 # total output row

group = np.array(["NDP"]*40 + ["DCP"]*40 + ["DP"]*14)
group[40:80] = "DCP"
group[80:94] = "DP"

SECTOR_NAMES = [
    # --- 1–40: Non-digitalised products (NDP) ---
    "Agriculture", "Forestry", "Fishery", "Mining", "Food", "Textiles",
     "Paper", "Wood", "Petroleum", "Chemicals", "Pharma", "Rubber",
    "Nonmetal", "BasicMetals", "FabMetals", "Machinery", "Vehicles", "TransportEquip",
    "Electrical", "Instruments", "OtherMfg", "Electricity", "Water", "Construction",
    "WholesaleRetail", "WaterTrans", "AirTrans", "LandTrans", "Post", "Hospitality",
    "Finance", "RealEstate", "BusinessSvc", "R&D", "TechSvc", "HouseholdSvc",
    "Education", "Health", "Culture", "PublicAdmin",

    # --- 41–80: Digitalised Conventional Products (DCP) ---
    "D-Agriculture", "D-Forestry", "D-Fishery", "D-Mining", "D-Food", "D-Textiles",
    "D-Paper", "D-Wood", "D-Petroleum", "D-Chemicals", "D-Pharma", "D-Rubber",
    "D-Nonmetal", "D-BasicMetals", "D-FabMetals", "D-Machinery", "D-Vehicles", "D-TransportEquip",
    "D-Electrical", "D-Instruments", "D-OtherMfg", "D-Electricity", "D-Water", "D-Construction",
    "D-WholesaleRetail", "D-WaterTrans", "D-AirTrans", "D-LandTrans", "D-Post", "D-Hospitality",
    "D-Finance", "D-RealEstate", "D-BusinessSvc", "D-R&D", "D-TechSvc", "D-HouseholdSvc",
    "D-Education", "D-Health", "D-Culture", "D-PublicAdmin",

    # --- 81–94: Digital Products (DP) ---
    "DP-Media", "DP-Infochem", "DP-IntEquip", "DP-Opto", "DP-ICT", "DP-Infra",
    "DP-Ecommerce", "DP-Telecom", "DP-ITservices", "DP-Fintech", "DP-BizOnline",
    "DP-DigiR&D", "DP-3Dprint", "DP-ICTrepair"
]

yearly_rows = []
sector_rows = []
store = {}  
DP_SLICE = slice(80, 94)

for yr in years:
    print(f"Processing year {yr}...")

    # ---- read sheet ----
    df = pd.read_excel(file_path, sheet_name=str(yr), header=None)

    # ---- extract matrices ----
    Z = df.iloc[rows, cols].to_numpy(dtype=float)        # (N,N)
    C = df.iloc[rows, col_C].to_numpy(dtype=float)       # (N,)
    v = df.iloc[row_VA, cols].to_numpy(dtype=float)      # (N,)
    x = df.iloc[row_X,  cols].to_numpy(dtype=float)      # (N,)

    # ---- A and Leontief inverse L ----
    x_safe = np.where(x == 0, 1e-9, x)
    A = Z / x_safe
    I = np.eye(N)
    L = np.linalg.inv(I - A)

    # ---- (1) direct digital share in C ----
    C_total = C.sum()
    C_digital = C[40:].sum()  # DCP+DP = indices 40..93
    direct_share = C_digital / C_total if C_total != 0 else np.nan

    # ---- (2) digital input intensity (per column / per buyer sector) ----
    digital_rows_mask = np.zeros(N, dtype=bool)
    digital_rows_mask[40:] = True  # DCP+DP rows are "digital inputs"
    digital_inputs = Z[digital_rows_mask, :].sum(axis=0)  # sum over rows -> by column
    total_inputs = Z.sum(axis=0)
    total_inputs_safe = np.where(total_inputs == 0, 1e-9, total_inputs)
    d_input = digital_inputs / total_inputs_safe
    avg_d_input = float(d_input.mean())

    # ---- (3) VA coefficients  ----
    k_total = v / x_safe

    # ---- (4) embodied VA in C: base case ----
    y_base = L @ C
    TotalVA = float(np.dot(k_total, y_base))

    # ---- (5) Counterfactual 1: remove DP rows from A (DP not an intermediate input supplier) ----
    A_noDP = A.copy()
    A_noDP[DP_SLICE, :] = 0.0     # DP rows = indices 80..93
    L_noDP = np.linalg.inv(I - A_noDP)
    y_noDP = L_noDP @ C
    TotalVA_noDP = float(np.dot(k_total, y_noDP))

    # ---- (6) "Support/embedding" effect of DP as intermediate inputs ----
    DP_Support = TotalVA - TotalVA_noDP

    # ---- (7) Counterfactual 2: also remove DP from final demand C ----
    C_noDP = C.copy()
    C_noDP[DP_SLICE] = 0.0
    y_noDP2 = L_noDP @ C_noDP
    TotalVA_noDP2 = float(np.dot(k_total, y_noDP2))

    # ---- (8) DP_TotalEffect and DP_FinalDemand ----
    DP_TotalEffect = TotalVA - TotalVA_noDP2            
    DP_FinalDemand = TotalVA_noDP - TotalVA_noDP2     
    # (optional but recommended) numerical identity check:
    # DP_TotalEffect = DP_Support + DP_FinalDemand
    identity_gap = DP_TotalEffect - (DP_Support + DP_FinalDemand)  

    # ---- store yearly metrics ----
    yearly_rows.append({
        "year": yr,
        "direct_digital_share_C": direct_share,
        "avg_digital_input_intensity": avg_d_input,

        "TotalVA_in_C": TotalVA,
        "TotalVA_in_C_minus_DP_supply": TotalVA_noDP,        
        "TotalVA_in_C_minus_DP_supply_and_demand": TotalVA_noDP2, 

        "DP_Support": DP_Support,                           
        "DP_FinalDemand": DP_FinalDemand,                   
        "DP_TotalEffect": DP_TotalEffect,                    
        "dp_identity_gap": identity_gap,                     #  (should be ~0)
    })

    # ============================
    # Sector-level outputs (split by baseline / CF1 / CF2)
    # ============================

    # baseline sector VA: TotalVA_{t,i} = k_i * y_i
    TotalVA_sector = k_total * y_base

    # CF1 sector VA: VA_{t,i}^{(-DP)} = k_i * y_i^{(-DP)}
    VA_sector_noDP = k_total * y_noDP                      

    # CF2 sector VA: VA_{t,i}^{(-DP2)} = k_i * y_i^{(-DP2)}
    VA_sector_noDP2 = k_total * y_noDP2                    

    # sectoral DP_Support_{t,i} = k_i * (y_i - y_i^{(-DP)})
    DP_Support_sector = TotalVA_sector - VA_sector_noDP    

    # sectoral DP_FinalDemand_{t,i} = k_i * (y_i^{(-DP)} - y_i^{(-DP2)})
    DP_FinalDemand_sector = VA_sector_noDP - VA_sector_noDP2  

    # sectoral DP_TotalEffect_{t,i} = TotalVA_{t,i} - VA_{t,i}^{(-DP2)}
    DP_TotalEffect_sector = TotalVA_sector - VA_sector_noDP2  

    for i in range(N):
        sector_rows.append({
            "year": yr,
            "sector_index": i,
            "sector_name": SECTOR_NAMES[i],
            "group": group[i],
            "digital_input_intensity": float(d_input[i]),

            # baseline
            "TotalVA_t_i": float(TotalVA_sector[i]),             
            # counterfactual sector VA
            "VA_t_i_minus_DP_supply": float(VA_sector_noDP[i]),  
            "VA_t_i_minus_DP_supply_and_demand": float(VA_sector_noDP2[i]),  

            # decomposed DP effects at sector level
            "DP_Support_t_i": float(DP_Support_sector[i]),       
            "DP_FinalDemand_t_i": float(DP_FinalDemand_sector[i]),  
            "DP_TotalEffect_t_i": float(DP_TotalEffect_sector[i]),  
        })

    # ---- store objects for later comparisons (optional) ----
    if yr in [2000, 2020]:
        store[yr] = {
            "k": k_total,
            "L": L,
            "L_noDP": L_noDP,
            "C": C,
            "C_noDP": C_noDP,
            "y_base": y_base,                
            "y_noDP": y_noDP,                
            "y_noDP2": y_noDP2,              
            "TotalVA": TotalVA,
            "TotalVA_noDP": TotalVA_noDP,
            "TotalVA_noDP2": TotalVA_noDP2,
        }

# =========================
# 3) SAVE YEARLY + SECTOR RESULTS
# =========================
df_yearly = pd.DataFrame(yearly_rows).sort_values("year")
df_sector = pd.DataFrame(sector_rows)

df_yearly.to_csv(out_dir / "digital_consumption_metrics_2000_2020.csv", index=False)
df_sector.to_csv(out_dir / "sector_level_digital_metrics_2000_2020.csv", index=False)

print("Saved:")
print(out_dir / "digital_consumption_metrics_2000_2020.csv")
print(out_dir / "sector_level_digital_metrics_2000_2020.csv")

# =========================
# 4) SDA (2000 -> 2020) in the same flat style
# =========================
def digital_va(k, L, C):
    return float(np.dot(k, L @ C))

k0, L0, C0 = store[2000]["k"], store[2000]["L"], store[2000]["C"]
k1, L1, C1 = store[2020]["k"], store[2020]["L"], store[2020]["C"]

D0 = digital_va(k0, L0, C0)
D1 = digital_va(k1, L1, C1)
Delta_total = D1 - D0

# ordering 1: k -> L -> C
term_k_1 = float(np.dot((k1 - k0), L0 @ C0))
term_L_1 = float(np.dot(k1, (L1 - L0) @ C0))
term_C_1 = float(np.dot(k1, L1 @ (C1 - C0)))

# ordering 2: C -> L -> k
term_C_2 = float(np.dot(k0, L0 @ (C1 - C0)))
term_L_2 = float(np.dot(k0, (L1 - L0) @ C1))
term_k_2 = float(np.dot((k1 - k0), L1 @ C1))

Delta_k = 0.5 * (term_k_1 + term_k_2)
Delta_L = 0.5 * (term_L_1 + term_L_2)
Delta_C = 0.5 * (term_C_1 + term_C_2)

df_sda = pd.DataFrame([{
    "year_start": 2000,
    "year_end": 2020,
    "Delta_total_DigitalVA_in_C": Delta_total,
    "Delta_k_effect": Delta_k,
    "Delta_L_effect": Delta_L,
    "Delta_C_effect": Delta_C
}])

df_sda.to_csv(out_dir / "sda_digital_va_2000_2020.csv", index=False)

print("Saved:")
print(out_dir / "sda_digital_va_2000_2020.csv")
# %%
