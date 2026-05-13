#%% ===============================
# Supplementary Note 1:
# Robustness – Narrow DP (ICT only)
# ===============================
import pandas as pd
import numpy as np
from pathlib import Path

# %%
file_path = Path("./digital_SUTs_IOTs/data_for_cal/Digital IO Tables 2000-2020 for China.xlsx")
out_dir = Path("./results/robustness_DP_ICT")
out_dir.mkdir(parents=True, exist_ok=True)

years = [2000,2001,2002,2003,2004,2005,2006,2007,
         2008,2009,2010,2011,2012,2013,2014,2017,2018,2020]

N = 94  
row0 = 1          # first sector row
col0 = 1          # first sector col
rows = slice(row0, row0 + N)
cols = slice(col0, col0 + N)

col_C = col0 + N + 0  # household consumption column
row_VA = row0 + N     # value added row
row_X  = row0 + N + 1 # total output row

# ===============================
# NEW: ICT-only DP definition
# ===============================
DP_ICT_INDEX = [85, 88, 89]  # ICT, Telecom, IT services

yearly_rows = []
for yr in years:
    print(f"[Robustness ICT] Processing {yr}")

    df = pd.read_excel(file_path, sheet_name=str(yr), header=None)
    Z = df.iloc[rows, cols].to_numpy(dtype=float)        # (N,N)
    C = df.iloc[rows, col_C].to_numpy(dtype=float)       # (N,)
    v = df.iloc[row_VA, cols].to_numpy(dtype=float)      # (N,)
    x = df.iloc[row_X,  cols].to_numpy(dtype=float)      # (N,)
    x_safe = np.where(x == 0, 1e-9, x)
    A = Z / x_safe
    I = np.eye(N)
    L = np.linalg.inv(I - A)

    # ===============================
    # baseline VA
    # ===============================
    k = v / x_safe
    y_base = L @ C
    TotalVA = float(np.dot(k, y_base))

    # ===============================
    # Counterfactual 1 (ICT-only)
    # ===============================
    A_noDP = A.copy()
    A_noDP[DP_ICT_INDEX, :] = 0.0   # ONLY ICT rows removed
    L_noDP = np.linalg.inv(I - A_noDP)
    y_noDP = L_noDP @ C
    TotalVA_noDP = float(np.dot(k, y_noDP))
    DP_Support = TotalVA - TotalVA_noDP

    # ===============================
    # Counterfactual 2
    # ===============================
    C_noDP = C.copy()
    C_noDP[DP_ICT_INDEX] = 0.0
    y_noDP2 = L_noDP @ C_noDP
    TotalVA_noDP2 = float(np.dot(k, y_noDP2))
    DP_TotalEffect = TotalVA - TotalVA_noDP2
    DP_FinalDemand = TotalVA_noDP - TotalVA_noDP2

    yearly_rows.append({
        "year": yr,
        "TotalVA": TotalVA,
        "TotalVA_noDP_ICT_supply": TotalVA_noDP,
        "TotalVA_noDP_ICT_supply_demand": TotalVA_noDP2,
        "DP_ICT_Support": DP_Support,
        "DP_ICT_FinalDemand": DP_FinalDemand,
        "DP_ICT_TotalEffect": DP_TotalEffect

    })

df_yearly = pd.DataFrame(yearly_rows).sort_values("year")
df_yearly.to_csv(out_dir / "robustness_ICT_only_results.csv", index=False)
print("Saved ICT robustness results")
# %%
