#%%
import pandas as pd
import numpy as np
from pathlib import Path

#%% =========================
# Supplementary Note 3:
# Sensitivity to substitution and technological adjustment
# =========================
file_path = Path("./digital_SUTs_IOTs/data_for_cal/Digital IO Tables 2000-2020 for China.xlsx")
out_dir = Path("./results/robustness_substitution")
out_dir.mkdir(parents=True, exist_ok=True)
years = [2000,2001,2002,2003,2004,2005,2006,2007,
         2008,2009,2010,2011,2012,2013,2014,2017,2018,2020]
N = 94
# table layout
row0 = 1
col0 = 1
rows = slice(row0, row0 + N)
cols = slice(col0, col0 + N)
col_C = col0 + N + 0
row_VA = row0 + N
row_X  = row0 + N + 1
group = np.array(["NDP"] * 40 + ["DCP"] * 40 + ["DP"] * 14)
SECTOR_NAMES = [
    "Agriculture", "Forestry", "Fishery", "Mining", "Food", "Textiles",
    "Paper", "Wood", "Petroleum", "Chemicals", "Pharma", "Rubber",
    "Nonmetal", "BasicMetals", "FabMetals", "Machinery", "Vehicles", "TransportEquip",
    "Electrical", "Instruments", "OtherMfg", "Electricity", "Water", "Construction",
    "WholesaleRetail", "WaterTrans", "AirTrans", "LandTrans", "Post", "Hospitality",
    "Finance", "RealEstate", "BusinessSvc", "R&D", "TechSvc", "HouseholdSvc",
    "Education", "Health", "Culture", "PublicAdmin",
    "D-Agriculture", "D-Forestry", "D-Fishery", "D-Mining", "D-Food", "D-Textiles",
    "D-Paper", "D-Wood", "D-Petroleum", "D-Chemicals", "D-Pharma", "D-Rubber",
    "D-Nonmetal", "D-BasicMetals", "D-FabMetals", "D-Machinery", "D-Vehicles", "D-TransportEquip",
    "D-Electrical", "D-Instruments", "D-OtherMfg", "D-Electricity", "D-Water", "D-Construction",
    "D-WholesaleRetail", "D-WaterTrans", "D-AirTrans", "D-LandTrans", "D-Post", "D-Hospitality",
    "D-Finance", "D-RealEstate", "D-BusinessSvc", "D-R&D", "D-TechSvc", "D-HouseholdSvc",
    "D-Education", "D-Health", "D-Culture", "D-PublicAdmin",
    "DP-Media", "DP-Infochem", "DP-IntEquip", "DP-Opto", "DP-ICT", "DP-Infra",
    "DP-Ecommerce", "DP-Telecom", "DP-ITservices", "DP-Fintech", "DP-BizOnline",
    "DP-DigiR&D", "DP-3Dprint", "DP-ICTrepair"
]
# =========================
# DP and non-DP masks
# =========================
dp_mask = np.zeros(N, dtype=bool)
dp_mask[80:94] = True
non_dp_mask = ~dp_mask

# substitution scenarios
theta_values = [0.0, 0.2, 0.5, 0.8]
yearly_rows = []
sector_rows = []

#%% =========================
# construct A with partial substitution
# =========================
def construct_A_with_substitution(A, dp_mask, theta):
    """
    Construct adjusted technical coefficients matrix under partial substitution.
    Step 1:
    Remove DP rows as upstream suppliers.
    Step 2:
    Reallocate theta share of removed DP inputs to non-DP suppliers,
    proportionally according to their original input shares.
    theta = 0: no substitution, identical to baseline no-DP supply scenario.
    theta > 0: partial replacement of removed DP inputs by non-DP inputs.
    """
    A_adj = A.copy()
    # total removed DP inputs for each purchasing sector j
    removed_dp_inputs = A[dp_mask, :].sum(axis=0)
    # original non-DP input coefficients for each purchasing sector j
    non_dp_inputs = A[non_dp_mask, :]
    non_dp_sum = non_dp_inputs.sum(axis=0)
    # remove DP suppliers
    A_adj[dp_mask, :] = 0.0
    # reallocate theta * removed DP inputs to non-DP suppliers
    for j in range(A.shape[1]):
        if removed_dp_inputs[j] == 0:
            continue
        if non_dp_sum[j] > 0:
            allocation_weights = A[non_dp_mask, j] / non_dp_sum[j]
            A_adj[non_dp_mask, j] += theta * removed_dp_inputs[j] * allocation_weights
        else:
            # If there are no non-DP suppliers in the original column,
            # no reallocation is performed.
            pass
    return A_adj

#%% =========================
# Main loop
# =========================
for yr in years:
    print(f"Processing year {yr}...")
    df = pd.read_excel(file_path, sheet_name=str(yr), header=None)
    Z = df.iloc[rows, cols].to_numpy(dtype=float)
    C = df.iloc[rows, col_C].to_numpy(dtype=float)
    v = df.iloc[row_VA, cols].to_numpy(dtype=float)
    x = df.iloc[row_X, cols].to_numpy(dtype=float)
    x_safe = np.where(x == 0, 1e-9, x)
    A = Z / x_safe
    I = np.eye(N)
    # baseline Leontief inverse
    L = np.linalg.inv(I - A)
    # value-added coefficients
    k = v / x_safe
    # baseline household-consumption-induced output and EVA
    x_c_base = L @ C
    TotalVA_base = float(np.dot(k, x_c_base))
    TotalVA_sector_base = k * x_c_base
    # final demand vector excluding DP
    C_noDP = C.copy()
    C_noDP[dp_mask] = 0.0

    for theta in theta_values:
        # adjusted A under substitution scenario
        A_sub = construct_A_with_substitution(A, dp_mask, theta)
        # check column sums and spectral radius
        col_sums = A_sub.sum(axis=0)
        max_col_sum = float(np.max(col_sums))
        eigvals = np.linalg.eigvals(A_sub)
        spectral_radius = float(np.max(np.abs(eigvals)))
        # Leontief inverse under substitution scenario
        L_sub = np.linalg.inv(I - A_sub)
        # -------------------------
        # Counterfactual 1
        # DP removed as suppliers, with partial substitution
        # -------------------------
        x_c_sub = L_sub @ C
        TotalVA_sub = float(np.dot(k, x_c_sub))
        DP_Support_theta = TotalVA_base - TotalVA_sub

        # -------------------------
        # Counterfactual 2:
        # DP removed as suppliers and from household final demand
        # with partial substitution in supply structure
        # -------------------------
        x_c_sub_noDPfd = L_sub @ C_noDP
        TotalVA_sub_noDPfd = float(np.dot(k, x_c_sub_noDPfd))
        DP_TotalEffect_theta = TotalVA_base - TotalVA_sub_noDPfd
        DP_FinalDemand_theta = TotalVA_sub - TotalVA_sub_noDPfd
        identity_gap = DP_TotalEffect_theta - (
            DP_Support_theta + DP_FinalDemand_theta)
        yearly_rows.append({
            "year": yr,
            "theta": theta,
            "TotalVA_base": TotalVA_base, #added value in household consumption under the baseline scenario
            "TotalVA_minus_DP_supply_with_substitution": TotalVA_sub,
            "TotalVA_minus_DP_supply_and_demand_with_substitution": TotalVA_sub_noDPfd,
            "DP_Support_theta": DP_Support_theta, # Supply chain support effect in substitution scenarios
            "DP_FinalDemand_theta": DP_FinalDemand_theta, # final-demand effect in substitution scenarios
            "DP_TotalEffect_theta": DP_TotalEffect_theta, # total effect in substitution scenarios
            "support_share_theta": (
                DP_Support_theta / DP_TotalEffect_theta
                if DP_TotalEffect_theta != 0 else np.nan),
            "identity_gap": identity_gap,
            "max_column_sum_A_sub": max_col_sum,
            "spectral_radius_A_sub": spectral_radius
        })
        
        # -------------------------
        # Sector-level decomposition
        # Identify which sectors are still the main transmission channels 
        # for digital investment affecting household consumption 
        # under different substitution capabilities.
        # -------------------------
        VA_sector_sub = k * x_c_sub
        VA_sector_sub_noDPfd = k * x_c_sub_noDPfd
        DP_Support_sector_theta = TotalVA_sector_base - VA_sector_sub
        DP_FinalDemand_sector_theta = VA_sector_sub - VA_sector_sub_noDPfd
        DP_TotalEffect_sector_theta = TotalVA_sector_base - VA_sector_sub_noDPfd
        for i in range(N):
            sector_rows.append({
                "year": yr,
                "theta": theta,
                "sector_index": i,
                "sector_name": SECTOR_NAMES[i],
                "group": group[i],
                "TotalVA_sector_base": float(TotalVA_sector_base[i]),
                "VA_sector_minus_DP_supply_with_substitution": float(VA_sector_sub[i]),
                "VA_sector_minus_DP_supply_and_demand_with_substitution": float(VA_sector_sub_noDPfd[i]),
                "DP_Support_sector_theta": float(DP_Support_sector_theta[i]),
                "DP_FinalDemand_sector_theta": float(DP_FinalDemand_sector_theta[i]),
                "DP_TotalEffect_sector_theta": float(DP_TotalEffect_sector_theta[i])
            })

# =========================
# Save results
# =========================
df_yearly = pd.DataFrame(yearly_rows).sort_values(["year", "theta"])
df_sector = pd.DataFrame(sector_rows).sort_values(["year", "theta", "sector_index"])
df_yearly.to_csv(out_dir / "robustness_substitution_yearly.csv", index=False)
df_sector.to_csv(out_dir / "robustness_substitution_sector.csv", index=False)
print("Saved:")
print(out_dir / "robustness_substitution_yearly.csv")
print(out_dir / "robustness_substitution_sector.csv")
# %%
