#%%
import pandas as pd
import numpy as np
from pathlib import Path

#%% =========================
# Supplementary Note 2:
# Robustness analysis on alternative specification of digital input intensity
# =========================
file_path = Path("./digital_SUTs_IOTs/data_for_cal/Digital IO Tables 2000-2020 for China.xlsx")
out_dir = Path("./results/robustness_digital_input_intensity")
out_dir.mkdir(parents=True, exist_ok=True)
years = [2000,2001,2002,2003,2004,2005,2006,2007,
         2008,2009,2010,2011,2012,2013,2014,2017,2018,2020]
N = 94
# table layout
row0 = 1
col0 = 1
rows = slice(row0, row0 + N)
cols = slice(col0, col0 + N)
col_C = col0 + N
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

#%% =========================
# Masks for baseline and robustness specifications
# =========================
mask_dcp_dp = np.zeros(N, dtype=bool)
mask_dcp_dp[40:94] = True      # baseline: DCP + DP
mask_dp_only = np.zeros(N, dtype=bool)

mask_dp_only[80:94] = True     # robustness: DP only
mask_dcp_only = np.zeros(N, dtype=bool)

mask_dcp_only[40:80] = True    # optional diagnostic: DCP only
# Used to observe how much of the benchmark results come from digital traditional products.

yearly_rows = []
sector_rows = []
for yr in years:
    print(f"Processing year {yr}...")
    df = pd.read_excel(file_path, sheet_name=str(yr), header=None)
    Z = df.iloc[rows, cols].to_numpy(dtype=float)
    C = df.iloc[rows, col_C].to_numpy(dtype=float)
    x = df.iloc[row_X, cols].to_numpy(dtype=float)
    # -------------------------
    # Total intermediate inputs by purchasing sector
    # -------------------------
    total_inputs = Z.sum(axis=0)
    total_inputs_safe = np.where(total_inputs == 0, 1e-9, total_inputs)
    # -------------------------
    # Baseline digital input intensity:
    # DCP + DP inputs / total intermediate inputs
    # -------------------------
    digital_inputs_baseline = Z[mask_dcp_dp, :].sum(axis=0)
    d_input_baseline = digital_inputs_baseline / total_inputs_safe
    # -------------------------
    # Alternative digital input intensity:
    # DP-only inputs / total intermediate inputs
    # -------------------------
    digital_inputs_dp_only = Z[mask_dp_only, :].sum(axis=0)
    d_input_dp_only = digital_inputs_dp_only / total_inputs_safe
    # -------------------------
    # Optional diagnostic:
    # DCP-only inputs / total intermediate inputs
    # -------------------------
    digital_inputs_dcp_only = Z[mask_dcp_only, :].sum(axis=0)
    d_input_dcp_only = digital_inputs_dcp_only / total_inputs_safe
    # -------------------------
    # Economy-wide average indicators
    # -------------------------
    avg_baseline = float(np.mean(d_input_baseline))
    avg_dp_only = float(np.mean(d_input_dp_only))
    avg_dcp_only = float(np.mean(d_input_dcp_only))
    # -------------------------
    # Weighted average indicators
    # Weight by purchasing sector's total intermediate inputs
    # -------------------------
    weight = total_inputs / total_inputs.sum() if total_inputs.sum() != 0 else np.ones(N) / N
    weighted_baseline = float(np.sum(d_input_baseline * weight))
    weighted_dp_only = float(np.sum(d_input_dp_only * weight))
    weighted_dcp_only = float(np.sum(d_input_dcp_only * weight))
    # -------------------------
    # Rank correlation between baseline and alternative sectoral indicators
    # This tests whether sector rankings are stable.
    # -------------------------
    rank_baseline = pd.Series(d_input_baseline).rank(ascending=False)
    rank_dp_only = pd.Series(d_input_dp_only).rank(ascending=False)
    rank_corr = float(rank_baseline.corr(rank_dp_only, method="spearman"))

    yearly_rows.append({
        "year": yr,
        # unweighted economy-wide averages
        "avg_digital_input_intensity_baseline_DCP_DP": avg_baseline,
        "avg_digital_input_intensity_DP_only": avg_dp_only,
        "avg_digital_input_intensity_DCP_only": avg_dcp_only,
        # weighted averages
        "weighted_digital_input_intensity_baseline_DCP_DP": weighted_baseline,
        "weighted_digital_input_intensity_DP_only": weighted_dp_only,
        "weighted_digital_input_intensity_DCP_only": weighted_dcp_only,
        # stability of sector ranking
        "spearman_rank_corr_baseline_vs_DP_only": rank_corr
    })

    for i in range(N):
        sector_rows.append({
            "year": yr,
            "sector_index": i,
            "sector_name": SECTOR_NAMES[i],
            "group": group[i],
            "total_intermediate_inputs": float(total_inputs[i]),
            # baseline and alternative indicators
            "digital_input_intensity_baseline_DCP_DP": float(d_input_baseline[i]),
            "digital_input_intensity_DP_only": float(d_input_dp_only[i]),
            "digital_input_intensity_DCP_only": float(d_input_dcp_only[i]),
            # absolute digital inputs
            "digital_inputs_baseline_DCP_DP": float(digital_inputs_baseline[i]),
            "digital_inputs_DP_only": float(digital_inputs_dp_only[i]),
            "digital_inputs_DCP_only": float(digital_inputs_dcp_only[i]),
        })

# =========================
# Save results
# =========================

df_yearly = pd.DataFrame(yearly_rows).sort_values("year")
df_sector = pd.DataFrame(sector_rows).sort_values(["year", "sector_index"])
df_yearly.to_csv(out_dir / "robustness_digital_input_intensity_yearly.csv", index=False)
df_sector.to_csv(out_dir / "robustness_digital_input_intensity_sector.csv", index=False)
print("Saved:")
print(out_dir / "robustness_digital_input_intensity_yearly.csv")
print(out_dir / "robustness_digital_input_intensity_sector.csv")
# %%
