import pandas as pd
import numpy as np
import statsmodels.api as sm

WINSOR_P = 0.01

def winsorize_series(s, p=0.01):
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lo, hi)

def winsorize_df(df, cols, p=0.01):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize_series(out[c], p=p)
    return out

df = pd.read_parquet("outputs/panel_final.parquet")

rhs = ["PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S", "NOL"]
need = ["BTD"] + rhs

# 1) winsoriza insumos do ABTD no painel inteiro (igual ao pipeline)
ins_cols = ["BTD", "PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S"]
df2 = winsorize_df(df, ins_cols, p=WINSOR_P)

abtd_recalc = pd.Series(np.nan, index=df2.index, dtype="float64")

for year, g in df2.groupby("ANO"):
    g = g.dropna(subset=need).copy()
    if len(g) < 30:
        continue
    y = g["BTD"].astype(float)
    X = sm.add_constant(g[rhs].astype(float), has_constant="add")
    m = sm.OLS(y, X).fit()
    abtd_recalc.loc[g.index] = m.resid

# 2) winsoriza o resíduo (igual ao pipeline)
abtd_recalc = winsorize_series(abtd_recalc, p=WINSOR_P)

# 3) compara
for year in [2019, 2022]:
    sub = df2[df2["ANO"] == year]
    comp = pd.DataFrame({"ABTD_saved": sub["ABTD"], "ABTD_recalc": abtd_recalc.loc[sub.index]})
    diff = (comp["ABTD_saved"] - comp["ABTD_recalc"]).abs()
    print(year, "max|diff|:", diff.max(), "mean|diff|:", diff.mean())