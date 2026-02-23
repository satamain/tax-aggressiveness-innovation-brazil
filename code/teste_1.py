import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet("outputs/panel_final.parquet")

# ajuste a lista para bater com o que seu compute_abtd usa
rhs = ["PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S", "NOL"]
need = ["BTD"] + rhs

def recompute_abtd(df_year):
    g = df_year.dropna(subset=need).copy()
    y = g["BTD"].astype(float)
    X = g[rhs].astype(float)
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X).fit()
    out = pd.Series(np.nan, index=df_year.index)
    out.loc[g.index] = m.resid
    return out, m.rsquared, int(g.shape[0])

# escolha 2 anos para spot-check
for year in [2019, 2022]:
    sub = df[df["ANO"] == year].copy()
    abtd2, r2, n = recompute_abtd(sub)
    comp = pd.DataFrame({"ABTD_saved": sub["ABTD"], "ABTD_recalc": abtd2})
    diff = (comp["ABTD_saved"] - comp["ABTD_recalc"]).abs()
    print(year, "N usado:", n, "R2:", round(r2, 4), "max|diff|:", diff.max(), "mean|diff|:", diff.mean())


# 2a) média e desvio do ABTD por ano
print(df.groupby("ANO")["ABTD"].agg(["count", "mean", "std"]).round(6))

# 2b) ABTD "explicado" pelos regressors dentro do mesmo ano deve dar ~0
rhs = ["PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S", "NOL"]
for year, g in df.groupby("ANO"):
    g = g.dropna(subset=["ABTD"] + rhs)
    if len(g) < 30:
        continue
    y = g["ABTD"].astype(float)
    X = sm.add_constant(g[rhs].astype(float), has_constant="add")
    m = sm.OLS(y, X).fit()
    print(year, "R2(ABTD~rhs) =", round(m.rsquared, 6))


cols_check = ["BTD","IMPCOR","LAIR","ATIVO_L1","PPE_L1S","INTA_L1S","DREV_L1S","SIZE_L1S"]
miss = df[cols_check].isna().mean().sort_values(ascending=False)
print(miss)

# Onde ABTD está faltando, qual coluna está quebrando mais?
mask = df["ABTD"].isna()
print(df.loc[mask, cols_check].isna().mean().sort_values(ascending=False))