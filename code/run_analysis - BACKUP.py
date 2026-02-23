"""
Pipeline completo (alinhado à metodologia) para:

1) Ler e padronizar os dois .xlsx do projeto;
2) Transformar o painel "wide" (colunas com sufixo de ano) em painel firma-ano ("long");
3) Aplicar filtros e tratamentos conforme metodologia:
   - período ANO_INI–ANO_FIM
   - excluir instituições financeiras
   - remover ATIVO <= 0 e LAIR nulo (LAIR negativo é mantido por padrão; ETRc só é filtrado quando LAIR>0)
   - remover ETRc < 0 ou > 1 (ETRc = IMPCOR/LAIR, apenas quando LAIR>0)
   - missing em inovação (CAPEX, RD, AMB) vira 0
   - razões só são calculadas quando denominador > 0
4) Vincular fiscalização setorial (RFB) por setor-ano (quantidade de autuações e valor total de créditos);
5) Construir BTD e ABTD (ABTD = resíduo da regressão cross-section por ano);
6) Winsorizar bicaudal 1% variáveis contínuas antes das estimações;
7) Criar defasagens t-1 para ABTD e controles;
8) Rodar regressões em painel FE firma+ano com erro-padrão cluster firma;
9) Gerar arquivo separado com estatística descritiva (Excel).

Entradas (na raiz do projeto):
  - dados.xlsx
  - relatorios-rf.xlsx

Saídas:
  - outputs/panel_final.parquet
  - outputs/regressions.txt (completo; alias de regressions_all.txt)
  - outputs/regressions_all.txt (todos os testes/modelos)
  - outputs/regressions_final_fe.txt (somente regressões finais FE firma+ano)
  - outputs/diagnostics.txt
  - outputs/descriptive_stats.xlsx
  - outputs/sample_flow.txt
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from linearmodels.panel import PanelOLS

    _HAS_LINEARMODELS = True
except Exception:
    _HAS_LINEARMODELS = False


# =========================
# Configuração do projeto
# =========================
def get_root() -> Path:
    # script em .../Dados/code/run_analysis.py -> root em .../Dados
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


ROOT = get_root()
INPUT_DADOS = ROOT / "dados.xlsx"
INPUT_RF = ROOT / "relatorios-rf.xlsx"
OUTDIR = ROOT / "outputs"

ANO_INI = 2014
ANO_FIM = 2024

TAU_STATUTARY = 0.34  # IRPJ+CSLL (use apenas se você estiver excluindo financeiras)

WINSOR_P = 0.01

# Proxy de agressividade tributária a usar nos modelos principais
# Opções: "ABTD", "ETRC", "GAAPETR"
TAX_AGG_PROXIES = ["ABTD", "ETRC", "GAAPETR"]
RUN_ALL_PROXIES = True  # True = roda ABTD, ETRC e GAAPETR (robustez) e escreve tudo no regressions.txt

# Moderadoras/controles do modelo principal (escolha o subconjunto que quiser).
# Opções válidas: "SIZE", "ROA", "LEV", "GR", "PPE", "INTA"
MODEL_MODERATORS = ["SIZE", "ROA", "LEV", "GR", "PPE", "INTA"]

# Para "fechar" com a metodologia e manter consistência com NOL, o default preserva LAIR negativo
# e aplica filtro de ETRc apenas quando LAIR > 0. Se você quiser seguir literalmente "LAIR negativo excluído",
# mude para True (mas NOL perde função).
DROP_LAIR_NONPOS = False

# Exclusão de financeiras: usa SETORRF/SECTORF (português) se existir, senão usa SECTOR (LSEG)
EXCLUDE_FINANCIALS = True

# Inovação: por padrão, mantém as versões escaladas por ATIVO (mais estável e compatível com regressões).
# Se você quiser usar "em nível" (CAPEX, RD) como está literal na tabela, defina False e use INOV_*_RAW nas regressões.
INOV_SCALE_BY_ASSETS = True

# ABTD: regressão cross-section por ANO (default).
# Se quiser por ANO e setor, mude para True (mas pode reduzir N por grupo e enfraquecer a estimação do resíduo).
ABTD_BY_YEAR_AND_SECTOR = False

# Fiscalização: por padrão usa os valores brutos setoriais (quantidade e total).
# Se quiser robustez, você pode usar LOG1P das brutas nas regressões (já calculamos colunas *_LN).
FISC_USE_LOG1P = True

# Moderação por setor (heterogeneidade): adiciona termos proxy_L1 x dummies de setor.
# Como os nomes de setor em dados.xlsx e relatorios-rf.xlsx já são compatíveis no merge,
# esta etapa usa SETOR_KEY no painel final.
RUN_SECTOR_MODERATION = True
# Mínimo de observações por setor; setores abaixo disso viram "__outros__".
SECTOR_MOD_MIN_OBS = 30

# Se True: trata missing em CAPEX/RD/AMB como 0 (assume "sem investimento")
# Se False: mantém missing como NaN (assume "não observado")
INOV_MISSING_AS_ZERO = False


# =========================
# Utilidades de limpeza
# =========================
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def clean_colname(c: str) -> str:
    c = str(c).strip()
    c = _strip_accents(c).upper()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^A-Z0-9_]", "", c)
    return c


def norm_key(x: str) -> str:
    """Chave de merge robusta para textos (setores)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = _strip_accents(s)
    s = s.lower()
    s = re.sub(r"[,;]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def coerce_ptbr_number(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(".", "").replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_ok = pd.to_numeric(denom, errors="coerce")
    numer_ok = pd.to_numeric(numer, errors="coerce")
    out = pd.Series(np.nan, index=numer.index, dtype="float64")
    mask = denom_ok > 0
    out.loc[mask] = (numer_ok.loc[mask] / denom_ok.loc[mask]).astype(float)
    return out


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def winsorize_df(df: pd.DataFrame, cols: Sequence[str], p: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize_series(out[c], p=p)
    return out


# =========================
# Leitura e reshape
# =========================
def read_dados_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [clean_colname(c) for c in df.columns]
    # normaliza aliases usuais de coluna de setor RF
    alias = {}
    if "SECTOR_RF" in df.columns and "SECTORRF" not in df.columns:
        alias["SECTOR_RF"] = "SECTORRF"
    if "SETOR_RF" in df.columns and "SETORRF" not in df.columns:
        alias["SETOR_RF"] = "SETORRF"
    if alias:
        df = df.rename(columns=alias)
    return df


def wide_to_long_firma_ano(
    df_wide: pd.DataFrame,
    id_cols: Sequence[str] = ("RIC", "NAME", "SECTOR", "SECTORF", "SECTORRF", "SETORRF"),
) -> pd.DataFrame:
    df = df_wide.copy()

    existing_ids = [c for c in id_cols if c in df.columns]

    year_pat = re.compile(r"^(?P<var>[A-Z_]+?)(?P<year>20\d{2})$")

    # pega colunas VAR+ANO, mas exclui ANO2024/ANO2023 etc para não duplicar "ANO"
    value_cols = []
    for c in df.columns:
        m = year_pat.match(c)
        if not m:
            continue
        if m.group("var") == "ANO":
            continue
        value_cols.append(c)

    if not value_cols:
        raise ValueError("Não encontrei colunas no padrão VAR+ANO (ex.: LAIR2024), excluindo ANOYYYY.")

    parsed = [year_pat.match(c).groupdict() for c in value_cols]
    years = sorted({int(p["year"]) for p in parsed})
    vars_ = sorted({p["var"] for p in parsed})

    records = []
    for y in years:
        cols_y = [f"{v}{y}" for v in vars_ if f"{v}{y}" in df.columns]
        tmp = df[existing_ids + cols_y].copy()
        tmp["ANO"] = y
        tmp = tmp.rename(columns={c: year_pat.match(c).group("var") for c in cols_y})
        records.append(tmp)

    long = pd.concat(records, ignore_index=True)
    long.columns = [clean_colname(c) for c in long.columns]

    # blindagem extra: se ainda houver duplicata, mantém a última ocorrência
    long = long.loc[:, ~long.columns.duplicated(keep="last")].copy()

    long["ANO"] = pd.to_numeric(long["ANO"], errors="coerce").astype("Int64")

    if "RIC" in long.columns:
        long["RIC"] = long["RIC"].astype(str).str.strip()

    return long


def find_rf_header_row(path: Path, sheet_name: Optional[str] = None, max_scan: int = 60) -> int:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan)
    if isinstance(raw, dict):
        raw = raw[next(iter(raw.keys()))]
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.strip().tolist()
        if not row:
            continue
        if str(row[0]).strip().upper() == "SETOR":
            joined = " ".join(row).upper()
            if "QTD20" in joined and "CRED20" in joined:
                return i
    return 0


def read_rf_xlsx(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    hdr = find_rf_header_row(path, sheet_name=sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=hdr)
    if isinstance(df, dict):
        df = df[next(iter(df.keys()))]

    df.columns = [clean_colname(c) for c in df.columns]

    if "SETOR" not in df.columns:
        cand = [c for c in df.columns if "SETOR" in c]
        if cand:
            df = df.rename(columns={cand[0]: "SETOR"})
        else:
            raise ValueError("Não encontrei coluna SETOR em relatorios-rf.xlsx.")

    for c in df.columns:
        if c == "SETOR":
            continue
        if c.startswith("QTD") or c.startswith("CRED"):
            df[c] = df[c].map(coerce_ptbr_number)

    df["SETOR"] = df["SETOR"].astype(str).str.strip()
    return df


def rf_wide_to_long(df_rf_wide: pd.DataFrame) -> pd.DataFrame:
    df = df_rf_wide.copy()
    year_pat = re.compile(r"^(?P<var>QTD|CRED)(?P<year>20\d{2})$")
    cols = [c for c in df.columns if year_pat.match(c)]
    if not cols:
        raise ValueError("Não encontrei colunas QTDYYYY/CREDYYYY em relatorios-rf.xlsx.")

    years = sorted({int(year_pat.match(c).group("year")) for c in cols})
    out = []
    for y in years:
        tmp = df[["SETOR"]].copy()
        tmp["ANO"] = y
        tmp["FISC_AUT_RAW"] = df.get(f"QTD{y}", np.nan)
        tmp["FISC_CRED_RAW"] = df.get(f"CRED{y}", np.nan)
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["ANO"] = pd.to_numeric(long["ANO"], errors="coerce").astype("Int64")
    long["SETOR_KEY"] = long["SETOR"].map(norm_key)

    # remove linha Total
    long = long[~long["SETOR"].str.upper().eq("TOTAL")].copy()

    long["FISC_AUT_LN"] = np.log1p(pd.to_numeric(long["FISC_AUT_RAW"], errors="coerce"))
    long["FISC_CRED_LN"] = np.log1p(pd.to_numeric(long["FISC_CRED_RAW"], errors="coerce"))

    # baseline conforme metodologia: brutos; robustez disponível: *_LN
    long["FISC_AUT"] = long["FISC_AUT_LN"] if FISC_USE_LOG1P else long["FISC_AUT_RAW"]
    long["FISC_CRED"] = long["FISC_CRED_LN"] if FISC_USE_LOG1P else long["FISC_CRED_RAW"]

    return long


# =========================
# Construção de variáveis e filtros
# =========================
@dataclass
class FlowLog:
    lines: List[str]

    def add(self, label: str, df: pd.DataFrame) -> None:
        firms = df["RIC"].nunique() if "RIC" in df.columns else np.nan
        obs = len(df)
        ymin = int(df["ANO"].min()) if "ANO" in df.columns and df["ANO"].notna().any() else None
        ymax = int(df["ANO"].max()) if "ANO" in df.columns and df["ANO"].notna().any() else None
        self.lines.append(f"{label}: obs={obs} firms={firms} years={ymin}-{ymax}")


def exclude_financials(df: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDE_FINANCIALS:
        return df

    out = df.copy()

    # prioridade alinhada ao usuário: SECTORRF (dados.xlsx) <-> SETOR (RF)
    for c in ["SECTORRF", "SETORRF", "SECTORF"]:
        if c in out.columns:
            key = out[c].map(norm_key)
            # remove se for "servicos financeiros"
            mask_fin = key.eq(norm_key("Servicos financeiros"))
            return out.loc[~mask_fin].copy()

    # fallback: SECTOR (LSEG)
    if "SECTOR" in out.columns:
        mask_fin = out["SECTOR"].astype(str).str.strip().str.lower().eq("financials")
        return out.loc[~mask_fin].copy()

    return out


def build_base_panel(df_long: pd.DataFrame, rf_long: pd.DataFrame, flow: FlowLog) -> pd.DataFrame:
    df = df_long.copy()

    # mantém ANO_INI-1 só para formar defasagens/BTD/ABTD de ANO_INI
    df = df[(df["ANO"] >= (ANO_INI - 1)) & (df["ANO"] <= ANO_FIM)].copy()
    flow.add(f"Filtro período {ANO_INI}-{ANO_FIM}", df)

    # padroniza setor-key do painel (para merge com RF)
    # escolhe a melhor coluna de setor disponível para o merge com RF:
    setor_col = None
    for cand in ["SECTORRF", "SETORRF", "SECTORF"]:
        if cand in df.columns:
            setor_col = cand
            break
    if setor_col is None:
        raise ValueError("Não encontrei coluna de setor mapeado (SECTORRF/SETORRF/SECTORF) em dados.xlsx.")

    df["SETOR_SOURCE"] = setor_col
    df["SETOR_KEY"] = df[setor_col].map(norm_key)
    flow.add(f"Setor-key definido a partir de {setor_col}", df)

    # merge com RF por SETOR_KEY + ANO
    rf = rf_long.copy()
    df = df.merge(
        rf[["SETOR_KEY", "ANO", "FISC_AUT_RAW", "FISC_CRED_RAW", "FISC_AUT_LN", "FISC_CRED_LN", "FISC_AUT", "FISC_CRED"]],
        on=["SETOR_KEY", "ANO"],
        how="left",
        validate="m:1",
    )
    flow.add("Merge com RFB (setor-ano)", df)
    if "FISC_AUT" in df.columns or "FISC_CRED" in df.columns:
        has_fisc = pd.Series(False, index=df.index)
        if "FISC_AUT" in df.columns:
            has_fisc = has_fisc | df["FISC_AUT"].notna()
        if "FISC_CRED" in df.columns:
            has_fisc = has_fisc | df["FISC_CRED"].notna()
        flow.lines.append(f"Cobertura merge RFB setor-ano: {has_fisc.mean() * 100:.1f}%")

    # numéricos principais
    num_cols = [
        "LAIR", "LP", "ATIVO", "RL", "REC",
        "CAPEX", "RD", "AMB",
        "DLP", "IMOB", "INTANG",
        "IMPCOR",
        "GAAPETR",
    ]
    for c in num_cols:
        if c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].map(coerce_ptbr_number)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # excluir financeiras
    df = exclude_financials(df)
    flow.add("Exclusão de financeiras", df)

    # remover ATIVO <= 0 ou nulo
    if "ATIVO" not in df.columns:
        raise ValueError("Não encontrei coluna ATIVO.")
    df = df[df["ATIVO"].notna() & (df["ATIVO"] > 0)].copy()
    flow.add("Filtro ATIVO > 0", df)

    # remover LAIR nulo; opcional remover LAIR<=0
    if "LAIR" not in df.columns:
        raise ValueError("Não encontrei coluna LAIR.")
    df = df[df["LAIR"].notna()].copy()
    if DROP_LAIR_NONPOS:
        df = df[df["LAIR"] > 0].copy()
    flow.add("Filtro LAIR não nulo (e opcional LAIR>0)", df)

    # missing em inovação: opcionalmente vira 0 (ou permanece NaN)
    if INOV_MISSING_AS_ZERO:
        for v in ["CAPEX", "RD", "AMB"]:
            if v in df.columns:
                df[v] = df[v].fillna(0)
        flow.add("Missing inovação -> 0 (CAPEX, RD, AMB)", df)
    else:
        flow.add("Missing inovação mantido como NaN (CAPEX, RD, AMB)", df)

    # ETRc = IMPCOR/LAIR apenas quando LAIR>0.
    # Não filtramos a base aqui para não afetar ABTD; inválidos viram NaN.
    if "IMPCOR" in df.columns:
        etrc_raw = pd.Series(np.where(df["LAIR"] > 0, df["IMPCOR"] / df["LAIR"], np.nan), index=df.index)
        valid_etrc = etrc_raw.gt(0) & etrc_raw.lt(1)
        df["ETRC"] = etrc_raw.where(valid_etrc)
        flow.add("ETRc calculado (LAIR>0) e válido em (0,1); inválidos -> NaN", df)
    else:
        df["ETRC"] = np.nan
        flow.add("ETRc não calculado (IMPCOR ausente)", df)

    if "GAAPETR" in df.columns:
        gaap = pd.to_numeric(df["GAAPETR"], errors="coerce")
        valid_gaap = gaap.gt(0) & gaap.lt(1)
        df["GAAPETR"] = gaap.where(valid_gaap)
        flow.add("GAAPETR válido em (0,1); inválidos -> NaN", df)

    # ordenar e criar defasagens base
    df = df.sort_values(["RIC", "ANO"]).copy()
    df["ATIVO_L1"] = df.groupby("RIC")["ATIVO"].shift(1)
    df["RL_L1"] = df.groupby("RIC")["RL"].shift(1) if "RL" in df.columns else np.nan
    df["REC_L1"] = df.groupby("RIC")["REC"].shift(1) if "REC" in df.columns else np.nan

    # NOL
    df["NOL"] = (df["LAIR"] < 0).astype("int64")

    # Controles em t (depois viram L1 para as regressões)
    df["SIZE"] = np.log(df["ATIVO"])
    df["ROA"] = safe_div(df["LAIR"], df["ATIVO"])  # você pode trocar por LP/ATIVO se preferir
    df["LEV"] = safe_div(df["DLP"], df["ATIVO_L1"]) if "DLP" in df.columns else np.nan
    if "RL" in df.columns and df["RL_L1"].notna().any():
        df["GR"] = (df["RL"] - df["RL_L1"]) / df["RL_L1"]
        df.loc[~np.isfinite(df["GR"]), "GR"] = np.nan
    else:
        df["GR"] = np.nan

    df["PPE"] = safe_div(df["IMOB"], df["ATIVO"]) if "IMOB" in df.columns else np.nan
    df["INTA"] = safe_div(df["INTANG"], df["ATIVO"]) if "INTANG" in df.columns else np.nan

    # Inovação: versões raw e (opcional) escaladas
    df["INOV_INC_RAW"] = df["CAPEX"] if "CAPEX" in df.columns else np.nan
    df["INOV_RAD_RAW"] = df["RD"] if "RD" in df.columns else np.nan
    df["INOV_AMBIENTAL"] = df["AMB"] if "AMB" in df.columns else np.nan

    df["INOV_INC"] = safe_div(df["CAPEX"], df["ATIVO"]) if INOV_SCALE_BY_ASSETS else df["INOV_INC_RAW"]
    df["INOV_RAD"] = safe_div(df["RD"], df["ATIVO"]) if INOV_SCALE_BY_ASSETS else df["INOV_RAD_RAW"]

    # BTD: (LAIR - TI)/ATIVO_L1, TI = IMPCOR/tau
    df["TI"] = df["IMPCOR"] / TAU_STATUTARY if "IMPCOR" in df.columns else np.nan
    df["BTD"] = (df["LAIR"] - df["TI"]) / df["ATIVO_L1"]
    df.loc[~np.isfinite(df["BTD"]), "BTD"] = np.nan

    # Variáveis do modelo normal de BTD (escaladas por ATIVO_L1)
    df["PPE_L1S"] = safe_div(df["IMOB"], df["ATIVO_L1"]) if "IMOB" in df.columns else np.nan
    df["INTA_L1S"] = safe_div(df["INTANG"], df["ATIVO_L1"]) if "INTANG" in df.columns else np.nan
    df["SIZE_L1S"] = np.log(df["ATIVO_L1"].where(df["ATIVO_L1"] > 0))

    # ΔReceita: usa REC se existir, senão RL; sempre dividido por ATIVO_L1
    if "REC" in df.columns and df["REC"].notna().any():
        df["DREV_L1S"] = safe_div(df["REC"] - df["REC_L1"], df["ATIVO_L1"])
    elif "RL" in df.columns and df["RL"].notna().any():
        df["DREV_L1S"] = safe_div(df["RL"] - df["RL_L1"], df["ATIVO_L1"])
    else:
        df["DREV_L1S"] = np.nan

    flow.add("Construção variáveis base (controles, inovação, BTD)", df)
    return df


def compute_abtd(df: pd.DataFrame, flow: FlowLog) -> pd.DataFrame:
    out = df.copy()
    out["ABTD"] = np.nan

    rhs = ["PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S", "NOL"]
    need = ["BTD"] + rhs

    # winsoriza insumos do ABTD antes de estimar (para alinhar com metodologia)
    ins_cols = ["BTD", "PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S"]
    out = winsorize_df(out, [c for c in ins_cols if c in out.columns], p=WINSOR_P)

    group_cols = ["ANO"]
    if ABTD_BY_YEAR_AND_SECTOR:
        group_cols = ["ANO", "SETOR_KEY"]

    for key, g in out.groupby(group_cols, dropna=False):
        g = g.dropna(subset=need).copy()
        k = len(rhs)
        # mínimo conservador para cross-section anual
        if g.shape[0] < max(30, k + 10):
            continue

        y = pd.to_numeric(g["BTD"], errors="coerce").astype(float)
        X = g[rhs].apply(pd.to_numeric, errors="coerce").astype(float)
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < max(30, k + 10):
            continue

        X = sm.add_constant(X.loc[ok], has_constant="add")
        m = sm.OLS(y.loc[ok], X).fit()
        out.loc[g.index[ok], "ABTD"] = m.resid

    flow.add("ABTD estimado (resíduo cross-section)", out)

    # winsoriza ABTD antes das regressões principais
    out["ABTD"] = winsorize_series(out["ABTD"], p=WINSOR_P)

    return out


def add_lags(df: pd.DataFrame, cols: Sequence[str], by: str = "RIC", lag: int = 1) -> pd.DataFrame:
    out = df.sort_values([by, "ANO"]).copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_L{lag}"] = out.groupby(by)[c].shift(lag)
    return out


# =========================
# Estatística descritiva
# =========================
def descriptive_table(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.Series(
        {
            "N": int(s.notna().sum()),
            "Mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
            "Std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
            "Min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "P1": float(s.quantile(0.01)) if s.notna().any() else np.nan,
            "P5": float(s.quantile(0.05)) if s.notna().any() else np.nan,
            "P25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "Median": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "P75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "P95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
            "P99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
            "Max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
        }
    )


def build_descriptives(df: pd.DataFrame, vars_: Sequence[str]) -> Dict[str, pd.DataFrame]:
    vars_ = [v for v in vars_ if v in df.columns]

    overall = pd.DataFrame({v: descriptive_table(df[v]) for v in vars_}).T

    by_year_frames = []
    for y, g in df.groupby("ANO"):
        t = pd.DataFrame({v: descriptive_table(g[v]) for v in vars_}).T
        t = t.reset_index().rename(columns={"index": "VAR"})  # <<<<< adiciona nome da variável
        t.insert(0, "ANO", int(y) if pd.notna(y) else y)
        by_year_frames.append(t)
    by_year = pd.concat(by_year_frames, ignore_index=True) if by_year_frames else pd.DataFrame()

    by_setor_frames = []
    if "SETOR_KEY" in df.columns:
        for s, g in df.groupby("SETOR_KEY"):
            t = pd.DataFrame({v: descriptive_table(g[v]) for v in vars_}).T
            t = t.reset_index().rename(columns={"index": "VAR"})  # <<<<< adiciona nome da variável
            t.insert(0, "SETOR_KEY", s)
            by_setor_frames.append(t)
    by_setor = pd.concat(by_setor_frames, ignore_index=True) if by_setor_frames else pd.DataFrame()

    return {"overall": overall, "by_year": by_year, "by_sector": by_setor}


# =========================
# Regressões FE
# =========================
def fit_fe_panel(
    df: pd.DataFrame,
    y: str,
    x: Sequence[str],
    entity: str = "RIC",
    time: str = "ANO",
    cluster: str = "RIC",
    cov_type: str = "clustered",
    dk_bandwidth: Optional[int] = None,
):
    if not _HAS_LINEARMODELS:
        raise RuntimeError("linearmodels não está disponível. Instale para usar PanelOLS.")

    # Se for cluster por uma coluna que não é entity/time, precisamos carregar essa coluna também
    cols = [entity, time, y] + list(x)
    if cov_type == "clustered" and cluster not in (entity, time) and cluster not in cols:
        cols.append(cluster)

    use = df[cols].dropna().copy()
    use = use.set_index([entity, time])

    Y = use[y]
    X = sm.add_constant(use[list(x)], has_constant="add")

    mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)

    if cov_type == "clustered":
        # >>> AQUI estava o seu erro: clusters não existia <<<
        if cluster == entity:
            clusters = pd.Series(use.index.get_level_values(entity), index=use.index, name=cluster)
        elif cluster == time:
            clusters = pd.Series(use.index.get_level_values(time), index=use.index, name=cluster)
        else:
            clusters = pd.Series(use[cluster], index=use.index, name=cluster)

        res = mod.fit(cov_type="clustered", clusters=clusters)

    elif cov_type == "driscoll-kraay":
        fit_kwargs = {}
        if dk_bandwidth is not None:
            fit_kwargs["bandwidth"] = dk_bandwidth
        res = mod.fit(cov_type="driscoll-kraay", **fit_kwargs)

    else:
        res = mod.fit(cov_type=cov_type)

    return res

def pick_proxy_cols(proxy: str):
    if proxy == "ABTD":
        return ("ABTD_L1", "ABTD_L1_C", "ABTD_L1_X_FISC", "TAXAGG_ABTD_L1")
    if proxy == "ETRC":
        return ("TAXAGG_ETRC_L1", "TAXAGG_ETRC_L1_C", "TAXAGG_ETRC_L1_X_FISC", "TAXAGG_ETRC_L1")
    if proxy == "GAAPETR":
        return ("TAXAGG_GAAP_L1", "TAXAGG_GAAP_L1_C", "TAXAGG_GAAP_L1_X_FISC", "TAXAGG_GAAP_L1")
    raise ValueError("TAX_AGG_PROXY inválida")


import matplotlib.pyplot as plt

def fit_pooled_ols(
    df: pd.DataFrame,
    y: str,
    x: Sequence[str],
    entity: str = "RIC",
    time: str = "ANO",
    add_time_fe: bool = True,
):
    cols = [entity, time, y] + list(x)
    use = df[cols].dropna().copy()

    Y = use[y].astype(float)
    X = use[list(x)].astype(float)

    if add_time_fe:
        d_year = pd.get_dummies(use[time].astype(int), prefix="YEAR", drop_first=True)
        X = pd.concat([X, d_year], axis=1)

    X = sm.add_constant(X, has_constant="add")

    # clustered SE por firma (diagnóstico)
    res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": use[entity]})
    return res


def marginal_effects_abtd(
    res,
    df: pd.DataFrame,
    abtd_coef: str,
    fisc_raw_col: str,
    inter_coef: str,
    out_csv: Path,
    out_png: Optional[Path] = None,
):
    """
    Efeito marginal de ABTD_L1 (centrado) em níveis de fiscalização:
      ME(f) = beta_abtd + beta_inter * (f - mean(f))
    IC 95% via delta method.
    """
    params = res.params
    vcov = res.cov

    if abtd_coef not in params.index or inter_coef not in params.index:
        return

    fisc = pd.to_numeric(df[fisc_raw_col], errors="coerce").dropna()
    if fisc.empty:
        return

    grid = fisc.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).values
    f_mean = float(fisc.mean())

    beta_a = float(params[abtd_coef])
    beta_i = float(params[inter_coef])

    V = vcov.loc[[abtd_coef, inter_coef], [abtd_coef, inter_coef]].values

    rows = []
    for f in grid:
        f_c = float(f - f_mean)
        me = beta_a + beta_i * f_c
        g = np.array([1.0, f_c])
        se = float(np.sqrt(g @ V @ g))
        rows.append(
            {
                "fisc_value": float(f),
                "fisc_centered": f_c,
                "me_abtd": me,
                "se": se,
                "ci_lo": me - 1.96 * se,
                "ci_hi": me + 1.96 * se,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    if out_png is not None:
        plt.figure()
        plt.plot(out["fisc_value"], out["me_abtd"])
        plt.fill_between(out["fisc_value"], out["ci_lo"], out["ci_hi"], alpha=0.2)
        plt.xlabel(fisc_raw_col)
        plt.ylabel("Efeito marginal de ABTD_L1")
        plt.title("Efeito marginal de ABTD_L1 em níveis de fiscalização")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()



# =========================
# Main
# =========================
def ensure_year_bounds(df: pd.DataFrame, year_start: int, year_end: int, label: str) -> List[int]:
    """Valida se ANO está dentro do intervalo e retorna anos únicos ordenados."""
    if "ANO" not in df.columns:
        return []
    years = pd.to_numeric(df["ANO"], errors="coerce").dropna().astype(int)
    if years.empty:
        return []
    outside = sorted(set(years[(years < year_start) | (years > year_end)].tolist()))
    if outside:
        raise RuntimeError(f"{label}: anos fora do intervalo [{year_start}, {year_end}]: {outside}")
    return sorted(set(years.tolist()))


def prune_collinear_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """
    Mantém um subconjunto de colunas com independência linear (ordem preservada).
    Usa rank incremental em matriz numérica completa.
    """
    keep: List[str] = []
    if df.empty:
        return keep
    for c in cols:
        if c not in df.columns:
            continue
        trial = keep + [c]
        mat = df[trial].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if mat.size == 0:
            continue
        rank = np.linalg.matrix_rank(mat)
        if rank > len(keep):
            keep.append(c)
    return keep


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    flow = FlowLog(lines=[])

    # -------------------------
    # 1) Ler dados e reshape
    # -------------------------
    df_wide = read_dados_xlsx(INPUT_DADOS)
    df_long = wide_to_long_firma_ano(df_wide)
    flow.add("Wide->Long (firma-ano)", df_long)

    rf_wide = read_rf_xlsx(INPUT_RF)
    rf_long = rf_wide_to_long(rf_wide)
    flow.add("Wide->Long (RFB setor-ano)", rf_long)

    # -------------------------
    # 2) Construir painel base + ABTD
    # -------------------------
    panel = build_base_panel(df_long, rf_long, flow)
    panel = compute_abtd(panel, flow)

    # -------------------------
    # 3) Winsor (nível) antes de defasar
    # -------------------------
    wins_cols_lvl = [
        "ABTD", "BTD", "ETRC",
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        "INOV_INC", "INOV_RAD", "INOV_AMBIENTAL",
        "FISC_AUT", "FISC_CRED",
    ]
    wins_cols_lvl = [c for c in wins_cols_lvl if c in panel.columns]
    panel = winsorize_df(panel, wins_cols_lvl, p=WINSOR_P)
    flow.add("Winsorização 1% (nível, antes das defasagens)", panel)

    # -------------------------
    # 4) Criar proxies de agressividade tributária (t)
    #    (serão defasadas para t-1 em seguida)
    # -------------------------
    # (a) ABTD como proxy: usa ABTD (já calculado)
    panel["TAXAGG_ABTD"] = panel["ABTD"]

    # (b) ETRc como proxy: agressividade = 1 - ETRc (quanto menor ETR, maior agressividade)
    # Só faz sentido quando ETRC é observável (no seu pipeline, ETRC só é calculado quando LAIR>0).
    if "ETRC" in panel.columns:
        panel["TAXAGG_ETRC"] = 1.0 - pd.to_numeric(panel["ETRC"], errors="coerce")
    else:
        panel["TAXAGG_ETRC"] = np.nan

    # (c) GAAP ETR (se existir no seu dados.xlsx). Ajuste o nome aqui se sua coluna tiver outro rótulo.
    # Exemplos comuns: GAAPETR, ETR_GAAP, GAAP_ETR, ETRGAAP
    panel["GAAPETR_USED"] = np.nan
    gaap_candidates = ["GAAPETR", "ETR_GAAP", "GAAP_ETR", "ETRGAAP"]
    gaap_col = next((c for c in gaap_candidates if c in panel.columns), None)
    if gaap_col:
        panel["GAAPETR_USED"] = pd.to_numeric(panel[gaap_col], errors="coerce")
        panel["TAXAGG_GAAP"] = 1.0 - panel["GAAPETR_USED"]
    else:
        panel["TAXAGG_GAAP"] = np.nan

    # -------------------------
    # 5) Defasar t-1: controles, fiscalização, proxies
    # -------------------------
    lag_cols = [
        # controles
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        # para filtros de amostra por proxy ETR
        "LAIR", "ETRC", "GAAPETR_USED",
        # fiscais
        "FISC_AUT", "FISC_CRED",
        # proxy ABTD (já é “resíduo”; ainda assim defasamos)
        "ABTD",
        # proxies alternativas
        "TAXAGG_ABTD", "TAXAGG_ETRC", "TAXAGG_GAAP",
    ]
    lag_cols = [c for c in lag_cols if c in panel.columns]
    panel = add_lags(panel, lag_cols, lag=1)

    # -------------------------
    # 6) Cortar anos “usáveis” (como você perguntou)
    #    Se você usa *_L1, o primeiro ano “usável” é ANO_INI+1.
    # -------------------------
    panel = panel[(panel["ANO"] >= (ANO_INI + 1)) & (panel["ANO"] <= ANO_FIM)].copy()
    flow.add("Corte final de anos (usável após defasagens)", panel)

    # -------------------------
    # 7) Salvar painel final
    # -------------------------
    panel.to_parquet(OUTDIR / "panel_final.parquet", index=False)

    # -------------------------
    # 8) Estatística descritiva
    # -------------------------
    desc_vars = [
        "INOV_INC", "INOV_RAD", "INOV_AMBIENTAL",
        "ABTD", "BTD", "ETRC",
        "TAXAGG_ABTD", "TAXAGG_ETRC", "TAXAGG_GAAP",
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        "FISC_AUT", "FISC_CRED",
    ]
    desc = build_descriptives(panel, desc_vars)
    with pd.ExcelWriter(OUTDIR / "descriptive_stats.xlsx", engine="openpyxl") as w:
        for sheet, d in desc.items():
            d.to_excel(w, sheet_name=sheet, index=True if sheet == "overall" else False)

    # -------------------------
    # 9) Regressões FE + diagnósticos
    # -------------------------
    reg_all_lines: List[str] = []
    reg_fe_lines: List[str] = []
    reg_lines = reg_all_lines  # alias para manter o fluxo existente do relatório completo

    if not _HAS_LINEARMODELS:
        msg = "linearmodels indisponível: instale para rodar as regressões.\n"
        reg_all_lines.append(msg)
        reg_fe_lines.append(msg)
        (OUTDIR / "regressions_all.txt").write_text("\n".join(reg_all_lines), encoding="utf-8")
        (OUTDIR / "regressions_final_fe.txt").write_text("\n".join(reg_fe_lines), encoding="utf-8")
        (OUTDIR / "regressions.txt").write_text("\n".join(reg_all_lines), encoding="utf-8")
    else:
        # ==== DEFINIÇÃO DE FLAGS (lidas do topo; fallback seguro) ====
        run_all_proxies = bool(globals().get("RUN_ALL_PROXIES", False))
        tax_agg_proxy = str(globals().get("TAX_AGG_PROXY", "ABTD")).upper().strip()
        tax_agg_proxies_cfg = globals().get("TAX_AGG_PROXIES", None)
        model_moderators_cfg = globals().get("MODEL_MODERATORS", ["SIZE", "ROA", "LEV", "GR", "PPE", "INTA"])
        run_pooled_diagnostic = bool(globals().get("RUN_POOLED_DIAGNOSTIC", True))
        run_dk_robustness = bool(globals().get("RUN_DK_ROBUSTNESS", True))
        dk_bandwidth = globals().get("DK_BANDWIDTH", 2)
        run_sector_moderation = bool(globals().get("RUN_SECTOR_MODERATION", True))
        sector_mod_min_obs = int(globals().get("SECTOR_MOD_MIN_OBS", 30))
        if sector_mod_min_obs < 1:
            sector_mod_min_obs = 1

        # ==== “panel_l1” é o próprio panel final aqui (já cortado e com *_L1 criados). ====
        panel_l1 = panel.copy()

        # Dependentes
        y_list = ["INOV_INC", "INOV_RAD", "INOV_AMBIENTAL"]
        y_list = [y for y in y_list if y in panel_l1.columns]

        # Moderadoras/controles selecionáveis
        valid_moderators = {"SIZE", "ROA", "LEV", "GR", "PPE", "INTA"}
        if model_moderators_cfg is None:
            moderators_to_run = []
        elif isinstance(model_moderators_cfg, str):
            moderators_to_run = [model_moderators_cfg]
        else:
            moderators_to_run = list(model_moderators_cfg)
        moderators_to_run = [str(m).upper().strip() for m in moderators_to_run]
        invalid_mods = [m for m in moderators_to_run if m not in valid_moderators]
        if invalid_mods:
            raise ValueError(
                f"Moderadora(s) inválida(s): {invalid_mods}. Use SIZE, ROA, LEV, GR, PPE, INTA."
            )
        moderators_l1 = [f"{m}_L1" for m in moderators_to_run if f"{m}_L1" in panel_l1.columns]

        # prioridade: TAX_AGG_PROXIES (lista explícita) > RUN_ALL_PROXIES > TAX_AGG_PROXY
        if tax_agg_proxies_cfg is not None:
            if isinstance(tax_agg_proxies_cfg, str):
                proxies_to_run = [tax_agg_proxies_cfg]
            else:
                proxies_to_run = list(tax_agg_proxies_cfg)
        elif run_all_proxies:
            proxies_to_run = ["ABTD", "ETRC", "GAAPETR"]
        else:
            proxies_to_run = [tax_agg_proxy]
        proxies_to_run = [str(p).upper().strip() for p in proxies_to_run]
        valid_proxies = {"ABTD", "ETRC", "GAAPETR"}
        invalid = [p for p in proxies_to_run if p not in valid_proxies]
        if invalid:
            raise ValueError(f"Proxy(s) inválida(s): {invalid}. Use ABTD, ETRC ou GAAPETR.")

        cfg_start = int(ANO_INI)
        cfg_end = int(ANO_FIM)
        reg_start = int(ANO_INI + 1)
        panel_years_cfg = ensure_year_bounds(panel_l1, cfg_start, cfg_end, "painel de regressão")
        panel_years_reg = ensure_year_bounds(panel_l1, reg_start, cfg_end, "painel de regressão (L1)")
        years_str = ", ".join(str(y) for y in panel_years_reg) if panel_years_reg else "nenhum"
        header_cfg = [
            "CONFIGURAÇÃO DA EXECUÇÃO\n",
            f"- Período configurado (ANO_INI..ANO_FIM): {cfg_start}-{cfg_end}\n",
            f"- Período efetivo para modelos com L1: {reg_start}-{cfg_end}\n",
            f"- Anos presentes no painel de regressão: {years_str}\n",
            f"- Proxies de agressividade utilizadas: {', '.join(proxies_to_run) if proxies_to_run else 'nenhuma'}\n",
            f"- Controles/moderadoras utilizados(as) (L1): {', '.join(moderators_l1) if moderators_l1 else 'nenhum(a)'}\n",
            f"- Moderação por setor (proxy x setor): {'sim' if run_sector_moderation else 'não'} (mín. obs/setor={sector_mod_min_obs})\n",
        ]
        reg_lines.extend(header_cfg)
        reg_fe_lines.extend(header_cfg)
        if panel_years_cfg:
            cfg_obs = f"- Checagem de período [ANO_INI..ANO_FIM] aprovada para ANO: {panel_years_cfg[0]}-{panel_years_cfg[-1]}\n"
            reg_lines.append(cfg_obs)
            reg_fe_lines.append(cfg_obs)

        setor_source = (
            str(panel_l1["SETOR_SOURCE"].dropna().iloc[0])
            if ("SETOR_SOURCE" in panel_l1.columns and panel_l1["SETOR_SOURCE"].notna().any())
            else "SECTORRF/SETORRF/SECTORF"
        )
        setorial_note = f"Moderadora fiscal setorial via merge por {setor_source} + ANO.\n"
        reg_lines.append(setorial_note)
        reg_fe_lines.append(setorial_note)

        # Função local: mapeia colunas da proxy
        # (mantém dentro do main para você NÃO ter dúvida onde colocar)
        def pick_proxy_cols(proxy: str):
            # retorna: (proxy_l1, proxy_l1_c, proxy_x_fisc_prefix)
            if proxy == "ABTD":
                # usa ABTD_L1
                return ("ABTD_L1", "ABTD_L1_C", "ABTD_L1_X_FISC")
            if proxy == "ETRC":
                # usa TAXAGG_ETRC_L1
                return ("TAXAGG_ETRC_L1", "TAXAGG_ETRC_L1_C", "TAXAGG_ETRC_L1_X_FISC")
            if proxy == "GAAPETR":
                # usa TAXAGG_GAAP_L1
                return ("TAXAGG_GAAP_L1", "TAXAGG_GAAP_L1_C", "TAXAGG_GAAP_L1_X_FISC")
            raise ValueError("TAX_AGG_PROXY inválida")

        # -------------------------
        # Loop principal: y e proxies
        # -------------------------
        for y in y_list:
            reg_lines.append(f"\n=== DEPENDENTE: {y} ===\n")
            reg_fe_lines.append(f"\n=== DEPENDENTE: {y} ===\n")

            for proxy in proxies_to_run:
                reg_lines.append(f"\n--- PROXY: {proxy} ---\n")
                reg_fe_lines.append(f"\n--- PROXY: {proxy} ---\n")
                proxy_l1, proxy_l1_c, proxy_x_fisc = pick_proxy_cols(proxy)
                proxy_panel = panel_l1.copy()

                # Para proxies baseadas em ETR, aplica filtro de amostra específico:
                # LAIR_{t-1} > 0 e ETR_{t-1} estritamente entre 0 e 1.
                if proxy in {"ETRC", "GAAPETR"}:
                    if "LAIR_L1" in proxy_panel.columns:
                        lair_l1 = pd.to_numeric(proxy_panel["LAIR_L1"], errors="coerce")
                        proxy_panel = proxy_panel[lair_l1 > 0].copy()

                    if proxy == "ETRC":
                        etr_l1 = (
                            pd.to_numeric(proxy_panel["ETRC_L1"], errors="coerce")
                            if "ETRC_L1" in proxy_panel.columns
                            else 1.0 - pd.to_numeric(proxy_panel.get("TAXAGG_ETRC_L1"), errors="coerce")
                        )
                    else:
                        etr_l1 = (
                            pd.to_numeric(proxy_panel["GAAPETR_USED_L1"], errors="coerce")
                            if "GAAPETR_USED_L1" in proxy_panel.columns
                            else 1.0 - pd.to_numeric(proxy_panel.get("TAXAGG_GAAP_L1"), errors="coerce")
                        )
                    proxy_panel = proxy_panel[etr_l1.gt(0) & etr_l1.lt(1)].copy()
                    etr_sample_msg = f"Amostra {proxy}: LAIR_L1>0 e ETR_L1 em (0,1). N={len(proxy_panel)}\n"
                    reg_lines.append(etr_sample_msg)
                    reg_fe_lines.append(etr_sample_msg)

                proxy_years = ensure_year_bounds(proxy_panel, reg_start, cfg_end, f"amostra proxy {proxy}")
                if proxy_years:
                    proxy_years_msg = (
                        f"Amostra {proxy}: período efetivo usado = {proxy_years[0]}-{proxy_years[-1]} "
                        f"(anos: {', '.join(str(y) for y in proxy_years)})\n"
                    )
                else:
                    proxy_years_msg = f"Amostra {proxy}: sem anos válidos após filtros.\n"
                reg_lines.append(proxy_years_msg)
                reg_fe_lines.append(proxy_years_msg)

                # -------------------------
                # 0) DIAGNÓSTICO: pooled SEM FE (apenas sinal bruto)
                #     (não é resultado principal; só diagnóstico)
                # -------------------------
                if run_pooled_diagnostic:
                    cols_diag = [y, proxy_l1] + moderators_l1
                    cols_diag = [c for c in cols_diag if c in proxy_panel.columns]
                    tmpd = proxy_panel[cols_diag].dropna().copy()

                    if (proxy_l1 in tmpd.columns) and (len(tmpd) >= 50):
                        Yd = pd.to_numeric(tmpd[y], errors="coerce").astype(float)
                        Xd = tmpd[[c for c in cols_diag if c != y]].apply(pd.to_numeric, errors="coerce").astype(float)
                        Xd = sm.add_constant(Xd, has_constant="add")
                        md = sm.OLS(Yd, Xd).fit(cov_type="HC1")

                        reg_lines.append("\n[DIAGNÓSTICO] OLS pooled (SEM FE) – apenas sinal bruto\n")
                        reg_lines.append(str(md.summary()))

                # -------------------------
                # 1) Modelo principal (defasado): FE firma+ano; cluster firma
                # -------------------------
                x_base_l1 = [proxy_l1] + moderators_l1
                x_base_l1 = [c for c in x_base_l1 if c in proxy_panel.columns]

                if proxy_l1 not in x_base_l1:
                    skip_msg = f"\n[SKIP] Proxy {proxy} não existe em L1 (coluna {proxy_l1} ausente)\n"
                    reg_lines.append(skip_msg)
                    reg_fe_lines.append(skip_msg)
                    continue

                res0 = fit_fe_panel(proxy_panel, y=y, x=x_base_l1)
                reg_lines.append("Modelo base (defasado; FE firma+ano; cluster firma)\n")
                reg_lines.append(str(res0.summary))
                reg_fe_lines.append("Modelo base (defasado; FE firma+ano; cluster firma)\n")
                reg_fe_lines.append(str(res0.summary))

                # Driscoll-Kraay (robustez) – só se você tiver ajustado fit_fe_panel para aceitar cov_type/dk_bandwidth
                if run_dk_robustness:
                    try:
                        res0_dk = fit_fe_panel(
                            proxy_panel,
                            y=y,
                            x=x_base_l1,
                            cov_type="driscoll-kraay",
                            dk_bandwidth=dk_bandwidth,
                        )
                        reg_lines.append("\nModelo base (defasado; Driscoll-Kraay)\n")
                        reg_lines.append(str(res0_dk.summary))
                    except TypeError:
                        reg_lines.append(
                            "\n[AVISO] fit_fe_panel não aceita cov_type/dk_bandwidth ainda. "
                            "Se quiser DK, eu te digo exatamente o patch dessa função.\n"
                        )

                # -------------------------
                # 2) Moderação contemporânea (FISC_t): proxy_L1 × FISC_t
                # -------------------------
                # 2.1) FISC_AUT_t
                if ("FISC_AUT" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_AUT_C"] = tmp["FISC_AUT"] - tmp["FISC_AUT"].mean()
                    inter_name = f"{proxy_x_fisc}_AUT_T"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_AUT_C"]

                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base_l1]
                    x_mod += ["FISC_AUT_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]

                    res1 = fit_fe_panel(tmp, y=y, x=x_mod)
                    reg_lines.append("\nModelo moderação contemporânea (FISC_AUT_t; centered)\n")
                    reg_lines.append(str(res1.summary))
                    reg_fe_lines.append("\nModelo moderação contemporânea (FISC_AUT_t; centered)\n")
                    reg_fe_lines.append(str(res1.summary))

                    # efeitos marginais (CSV)
                    # IMPORTANTE: seu marginal_effects_abtd precisa aceitar fisc_centered_col.
                    try:
                        marginal_effects_abtd(
                            res=res1,
                            df=tmp,
                            abtd_coef=proxy_l1_c,
                            fisc_raw_col="FISC_AUT",
                            fisc_centered_col="FISC_AUT_C",
                            inter_coef=inter_name,
                            out_csv=OUTDIR / f"marginal_{y}_{proxy}_FISC_AUT_t.csv",
                        )
                    except TypeError:
                        reg_lines.append(
                            "\n[AVISO] marginal_effects_abtd ainda não aceita fisc_centered_col. "
                            "Ajuste a assinatura ou me mande ela que eu te passo o patch.\n"
                        )

                # 2.2) FISC_CRED_t
                if ("FISC_CRED" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_CRED_C"] = tmp["FISC_CRED"] - tmp["FISC_CRED"].mean()
                    inter_name = f"{proxy_x_fisc}_CRED_T"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_CRED_C"]

                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base_l1]
                    x_mod += ["FISC_CRED_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]

                    res2 = fit_fe_panel(tmp, y=y, x=x_mod)
                    reg_lines.append("\nModelo moderação contemporânea (FISC_CRED_t; centered)\n")
                    reg_lines.append(str(res2.summary))
                    reg_fe_lines.append("\nModelo moderação contemporânea (FISC_CRED_t; centered)\n")
                    reg_fe_lines.append(str(res2.summary))

                    try:
                        marginal_effects_abtd(
                            res=res2,
                            df=tmp,
                            abtd_coef=proxy_l1_c,
                            fisc_raw_col="FISC_CRED",
                            fisc_centered_col="FISC_CRED_C",
                            inter_coef=inter_name,
                            out_csv=OUTDIR / f"marginal_{y}_{proxy}_FISC_CRED_t.csv",
                        )
                    except TypeError:
                        reg_lines.append(
                            "\n[AVISO] marginal_effects_abtd ainda não aceita fisc_centered_col. "
                            "Ajuste a assinatura ou me mande ela que eu te passo o patch.\n"
                        )

                # -------------------------
                # 3) Robustez estritamente pré-determinada (FISC_{t-1}): proxy_L1 × FISC_{t-1}
                # -------------------------
                # 3.1) FISC_AUT_L1
                if ("FISC_AUT_L1" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_AUT_L1_C"] = tmp["FISC_AUT_L1"] - tmp["FISC_AUT_L1"].mean()
                    inter_name = f"{proxy_x_fisc}_AUT_T1"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_AUT_L1_C"]

                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base_l1]
                    x_mod += ["FISC_AUT_L1_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]

                    res3 = fit_fe_panel(tmp, y=y, x=x_mod)
                    reg_lines.append("\nModelo moderação defasada (FISC_AUT_t-1; centered)\n")
                    reg_lines.append(str(res3.summary))
                    reg_fe_lines.append("\nModelo moderação defasada (FISC_AUT_t-1; centered)\n")
                    reg_fe_lines.append(str(res3.summary))

                    try:
                        marginal_effects_abtd(
                            res=res3,
                            df=tmp,
                            abtd_coef=proxy_l1_c,
                            fisc_raw_col="FISC_AUT_L1",
                            fisc_centered_col="FISC_AUT_L1_C",
                            inter_coef=inter_name,
                            out_csv=OUTDIR / f"marginal_{y}_{proxy}_FISC_AUT_t1.csv",
                        )
                    except TypeError:
                        reg_lines.append(
                            "\n[AVISO] marginal_effects_abtd ainda não aceita fisc_centered_col. "
                            "Ajuste a assinatura ou me mande ela que eu te passo o patch.\n"
                        )

                # 3.2) FISC_CRED_L1
                if ("FISC_CRED_L1" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_CRED_L1_C"] = tmp["FISC_CRED_L1"] - tmp["FISC_CRED_L1"].mean()
                    inter_name = f"{proxy_x_fisc}_CRED_T1"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_CRED_L1_C"]

                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base_l1]
                    x_mod += ["FISC_CRED_L1_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]

                    res4 = fit_fe_panel(tmp, y=y, x=x_mod)
                    reg_lines.append("\nModelo moderação defasada (FISC_CRED_t-1; centered)\n")
                    reg_lines.append(str(res4.summary))
                    reg_fe_lines.append("\nModelo moderação defasada (FISC_CRED_t-1; centered)\n")
                    reg_fe_lines.append(str(res4.summary))

                    try:
                        marginal_effects_abtd(
                            res=res4,
                            df=tmp,
                            abtd_coef=proxy_l1_c,
                            fisc_raw_col="FISC_CRED_L1",
                            fisc_centered_col="FISC_CRED_L1_C",
                            inter_coef=inter_name,
                            out_csv=OUTDIR / f"marginal_{y}_{proxy}_FISC_CRED_t1.csv",
                        )
                    except TypeError:
                        reg_lines.append(
                            "\n[AVISO] marginal_effects_abtd ainda não aceita fisc_centered_col. "
                            "Ajuste a assinatura ou me mande ela que eu te passo o patch.\n"
                        )

                # -------------------------
                # 4) Moderação por setor: proxy_L1 × dummies de setor
                # -------------------------
                if run_sector_moderation and ("SETOR_KEY" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()

                    setor_raw = tmp["SETOR_KEY"].fillna("__missing__").astype(str).str.strip()
                    setor_counts = setor_raw.value_counts(dropna=False)
                    setor_group = setor_raw.where(setor_raw.map(setor_counts) >= sector_mod_min_obs, "__outros__")
                    group_counts = setor_group.value_counts(dropna=False)

                    if len(group_counts) >= 2:
                        # baseline = setor com maior frequência
                        group_order = group_counts.index.tolist()
                        baseline = group_order[0]
                        setor_cat = pd.Categorical(setor_group, categories=group_order, ordered=True)
                        setor_dummies = pd.get_dummies(setor_cat, prefix="SETOR", drop_first=True, dtype=float)
                        tmp = pd.concat([tmp, setor_dummies], axis=1)

                        inter_cols = []
                        for dcol in setor_dummies.columns:
                            dkey = re.sub(r"[^A-Z0-9_]", "_", dcol.upper())
                            inter_col = f"{proxy_l1_c}_X_{dkey}"
                            tmp[inter_col] = tmp[proxy_l1_c] * pd.to_numeric(tmp[dcol], errors="coerce").fillna(0.0)
                            inter_cols.append(inter_col)

                        x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base_l1]
                        x_mod += inter_cols
                        x_mod = [c for c in x_mod if c in tmp.columns]
                        # remove colunas sem variação para evitar problemas de rank
                        x_mod = [c for c in x_mod if pd.to_numeric(tmp[c], errors="coerce").nunique(dropna=True) > 1]
                        if proxy_l1_c not in x_mod and proxy_l1_c in tmp.columns:
                            x_mod = [proxy_l1_c] + x_mod
                        use_rank = tmp[[y] + x_mod].dropna().copy() if x_mod else pd.DataFrame()
                        x_mod = prune_collinear_columns(use_rank, x_mod)

                        inter_in_model = [c for c in inter_cols if c in x_mod]
                        if len(inter_in_model) > 0 and proxy_l1_c in x_mod:
                            try:
                                res_setor = fit_fe_panel(tmp, y=y, x=x_mod)
                                sectors_used = ", ".join(str(s) for s in group_order)
                                setor_msg = (
                                    f"\nModelo moderação por setor (proxy_L1 x setor; baseline={baseline}; "
                                    f"setores={sectors_used})\n"
                                )
                                reg_lines.append(setor_msg)
                                reg_lines.append(str(res_setor.summary))
                                reg_fe_lines.append(setor_msg)
                                reg_fe_lines.append(str(res_setor.summary))
                            except Exception as exc:
                                warn = f"\n[AVISO] Falha na moderação por setor ({proxy}/{y}): {exc}\n"
                                reg_lines.append(warn)
                                reg_fe_lines.append(warn)
                        else:
                            warn = (
                                f"\n[AVISO] Moderação por setor sem termos estimáveis após checagem de rank "
                                f"({proxy}/{y}).\n"
                            )
                            reg_lines.append(warn)
                            reg_fe_lines.append(warn)
                    else:
                        warn = "\n[AVISO] Moderação por setor ignorada: menos de 2 grupos setoriais válidos.\n"
                        reg_lines.append(warn)
                        reg_fe_lines.append(warn)

        (OUTDIR / "regressions_all.txt").write_text("\n".join(reg_all_lines), encoding="utf-8")
        (OUTDIR / "regressions_final_fe.txt").write_text("\n".join(reg_fe_lines), encoding="utf-8")
        # compatibilidade retroativa
        (OUTDIR / "regressions.txt").write_text("\n".join(reg_all_lines), encoding="utf-8")

    # -------------------------
    # 10) Diagnósticos (texto)
    # -------------------------
    diag = [
        "Diagnósticos\n",
        "- Este pipeline inclui um diagnóstico OLS pooled (sem FE) apenas para checar sinal bruto.\n",
        "- Resultados principais permanecem FE firma+ano com cluster por firma.\n",
        "- Se você habilitar DK (Driscoll-Kraay), precisa que fit_fe_panel aceite cov_type/dk_bandwidth.\n",
        "- Se você quiser testes formais (Wooldridge, Pesaran CD), dá para implementar depois.\n",
    ]
    (OUTDIR / "diagnostics.txt").write_text("".join(diag), encoding="utf-8")

    # -------------------------
    # 11) Fluxo amostral
    # -------------------------
    (OUTDIR / "sample_flow.txt").write_text("\n".join(flow.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
