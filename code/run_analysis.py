"""
Pipeline para montar o painel firma-ano, criar ABTD/proxies, rodar FE e exportar resultados.

Entradas:
  - data/dados.xlsx
  - data/relatorios-rf.xlsx

Saídas:
  - outputs/panel_final.parquet
  - outputs/regressions.txt
  - outputs/regressions_all.txt
  - outputs/regressions_final_fe.txt
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
    # .../Dados/code/run_analysis.py -> .../Dados
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


ROOT = get_root()
DATA_DIR = ROOT / "data"
INPUT_DADOS = DATA_DIR / "dados.xlsx"
INPUT_RF = DATA_DIR / "relatorios-rf.xlsx"
OUTDIR = ROOT / "outputs"

ANO_INI = 2014
ANO_FIM = 2024

TAU_STATUTARY = 0.34  # IRPJ+CSLL (use apenas se você estiver excluindo financeiras)

WINSOR_P = 0.01

# Proxy principal de agressividade tributária
# Opções: "ABTD", "ETRC", "GAAPETR"
TAX_AGG_PROXIES = ["ABTD"]
RUN_ALL_PROXIES = False  # True: roda ABTD, ETRC e GAAPETR

# Moderadoras/controles do modelo principal
# Opções válidas: "SIZE", "ROA", "LEV", "GR", "PPE", "INTA"
MODEL_MODERATORS = ["SIZE", "ROA", "LEV", "GR", "PPE", "INTA"]

# Filtro de LAIR<=0 por objetivo:
# - ABTD: mantém LAIR<=0 no padrão.
# - ETR: filtra LAIR<=0 no padrão.
DROP_LAIR_NONPOS_FOR_ABTD = False
DROP_LAIR_NONPOS_FOR_ETR = True

# Compatibilidade legada:
# - True: ABTD=True e ETR=True
# - False: ABTD=False e ETR=False
# - None: usa flags separadas acima
DROP_LAIR_NONPOS = None

# Exclusão de financeiras: tenta SETORRF/SECTORF e cai para SECTOR
EXCLUDE_FINANCIALS = True

# Inovação: True usa versão escalada por ATIVO; False usa valor bruto
INOV_SCALE_BY_ASSETS = True

# ABTD: regressão cross-section por ANO (padrão)
ABTD_BY_YEAR_AND_SECTOR = False

# Fiscalização: True usa log1p, False usa bruto
FISC_USE_LOG1P = True

# Moderação por setor: proxy_L1 x dummies de setor
RUN_SECTOR_MODERATION = True
# Mínimo de observações por setor; abaixo disso vira "__outros__".
SECTOR_MOD_MIN_OBS = 30

# Regressões FE separadas por setor (split sample)
RUN_SECTOR_SPLIT_REGRESSIONS = True
SECTOR_SPLIT_MIN_OBS = 150
SECTOR_SPLIT_MIN_FIRMS = 20
SECTOR_SPLIT_MIN_YEARS = 4
SECTOR_SPLIT_USE_GROUPING = True
SECTOR_SPLIT_EXCLUDE_OUTROS = True
OUTPUT_REGRESSIONS_BY_SECTOR = "regressions_by_sector.txt"
OUTPUT_REGRESSIONS_FINAL_FE_BY_SECTOR = "regressions_final_fe_by_sector.txt"

# Índice composto de inovação (Entropia + TOPSIS)
USE_INNOVATION_COMPOSITE = True
INNOVATION_COMPOSITE_METHOD = "entropy_topsis"
INNOVATION_COMPONENTS = ["INOV_INC", "INOV_RAD", "INOV_AMBIENTAL"]
INNOVATION_COMPOSITE_BY = "year"  # por enquanto: apenas "year"
INNOVATION_COMPOSITE_WINSORIZE = True
INNOVATION_COMPOSITE_SUFFIX = "COMP"  # gera alvo INOV_COMP

# True: missing em CAPEX/RD/AMB vira 0
# False: missing permanece NaN
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


def resolve_lair_filter_flags() -> Tuple[bool, bool]:
    """
    Resolve flags de LAIR com compatibilidade legada.
    Retorna (drop_lair_nonpos_for_abtd, drop_lair_nonpos_for_etr).
    """
    legacy = globals().get("DROP_LAIR_NONPOS", None)
    if legacy is True:
        return True, True
    if legacy is False:
        return False, False
    return bool(globals().get("DROP_LAIR_NONPOS_FOR_ABTD", False)), bool(globals().get("DROP_LAIR_NONPOS_FOR_ETR", True))


def innovation_target_name(suffix: str) -> str:
    sfx = clean_colname(suffix)
    if sfx.startswith("INOV_"):
        return sfx
    return f"INOV_{sfx}"


def _prepare_positive_criteria_matrix(df: pd.DataFrame, cols: Sequence[str], eps: float = 1e-9) -> pd.DataFrame:
    """
    Prepara matriz para critérios de benefício (maior = melhor), sem valores negativos.
    Se min <= 0, aplica shift: x := x - min + eps.
    """
    x = df[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    for c in cols:
        col_min = x[c].min(skipna=True)
        if pd.isna(col_min):
            continue
        if col_min <= 0:
            x[c] = x[c] - col_min + eps
    return x


def compute_entropy_weights(df_year: pd.DataFrame, cols: Sequence[str], eps: float = 1e-9) -> pd.Series:
    """
    Pesos por Entropia (Shannon) para um corte transversal (ex.: ano).
    """
    if len(cols) == 0:
        return pd.Series(dtype=float)

    x = _prepare_positive_criteria_matrix(df_year, cols, eps=eps)
    n = int(x.shape[0])
    if n <= 1:
        return pd.Series(np.repeat(1.0 / len(cols), len(cols)), index=list(cols), dtype=float)

    col_sum = x.sum(axis=0)
    p = x.div(col_sum.replace(0.0, np.nan), axis=1)
    p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    k = 1.0 / np.log(float(n))
    p_np = p.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        plnp = np.where(p_np > 0.0, p_np * np.log(p_np), 0.0)
    e = -k * plnp.sum(axis=0)

    d = 1.0 - e
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.clip(d, 0.0, None)
    d_sum = float(d.sum())

    if d_sum <= 0.0:
        w = np.repeat(1.0 / len(cols), len(cols))
    else:
        w = d / d_sum
    return pd.Series(w, index=list(cols), dtype=float)


def compute_topsis_score(
    df_year: pd.DataFrame,
    cols: Sequence[str],
    weights: pd.Series,
    eps: float = 1e-9,
) -> pd.Series:
    """
    Score TOPSIS para um corte transversal (ex.: ano), com critérios de benefício.
    """
    if len(cols) == 0:
        return pd.Series(index=df_year.index, dtype=float)

    x = _prepare_positive_criteria_matrix(df_year, cols, eps=eps)

    norm = np.sqrt((x**2).sum(axis=0))
    r = x.div(norm.replace(0.0, np.nan), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w = pd.Series(weights, dtype=float).reindex(list(cols))
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(w.sum()) <= 0.0:
        w[:] = 1.0 / len(cols)
    else:
        w = w / float(w.sum())

    v = r.mul(w, axis=1)

    a_pos = v.max(axis=0)
    a_neg = v.min(axis=0)

    s_pos = np.sqrt(((v - a_pos) ** 2).sum(axis=1))
    s_neg = np.sqrt(((v - a_neg) ** 2).sum(axis=1))

    denom = s_pos + s_neg
    score = pd.Series(np.where(denom > 0.0, s_neg / denom, 0.5), index=v.index, dtype=float)
    return score


def build_innovation_composite_entropy_topsis(
    df: pd.DataFrame,
    components: Sequence[str],
    by: str = "year",
    winsorize_components: bool = True,
    winsor_p: float = 0.01,
    min_obs: int = 5,
    suffix: str = "COMP",
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Constrói INOV_<SUFFIX> por Entropia + TOPSIS (hoje: por ano).
    Calcula score só para linhas com todos os componentes.
    """
    out = df.copy()
    target = innovation_target_name(suffix)
    logs: List[str] = []
    out[target] = np.nan

    cols = [c for c in components if c in out.columns]
    missing = [c for c in components if c not in out.columns]
    if missing:
        logs.append(f"[AVISO] Índice composto: componentes ausentes: {', '.join(missing)}.")
        logs.append("[AVISO] Índice composto: cálculo requer todos os componentes; variável não calculada.")
        return out, target, logs
    if len(cols) == 0:
        logs.append("[AVISO] Índice composto: nenhum componente disponível; variável não calculada.")
        return out, target, logs

    if by != "year":
        raise ValueError("INNOVATION_COMPOSITE_BY inválido. Atualmente, apenas 'year' é suportado.")

    work = out[["ANO"] + cols].copy()
    if winsorize_components:
        work = winsorize_df(work, cols, p=winsor_p)

    for ano, idx in work.groupby("ANO", dropna=False).groups.items():
        g = work.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
        valid = g.notna().all(axis=1)
        g_valid = g.loc[valid].copy()
        n_valid = int(g_valid.shape[0])
        ano_lbl = int(ano) if pd.notna(ano) else ano

        if n_valid < min_obs:
            logs.append(
                f"[AVISO] Índice composto ano {ano_lbl}: N válido={n_valid} (< {min_obs}); score não calculado."
            )
            continue

        weights = compute_entropy_weights(g_valid, cols)
        score = compute_topsis_score(g_valid, cols, weights)
        out.loc[g_valid.index, target] = score.values

        s = out.loc[g_valid.index, target]
        w_desc = ", ".join(f"{c}={float(weights[c]):.6f}" for c in cols)
        logs.append(
            f"Ano {ano_lbl} | pesos: {w_desc} | "
            f"{target}: min={float(s.min()):.6f}, mediana={float(s.median()):.6f}, "
            f"media={float(s.mean()):.6f}, max={float(s.max()):.6f}, N={n_valid}"
        )

    return out, target, logs


# =========================
# Leitura e reshape
# =========================
def read_dados_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [clean_colname(c) for c in df.columns]
    # normaliza aliases comuns de setor RF
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

    # pega colunas VAR+ANO; ignora ANO2024/ANO2023... para não duplicar "ANO"
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

    # se houver coluna duplicada, mantém a última
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

    # remove linha TOTAL
    long = long[~long["SETOR"].str.upper().eq("TOTAL")].copy()

    long["FISC_AUT_LN"] = np.log1p(pd.to_numeric(long["FISC_AUT_RAW"], errors="coerce"))
    long["FISC_CRED_LN"] = np.log1p(pd.to_numeric(long["FISC_CRED_RAW"], errors="coerce"))

    # padrão: usa bruto; robustez: *_LN
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

    # prioridade: SECTORRF/SETORRF/SECTORF
    for c in ["SECTORRF", "SETORRF", "SECTORF"]:
        if c in out.columns:
            key = out[c].map(norm_key)
            # remove "servicos financeiros"
            mask_fin = key.eq(norm_key("Servicos financeiros"))
            return out.loc[~mask_fin].copy()

    # fallback: SECTOR (LSEG)
    if "SECTOR" in out.columns:
        mask_fin = out["SECTOR"].astype(str).str.strip().str.lower().eq("financials")
        return out.loc[~mask_fin].copy()

    return out


def build_base_panel(df_long: pd.DataFrame, rf_long: pd.DataFrame, flow: FlowLog) -> pd.DataFrame:
    df = df_long.copy()

    # mantém ANO_INI-1 para montar defasagens de ANO_INI
    df = df[(df["ANO"] >= (ANO_INI - 1)) & (df["ANO"] <= ANO_FIM)].copy()
    flow.add(f"Filtro período {ANO_INI}-{ANO_FIM}", df)

    # escolhe coluna de setor para merge com RF
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

    # exclui financeiras
    df = exclude_financials(df)
    flow.add("Exclusão de financeiras", df)

    # remove ATIVO <= 0 ou nulo
    if "ATIVO" not in df.columns:
        raise ValueError("Não encontrei coluna ATIVO.")
    df = df[df["ATIVO"].notna() & (df["ATIVO"] > 0)].copy()
    flow.add("Filtro ATIVO > 0", df)

    # remove só LAIR nulo (LAIR<=0 fica nesta etapa)
    if "LAIR" not in df.columns:
        raise ValueError("Não encontrei coluna LAIR.")
    df = df[df["LAIR"].notna()].copy()
    flow.add("Filtro LAIR não nulo (LAIR<=0 preservado nesta etapa)", df)

    # missing em inovação: vira 0 ou fica NaN
    if INOV_MISSING_AS_ZERO:
        for v in ["CAPEX", "RD", "AMB"]:
            if v in df.columns:
                df[v] = df[v].fillna(0)
        flow.add("Missing inovação -> 0 (CAPEX, RD, AMB)", df)
    else:
        flow.add("Missing inovação mantido como NaN (CAPEX, RD, AMB)", df)

    # ETRc = IMPCOR/LAIR só quando LAIR>0; inválidos viram NaN
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

    # ordena e cria defasagens base
    df = df.sort_values(["RIC", "ANO"]).copy()
    df["ATIVO_L1"] = df.groupby("RIC")["ATIVO"].shift(1)
    df["RL_L1"] = df.groupby("RIC")["RL"].shift(1) if "RL" in df.columns else np.nan
    df["REC_L1"] = df.groupby("RIC")["REC"].shift(1) if "REC" in df.columns else np.nan

    # NOL
    df["NOL"] = (df["LAIR"] < 0).astype("int64")

    # controles em t (depois viram L1)
    df["SIZE"] = np.log(df["ATIVO"])
    df["ROA"] = safe_div(df["LAIR"], df["ATIVO"])  # alternativa: LP/ATIVO
    df["LEV"] = safe_div(df["DLP"], df["ATIVO_L1"]) if "DLP" in df.columns else np.nan
    if "RL" in df.columns and df["RL_L1"].notna().any():
        df["GR"] = (df["RL"] - df["RL_L1"]) / df["RL_L1"]
        df.loc[~np.isfinite(df["GR"]), "GR"] = np.nan
    else:
        df["GR"] = np.nan

    df["PPE"] = safe_div(df["IMOB"], df["ATIVO"]) if "IMOB" in df.columns else np.nan
    df["INTA"] = safe_div(df["INTANG"], df["ATIVO"]) if "INTANG" in df.columns else np.nan

    # inovação: bruto e (opcional) escalado
    df["INOV_INC_RAW"] = df["CAPEX"] if "CAPEX" in df.columns else np.nan
    df["INOV_RAD_RAW"] = df["RD"] if "RD" in df.columns else np.nan
    df["INOV_AMBIENTAL"] = df["AMB"] if "AMB" in df.columns else np.nan

    df["INOV_INC"] = safe_div(df["CAPEX"], df["ATIVO"]) if INOV_SCALE_BY_ASSETS else df["INOV_INC_RAW"]
    df["INOV_RAD"] = safe_div(df["RD"], df["ATIVO"]) if INOV_SCALE_BY_ASSETS else df["INOV_RAD_RAW"]

    # BTD: (LAIR - TI)/ATIVO_L1, com TI = IMPCOR/tau
    df["TI"] = df["IMPCOR"] / TAU_STATUTARY if "IMPCOR" in df.columns else np.nan
    df["BTD"] = (df["LAIR"] - df["TI"]) / df["ATIVO_L1"]
    df.loc[~np.isfinite(df["BTD"]), "BTD"] = np.nan

    # variáveis do modelo normal do BTD (ATIVO_L1)
    df["PPE_L1S"] = safe_div(df["IMOB"], df["ATIVO_L1"]) if "IMOB" in df.columns else np.nan
    df["INTA_L1S"] = safe_div(df["INTANG"], df["ATIVO_L1"]) if "INTANG" in df.columns else np.nan
    df["SIZE_L1S"] = np.log(df["ATIVO_L1"].where(df["ATIVO_L1"] > 0))

    # ΔReceita: usa REC; se não tiver, usa RL (sempre /ATIVO_L1)
    if "REC" in df.columns and df["REC"].notna().any():
        df["DREV_L1S"] = safe_div(df["REC"] - df["REC_L1"], df["ATIVO_L1"])
    elif "RL" in df.columns and df["RL"].notna().any():
        df["DREV_L1S"] = safe_div(df["RL"] - df["RL_L1"], df["ATIVO_L1"])
    else:
        df["DREV_L1S"] = np.nan

    flow.add("Construção variáveis base (controles, inovação, BTD)", df)
    return df


def compute_abtd(df: pd.DataFrame, flow: FlowLog, drop_lair_nonpos_for_abtd: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["ABTD"] = np.nan

    rhs = ["PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S", "NOL"]
    if drop_lair_nonpos_for_abtd:
        rhs = [c for c in rhs if c != "NOL"]
        flow.lines.append("Aviso: ABTD com LAIR>0 -> NOL removido do modelo normal do BTD por coerência.")
    need = ["BTD"] + rhs

    # winsoriza insumos do ABTD antes de estimar
    ins_cols = ["BTD", "PPE_L1S", "INTA_L1S", "DREV_L1S", "SIZE_L1S"]
    out = winsorize_df(out, [c for c in ins_cols if c in out.columns], p=WINSOR_P)

    group_cols = ["ANO"]
    if ABTD_BY_YEAR_AND_SECTOR:
        group_cols = ["ANO", "SETOR_KEY"]

    for key, g in out.groupby(group_cols, dropna=False):
        g = g.dropna(subset=need).copy()
        k = len(rhs)
        # mínimo para cross-section anual
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

    # se cluster não for entity/time, inclui a coluna no subset
    cols = [entity, time, y] + list(x)
    if cov_type == "clustered" and cluster not in (entity, time) and cluster not in cols:
        cols.append(cluster)

    use = df[cols].dropna().copy()
    use = use.set_index([entity, time])

    Y = use[y]
    X = sm.add_constant(use[list(x)], has_constant="add")

    mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)

    if cov_type == "clustered":
        # monta vetor de cluster no índice do painel
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

    # erro-padrão clusterizado por firma (diagnóstico)
    res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": use[entity]})
    return res


def run_sector_split_regressions(
    panel_l1: pd.DataFrame,
    y_list: Sequence[str],
    proxies_to_run: Sequence[str],
    moderators_l1: Sequence[str],
    reg_start: int,
    cfg_end: int,
    outdir: Path,
    sector_split_min_obs: int = 150,
    sector_split_min_firms: int = 20,
    sector_split_min_years: int = 4,
    sector_split_use_grouping: bool = True,
    sector_split_exclude_outros: bool = True,
    sector_mod_min_obs: int = 30,
    drop_lair_nonpos_for_abtd: bool = False,
    drop_lair_nonpos_for_etr: bool = True,
    output_regressions_by_sector: str = "regressions_by_sector.txt",
    output_regressions_final_fe_by_sector: str = "regressions_final_fe_by_sector.txt",
) -> Tuple[Path, Path]:
    """
    Roda regressões FE firma+ano (cluster firma) separadas por setor.
    """
    out_all = outdir / output_regressions_by_sector
    out_fe = outdir / output_regressions_final_fe_by_sector

    reg_all_lines: List[str] = []
    reg_fe_lines: List[str] = []

    def _emit(msg: str) -> None:
        reg_all_lines.append(msg)
        reg_fe_lines.append(msg)

    _emit("REGRESSOES FE POR SETOR (SPLIT SAMPLE)\n")
    _emit(f"- periodo efetivo esperado: {reg_start}-{cfg_end}\n")
    _emit(f"- min obs={sector_split_min_obs}; min firms={sector_split_min_firms}; min years={sector_split_min_years}\n")
    _emit(f"- Filtro LAIR: ABTD={'on' if drop_lair_nonpos_for_abtd else 'off'}, ETR={'on' if drop_lair_nonpos_for_etr else 'off'}\n")
    _emit(
        f"- grouping={'sim' if sector_split_use_grouping else 'nao'} "
        f"(cutoff={sector_mod_min_obs}); exclude_outros={'sim' if sector_split_exclude_outros else 'nao'}\n"
    )
    if drop_lair_nonpos_for_abtd:
        _emit("Aviso: ABTD com LAIR>0 -> NOL removido do modelo normal do BTD por coerência.\n")

    if "SETOR_KEY" not in panel_l1.columns:
        _emit("[SKIP] Coluna SETOR_KEY ausente no painel final.\n")
        out_all.write_text("\n".join(reg_all_lines), encoding="utf-8")
        out_fe.write_text("\n".join(reg_fe_lines), encoding="utf-8")
        return out_all, out_fe

    def _pick_proxy_cols_local(proxy: str) -> Tuple[str, str, str]:
        p = str(proxy).upper().strip()
        if p == "ABTD":
            return ("ABTD_L1", "ABTD_L1_C", "ABTD_L1_X_FISC")
        if p == "ETRC":
            return ("TAXAGG_ETRC_L1", "TAXAGG_ETRC_L1_C", "TAXAGG_ETRC_L1_X_FISC")
        if p == "GAAPETR":
            return ("TAXAGG_GAAP_L1", "TAXAGG_GAAP_L1_C", "TAXAGG_GAAP_L1_X_FISC")
        raise ValueError(f"Proxy inválida para split por setor: {proxy}")

    for y in y_list:
        _emit(f"\n=== DEPENDENTE: {y} ===\n")

        for proxy in proxies_to_run:
            proxy_l1, proxy_l1_c, proxy_x_fisc = _pick_proxy_cols_local(proxy)
            proxy_panel = panel_l1.copy()
            proxy_u = str(proxy).upper().strip()

            # mesmo filtro de amostra das proxies ETR
            if proxy_u in {"ETRC", "GAAPETR"}:
                if drop_lair_nonpos_for_etr and "LAIR_L1" in proxy_panel.columns:
                    lair_l1 = pd.to_numeric(proxy_panel["LAIR_L1"], errors="coerce")
                    proxy_panel = proxy_panel[lair_l1 > 0].copy()

                if proxy_u == "ETRC":
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
            elif proxy_u == "ABTD" and drop_lair_nonpos_for_abtd and "LAIR_L1" in proxy_panel.columns:
                lair_l1 = pd.to_numeric(proxy_panel["LAIR_L1"], errors="coerce")
                proxy_panel = proxy_panel[lair_l1 > 0].copy()

            try:
                _ = ensure_year_bounds(proxy_panel, reg_start, cfg_end, f"split-setor {proxy}/{y}")
            except RuntimeError as exc:
                _emit(f"\n[SKIP] Proxy {proxy} / {y}: {exc}\n")
                continue

            setor_raw = proxy_panel["SETOR_KEY"].fillna("__missing__").astype(str).str.strip().replace("", "__missing__")
            if sector_split_use_grouping:
                setor_counts = setor_raw.value_counts(dropna=False)
                setor_group = setor_raw.where(setor_raw.map(setor_counts) >= sector_mod_min_obs, "__outros__")
            else:
                setor_group = setor_raw
            proxy_panel = proxy_panel.copy()
            proxy_panel["_SECTOR_SPLIT_GROUP"] = setor_group

            _emit(f"\n--- PROXY: {proxy} ---\n")

            groups = proxy_panel["_SECTOR_SPLIT_GROUP"].value_counts(dropna=False).index.tolist()
            for setor_name in groups:
                if sector_split_exclude_outros and str(setor_name) == "__outros__":
                    _emit("[SKIP] setor __outros__: excluído por configuração.\n")
                    continue

                subset = proxy_panel[proxy_panel["_SECTOR_SPLIT_GROUP"] == setor_name].copy()
                obs = int(len(subset))
                firms = int(subset["RIC"].nunique()) if "RIC" in subset.columns else 0
                years = sorted(
                    set(pd.to_numeric(subset["ANO"], errors="coerce").dropna().astype(int).tolist())
                ) if "ANO" in subset.columns else []
                n_years = len(years)
                years_txt = ", ".join(str(v) for v in years) if years else "nenhum"

                _emit(f"\nSETOR: {setor_name} | N obs={obs} | N firms={firms} | anos={years_txt}\n")

                skip_reasons: List[str] = []
                if obs < sector_split_min_obs:
                    skip_reasons.append(f"obs<{sector_split_min_obs}")
                if firms < sector_split_min_firms:
                    skip_reasons.append(f"firms<{sector_split_min_firms}")
                if n_years < sector_split_min_years:
                    skip_reasons.append(f"years<{sector_split_min_years}")

                if skip_reasons:
                    _emit(f"[SKIP] setor {setor_name}: {'; '.join(skip_reasons)}\n")
                    continue

                x_base = [proxy_l1] + [c for c in moderators_l1 if c in subset.columns]
                x_base = [c for c in x_base if c in subset.columns]
                if proxy_l1 not in x_base:
                    _emit(f"[SKIP] setor {setor_name}: coluna proxy ausente ({proxy_l1}).\n")
                    continue
                if y not in subset.columns:
                    _emit(f"[SKIP] setor {setor_name}: dependente ausente ({y}).\n")
                    continue

                use_base = subset[["RIC", "ANO", y] + x_base].dropna().copy()
                if use_base.empty:
                    _emit(f"[SKIP] setor {setor_name}: sem observações válidas para o modelo base.\n")
                    continue
                if pd.to_numeric(use_base[y], errors="coerce").nunique(dropna=True) <= 1:
                    _emit(f"[SKIP] setor {setor_name}: dependente sem variação ({y}).\n")
                    continue
                if pd.to_numeric(use_base[proxy_l1], errors="coerce").nunique(dropna=True) <= 1:
                    _emit(f"[SKIP] setor {setor_name}: proxy sem variação ({proxy_l1}).\n")
                    continue
                if use_base["RIC"].nunique() < 2:
                    _emit(f"[SKIP] setor {setor_name}: menos de 2 firmas após dropna.\n")
                    continue
                if use_base["ANO"].nunique() < 2:
                    _emit(f"[SKIP] setor {setor_name}: menos de 2 anos após dropna.\n")
                    continue

                # (a) modelo base FE firma+ano
                try:
                    res0 = fit_fe_panel(subset, y=y, x=x_base)
                    _emit("Modelo base (FE firma+ano; cluster firma)\n")
                    _emit(str(res0.summary))
                except Exception as exc:
                    _emit(f"[SKIP] setor {setor_name}: falha no modelo base ({exc}).\n")
                    continue

                # (b) moderação contemporânea com FISC_AUT_t
                if "FISC_AUT" in subset.columns:
                    tmp = subset.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_AUT_C"] = tmp["FISC_AUT"] - tmp["FISC_AUT"].mean()
                    inter_name = f"{proxy_x_fisc}_AUT_T_SPLIT"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_AUT_C"]
                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base] + ["FISC_AUT_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]
                    use_mod = tmp[["RIC", "ANO", y] + x_mod].dropna().copy()
                    if use_mod.empty or use_mod[y].nunique() <= 1 or use_mod[proxy_l1_c].nunique() <= 1:
                        _emit(f"[SKIP] setor {setor_name}: sem variação para moderação FISC_AUT_t.\n")
                    else:
                        try:
                            res1 = fit_fe_panel(tmp, y=y, x=x_mod)
                            _emit("Modelo moderação contemporânea (FISC_AUT_t; centered)\n")
                            _emit(str(res1.summary))
                        except Exception as exc:
                            _emit(f"[SKIP] setor {setor_name}: falha moderação FISC_AUT_t ({exc}).\n")
                else:
                    _emit(f"[SKIP] setor {setor_name}: FISC_AUT ausente.\n")

                # (c) moderação contemporânea com FISC_CRED_t
                if "FISC_CRED" in subset.columns:
                    tmp = subset.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_CRED_C"] = tmp["FISC_CRED"] - tmp["FISC_CRED"].mean()
                    inter_name = f"{proxy_x_fisc}_CRED_T_SPLIT"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_CRED_C"]
                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base] + ["FISC_CRED_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]
                    use_mod = tmp[["RIC", "ANO", y] + x_mod].dropna().copy()
                    if use_mod.empty or use_mod[y].nunique() <= 1 or use_mod[proxy_l1_c].nunique() <= 1:
                        _emit(f"[SKIP] setor {setor_name}: sem variação para moderação FISC_CRED_t.\n")
                    else:
                        try:
                            res2 = fit_fe_panel(tmp, y=y, x=x_mod)
                            _emit("Modelo moderação contemporânea (FISC_CRED_t; centered)\n")
                            _emit(str(res2.summary))
                        except Exception as exc:
                            _emit(f"[SKIP] setor {setor_name}: falha moderação FISC_CRED_t ({exc}).\n")
                else:
                    _emit(f"[SKIP] setor {setor_name}: FISC_CRED ausente.\n")

                # (d) moderação defasada com FISC_AUT_t-1
                if "FISC_AUT_L1" in subset.columns:
                    tmp = subset.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_AUT_L1_C"] = tmp["FISC_AUT_L1"] - tmp["FISC_AUT_L1"].mean()
                    inter_name = f"{proxy_x_fisc}_AUT_T1_SPLIT"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_AUT_L1_C"]
                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base] + ["FISC_AUT_L1_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]
                    use_mod = tmp[["RIC", "ANO", y] + x_mod].dropna().copy()
                    if use_mod.empty or use_mod[y].nunique() <= 1 or use_mod[proxy_l1_c].nunique() <= 1:
                        _emit(f"[SKIP] setor {setor_name}: sem variação para moderação FISC_AUT_t-1.\n")
                    else:
                        try:
                            res3 = fit_fe_panel(tmp, y=y, x=x_mod)
                            _emit("Modelo moderação defasada (FISC_AUT_t-1; centered)\n")
                            _emit(str(res3.summary))
                        except Exception as exc:
                            _emit(f"[SKIP] setor {setor_name}: falha moderação FISC_AUT_t-1 ({exc}).\n")

                # (e) moderação defasada com FISC_CRED_t-1
                if "FISC_CRED_L1" in subset.columns:
                    tmp = subset.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()
                    tmp["FISC_CRED_L1_C"] = tmp["FISC_CRED_L1"] - tmp["FISC_CRED_L1"].mean()
                    inter_name = f"{proxy_x_fisc}_CRED_T1_SPLIT"
                    tmp[inter_name] = tmp[proxy_l1_c] * tmp["FISC_CRED_L1_C"]
                    x_mod = [proxy_l1_c if c == proxy_l1 else c for c in x_base] + ["FISC_CRED_L1_C", inter_name]
                    x_mod = [c for c in x_mod if c in tmp.columns]
                    use_mod = tmp[["RIC", "ANO", y] + x_mod].dropna().copy()
                    if use_mod.empty or use_mod[y].nunique() <= 1 or use_mod[proxy_l1_c].nunique() <= 1:
                        _emit(f"[SKIP] setor {setor_name}: sem variação para moderação FISC_CRED_t-1.\n")
                    else:
                        try:
                            res4 = fit_fe_panel(tmp, y=y, x=x_mod)
                            _emit("Modelo moderação defasada (FISC_CRED_t-1; centered)\n")
                            _emit(str(res4.summary))
                        except Exception as exc:
                            _emit(f"[SKIP] setor {setor_name}: falha moderação FISC_CRED_t-1 ({exc}).\n")

    out_all.write_text("\n".join(reg_all_lines), encoding="utf-8")
    out_fe.write_text("\n".join(reg_fe_lines), encoding="utf-8")
    return out_all, out_fe


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
    """Valida ANO no intervalo e retorna anos únicos ordenados."""
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
    Mantém um subconjunto de colunas linearmente independentes.
    Usa rank incremental com ordem preservada.
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
    drop_lair_nonpos_for_abtd, drop_lair_nonpos_for_etr = resolve_lair_filter_flags()

    # 1) Leitura e reshape
    df_wide = read_dados_xlsx(INPUT_DADOS)
    df_long = wide_to_long_firma_ano(df_wide)
    flow.add("Wide->Long (firma-ano)", df_long)

    rf_wide = read_rf_xlsx(INPUT_RF)
    rf_long = rf_wide_to_long(rf_wide)
    flow.add("Wide->Long (RFB setor-ano)", rf_long)

    # 2) Painel base + ABTD
    panel = build_base_panel(df_long, rf_long, flow)
    panel = compute_abtd(panel, flow, drop_lair_nonpos_for_abtd=drop_lair_nonpos_for_abtd)

    # 3) Winsor (nível) antes de defasar
    wins_cols_lvl = [
        "ABTD", "BTD", "ETRC",
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        "INOV_INC", "INOV_RAD", "INOV_AMBIENTAL",
        "FISC_AUT", "FISC_CRED",
    ]
    wins_cols_lvl = [c for c in wins_cols_lvl if c in panel.columns]
    panel = winsorize_df(panel, wins_cols_lvl, p=WINSOR_P)
    flow.add("Winsorização 1% (nível, antes das defasagens)", panel)

    # 4) Índice composto de inovação (opcional)
    innovation_target = innovation_target_name(INNOVATION_COMPOSITE_SUFFIX)
    innovation_logs: List[str] = []
    if USE_INNOVATION_COMPOSITE:
        method = str(INNOVATION_COMPOSITE_METHOD).strip().lower()
        if method != "entropy_topsis":
            raise ValueError("INNOVATION_COMPOSITE_METHOD inválido. Use 'entropy_topsis'.")

        panel, innovation_target, innovation_logs = build_innovation_composite_entropy_topsis(
            panel,
            components=INNOVATION_COMPONENTS,
            by=INNOVATION_COMPOSITE_BY,
            winsorize_components=INNOVATION_COMPOSITE_WINSORIZE,
            winsor_p=WINSOR_P,
            min_obs=5,
            suffix=INNOVATION_COMPOSITE_SUFFIX,
        )
        flow.add(f"Índice composto de inovação ({innovation_target})", panel)

    # 5) Proxies de agressividade tributária (t)
    # (a) ABTD
    panel["TAXAGG_ABTD"] = panel["ABTD"]

    # (b) ETRc: agressividade = 1 - ETRc
    if "ETRC" in panel.columns:
        panel["TAXAGG_ETRC"] = 1.0 - pd.to_numeric(panel["ETRC"], errors="coerce")
    else:
        panel["TAXAGG_ETRC"] = np.nan

    # (c) GAAP ETR (se existir no dados.xlsx)
    # nomes comuns: GAAPETR, ETR_GAAP, GAAP_ETR, ETRGAAP
    panel["GAAPETR_USED"] = np.nan
    gaap_candidates = ["GAAPETR", "ETR_GAAP", "GAAP_ETR", "ETRGAAP"]
    gaap_col = next((c for c in gaap_candidates if c in panel.columns), None)
    if gaap_col:
        panel["GAAPETR_USED"] = pd.to_numeric(panel[gaap_col], errors="coerce")
        panel["TAXAGG_GAAP"] = 1.0 - panel["GAAPETR_USED"]
    else:
        panel["TAXAGG_GAAP"] = np.nan

    # 6) Defasagem t-1: controles, fiscalização e proxies
    lag_cols = [
        # controles
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        # filtros de amostra para proxy ETR
        "LAIR", "ETRC", "GAAPETR_USED",
        # fiscalização
        "FISC_AUT", "FISC_CRED",
        # proxy ABTD
        "ABTD",
        # proxies alternativas
        "TAXAGG_ABTD", "TAXAGG_ETRC", "TAXAGG_GAAP",
    ]
    lag_cols = [c for c in lag_cols if c in panel.columns]
    panel = add_lags(panel, lag_cols, lag=1)

    # 7) Recorte final de anos (com L1, começa em ANO_INI+1)
    panel = panel[(panel["ANO"] >= (ANO_INI + 1)) & (panel["ANO"] <= ANO_FIM)].copy()
    flow.add("Corte final de anos (usável após defasagens)", panel)

    # 8) Salva painel final
    panel.to_parquet(OUTDIR / "panel_final.parquet", index=False)

    # 9) Estatística descritiva
    desc_vars = [
        "INOV_INC", "INOV_RAD", "INOV_AMBIENTAL",
        "ABTD", "BTD", "ETRC",
        "TAXAGG_ABTD", "TAXAGG_ETRC", "TAXAGG_GAAP",
        "SIZE", "ROA", "LEV", "GR", "PPE", "INTA",
        "FISC_AUT", "FISC_CRED",
    ]
    if USE_INNOVATION_COMPOSITE and innovation_target in panel.columns:
        desc_vars.append(innovation_target)
    desc = build_descriptives(panel, desc_vars)
    with pd.ExcelWriter(OUTDIR / "descriptive_stats.xlsx", engine="openpyxl") as w:
        for sheet, d in desc.items():
            d.to_excel(w, sheet_name=sheet, index=True if sheet == "overall" else False)

    # 10) Regressões FE
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
        # flags de execução
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
        run_sector_split_flag = bool(globals().get("RUN_SECTOR_SPLIT_REGRESSIONS", False))
        sector_split_min_obs = max(1, int(globals().get("SECTOR_SPLIT_MIN_OBS", 150)))
        sector_split_min_firms = max(1, int(globals().get("SECTOR_SPLIT_MIN_FIRMS", 20)))
        sector_split_min_years = max(1, int(globals().get("SECTOR_SPLIT_MIN_YEARS", 4)))
        sector_split_use_grouping = bool(globals().get("SECTOR_SPLIT_USE_GROUPING", True))
        sector_split_exclude_outros = bool(globals().get("SECTOR_SPLIT_EXCLUDE_OUTROS", True))
        output_regressions_by_sector = str(globals().get("OUTPUT_REGRESSIONS_BY_SECTOR", "regressions_by_sector.txt")).strip()
        output_regressions_final_fe_by_sector = str(
            globals().get("OUTPUT_REGRESSIONS_FINAL_FE_BY_SECTOR", "regressions_final_fe_by_sector.txt")
        ).strip()
        if not output_regressions_by_sector:
            output_regressions_by_sector = "regressions_by_sector.txt"
        if not output_regressions_final_fe_by_sector:
            output_regressions_final_fe_by_sector = "regressions_final_fe_by_sector.txt"
        use_innovation_composite = bool(globals().get("USE_INNOVATION_COMPOSITE", False))
        innovation_method = str(globals().get("INNOVATION_COMPOSITE_METHOD", "entropy_topsis")).strip().lower()
        innovation_by = str(globals().get("INNOVATION_COMPOSITE_BY", "year")).strip().lower()
        innovation_winsor = bool(globals().get("INNOVATION_COMPOSITE_WINSORIZE", True))
        innovation_components_cfg = globals().get("INNOVATION_COMPONENTS", ["INOV_INC", "INOV_RAD", "INOV_AMBIENTAL"])
        if isinstance(innovation_components_cfg, str):
            innovation_components = [clean_colname(innovation_components_cfg)]
        else:
            innovation_components = [clean_colname(c) for c in innovation_components_cfg]
        innovation_components = [c for c in innovation_components if c]

        # panel_l1 é o painel final (já com *_L1)
        panel_l1 = panel.copy()

        # dependentes
        y_list = ["INOV_INC", "INOV_RAD", "INOV_AMBIENTAL"]
        if use_innovation_composite and innovation_target in panel_l1.columns and panel_l1[innovation_target].notna().any():
            y_list.append(innovation_target)
        y_list = [y for y in y_list if y in panel_l1.columns]

        # moderadoras/controles selecionáveis
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

        # prioridade: TAX_AGG_PROXIES > RUN_ALL_PROXIES > TAX_AGG_PROXY
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
            f"- Período configurado: {cfg_start}-{cfg_end}\n",
            f"- Período efetivo para modelos com L1: {reg_start}-{cfg_end}\n",
            f"- Anos presentes no painel de regressão: {years_str}\n",
            f"- Proxies de agressividade utilizadas: {', '.join(proxies_to_run) if proxies_to_run else 'nenhuma'}\n",
            f"- Controles/moderadoras utilizados(as) (L1): {', '.join(moderators_l1) if moderators_l1 else 'nenhum(a)'}\n",
            f"- Filtro LAIR: ABTD={'on' if drop_lair_nonpos_for_abtd else 'off'}, ETR={'on' if drop_lair_nonpos_for_etr else 'off'}\n",
            f"- Moderação por setor (proxy x setor): {'sim' if run_sector_moderation else 'não'} (mín. obs/setor={sector_mod_min_obs})\n",
            f"- Regressões FE separadas por setor: {'sim' if run_sector_split_flag else 'não'}\n",
        ]
        if run_sector_split_flag:
            header_cfg.extend(
                [
                    f"- Split por setor: min obs={sector_split_min_obs}, min firms={sector_split_min_firms}, "
                    f"min years={sector_split_min_years}\n",
                    f"- Split por setor usa agrupamento={'sim' if sector_split_use_grouping else 'não'} "
                    f"(cutoff={sector_mod_min_obs}); exclui __outros__={'sim' if sector_split_exclude_outros else 'não'}\n",
                    f"- Arquivos split: {output_regressions_by_sector} | {output_regressions_final_fe_by_sector}\n",
                ]
            )
        if use_innovation_composite:
            header_cfg.extend(
                [
                    "- Índice composto de inovação: sim\n",
                    f"- Dependente composta: {innovation_target}\n",
                    f"- Método do índice composto: {innovation_method}\n",
                    f"- Componentes do índice composto: {', '.join(innovation_components) if innovation_components else 'nenhum'}\n",
                    f"- Cálculo do índice composto por: {innovation_by}\n",
                    f"- Winsorização dos componentes para índice: {'sim' if innovation_winsor else 'não'}\n",
                ]
            )
        reg_lines.extend(header_cfg)
        reg_fe_lines.extend(header_cfg)
        if drop_lair_nonpos_for_abtd:
            warn_abtd_lair = "Aviso: ABTD com LAIR>0 -> NOL removido do modelo normal do BTD por coerência.\n"
            reg_lines.append(warn_abtd_lair)
            reg_fe_lines.append(warn_abtd_lair)
        if panel_years_cfg:
            cfg_obs = f"- Checagem de período aprovada para ANO: {panel_years_cfg[0]}-{panel_years_cfg[-1]}\n"
            reg_lines.append(cfg_obs)
            reg_fe_lines.append(cfg_obs)
        if use_innovation_composite:
            reg_lines.append("\nResumo anual do índice composto (Entropia+TOPSIS)\n")
            reg_fe_lines.append("\nResumo anual do índice composto (Entropia+TOPSIS)\n")
            if innovation_logs:
                reg_lines.extend([f"{ln}\n" for ln in innovation_logs])
                reg_fe_lines.extend([f"{ln}\n" for ln in innovation_logs])
            else:
                reg_lines.append("[AVISO] Índice composto ativado, mas não houve logs de cálculo.\n")
                reg_fe_lines.append("[AVISO] Índice composto ativado, mas não houve logs de cálculo.\n")

        setor_source = (
            str(panel_l1["SETOR_SOURCE"].dropna().iloc[0])
            if ("SETOR_SOURCE" in panel_l1.columns and panel_l1["SETOR_SOURCE"].notna().any())
            else "SECTORRF/SETORRF/SECTORF"
        )
        setorial_note = f"Moderadora fiscal setorial via merge por {setor_source} + ANO.\n"
        reg_lines.append(setorial_note)
        reg_fe_lines.append(setorial_note)

        if run_sector_split_flag:
            run_sector_split_regressions(
                panel_l1=panel_l1,
                y_list=y_list,
                proxies_to_run=proxies_to_run,
                moderators_l1=moderators_l1,
                reg_start=reg_start,
                cfg_end=cfg_end,
                outdir=OUTDIR,
                sector_split_min_obs=sector_split_min_obs,
                sector_split_min_firms=sector_split_min_firms,
                sector_split_min_years=sector_split_min_years,
                sector_split_use_grouping=sector_split_use_grouping,
                sector_split_exclude_outros=sector_split_exclude_outros,
                sector_mod_min_obs=sector_mod_min_obs,
                drop_lair_nonpos_for_abtd=drop_lair_nonpos_for_abtd,
                drop_lair_nonpos_for_etr=drop_lair_nonpos_for_etr,
                output_regressions_by_sector=output_regressions_by_sector,
                output_regressions_final_fe_by_sector=output_regressions_final_fe_by_sector,
            )

        # função local: mapeia colunas da proxy
        def pick_proxy_cols(proxy: str):
            # retorna: (proxy_l1, proxy_l1_c, proxy_x_fisc_prefix)
            if proxy == "ABTD":
                # ABTD_L1
                return ("ABTD_L1", "ABTD_L1_C", "ABTD_L1_X_FISC")
            if proxy == "ETRC":
                # TAXAGG_ETRC_L1
                return ("TAXAGG_ETRC_L1", "TAXAGG_ETRC_L1_C", "TAXAGG_ETRC_L1_X_FISC")
            if proxy == "GAAPETR":
                # TAXAGG_GAAP_L1
                return ("TAXAGG_GAAP_L1", "TAXAGG_GAAP_L1_C", "TAXAGG_GAAP_L1_X_FISC")
            raise ValueError("TAX_AGG_PROXY inválida")

        # loop principal: y e proxies
        for y in y_list:
            reg_lines.append(f"\n=== DEPENDENTE: {y} ===\n")
            reg_fe_lines.append(f"\n=== DEPENDENTE: {y} ===\n")

            for proxy in proxies_to_run:
                reg_lines.append(f"\n--- PROXY: {proxy} ---\n")
                reg_fe_lines.append(f"\n--- PROXY: {proxy} ---\n")
                proxy_l1, proxy_l1_c, proxy_x_fisc = pick_proxy_cols(proxy)
                proxy_panel = panel_l1.copy()
                proxy_u = str(proxy).upper().strip()

                # proxies ETR: filtra LAIR_{t-1}>0 e ETR_{t-1} em (0,1)
                if proxy_u in {"ETRC", "GAAPETR"}:
                    if drop_lair_nonpos_for_etr and "LAIR_L1" in proxy_panel.columns:
                        lair_l1 = pd.to_numeric(proxy_panel["LAIR_L1"], errors="coerce")
                        proxy_panel = proxy_panel[lair_l1 > 0].copy()

                    if proxy_u == "ETRC":
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
                    if drop_lair_nonpos_for_etr:
                        etr_sample_msg = f"Amostra {proxy}: LAIR_L1>0 e ETR_L1 em (0,1). N={len(proxy_panel)}\n"
                    else:
                        etr_sample_msg = f"Amostra {proxy}: ETR_L1 em (0,1) (sem filtro LAIR_L1>0). N={len(proxy_panel)}\n"
                    reg_lines.append(etr_sample_msg)
                    reg_fe_lines.append(etr_sample_msg)
                elif proxy_u == "ABTD" and drop_lair_nonpos_for_abtd and "LAIR_L1" in proxy_panel.columns:
                    lair_l1 = pd.to_numeric(proxy_panel["LAIR_L1"], errors="coerce")
                    proxy_panel = proxy_panel[lair_l1 > 0].copy()
                    abtd_sample_msg = f"Amostra {proxy}: LAIR_L1>0 (robustez extrema). N={len(proxy_panel)}\n"
                    reg_lines.append(abtd_sample_msg)
                    reg_fe_lines.append(abtd_sample_msg)

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

                # 0) diagnóstico pooled sem FE (só sinal bruto)
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

                # 1) modelo principal (defasado): FE firma+ano
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

                # Driscoll-Kraay (robustez), se fit_fe_panel suportar cov_type/dk_bandwidth
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

                # 2) moderação contemporânea (FISC_t): proxy_L1 x FISC_t
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

                # 3) robustez pré-determinada (FISC_{t-1}): proxy_L1 x FISC_{t-1}
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

                # 4) moderação por setor: proxy_L1 x dummies de setor
                if run_sector_moderation and ("SETOR_KEY" in proxy_panel.columns) and (proxy_l1 in proxy_panel.columns):
                    tmp = proxy_panel.copy()
                    tmp[proxy_l1_c] = tmp[proxy_l1] - tmp[proxy_l1].mean()

                    setor_raw = tmp["SETOR_KEY"].fillna("__missing__").astype(str).str.strip()
                    setor_counts = setor_raw.value_counts(dropna=False)
                    setor_group = setor_raw.where(setor_raw.map(setor_counts) >= sector_mod_min_obs, "__outros__")
                    group_counts = setor_group.value_counts(dropna=False)

                    if len(group_counts) >= 2:
                        # baseline = setor mais frequente
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
                        # remove colunas sem variação para evitar problema de rank
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

    # 11) Fluxo amostral
    (OUTDIR / "sample_flow.txt").write_text("\n".join(flow.lines), encoding="utf-8")


if __name__ == "__main__":
    print("Iniciando...")
    main()
    print("Concluido.")
