
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import io, re, hashlib
from datetime import datetime, time
from typing import Dict, Tuple, Optional

# =============================
# Utils / cleaners
# =============================
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names: remove BOM, NBSP, tabs; collapse spaces; strip."""
    def _norm(s):
        if not isinstance(s, str):
            s = str(s)
        s = s.replace("\ufeff","").replace("\u00a0"," ").replace("\t"," ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df

def _series_str(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").str.strip()

# =============================
# CSV Loaders (DaaS & VFleets)
# =============================
def read_any_csv(uploaded_file):
    """Try multiple separators/encodings; return DF and read-info (incl. md5)."""
    name = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    data = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()
    trials = [
        dict(sep=";", encoding="latin-1", engine="python"),  # VFleets comum
        dict(sep=";", encoding="utf-8", engine="python"),
        dict(sep=",", encoding="utf-8"),
        dict(sep=",", encoding="latin-1"),
        dict(sep=None, encoding="utf-8", engine="python"),
        dict(sep=None, encoding="latin-1", engine="python"),
    ]
    last_err = None
    best = None
    for kw in trials:
        try:
            df = pd.read_csv(io.BytesIO(data), **kw)
            df = _clean_columns(df)
            cols = len(df.columns)
            if best is None or cols > len(best[0].columns) or (len(best[0].columns) == 1 and cols > 1):
                best = (df, kw)
        except Exception as e:
            last_err = e
    if best is not None:
        df, kw = best
        md5 = hashlib.md5(data).hexdigest()
        return df, {"file": name, **kw, "md5": md5, "size": len(data)}
    raise ValueError(f"Falha ao ler {name}: {last_err}")

def looks_like_vfleets(df: pd.DataFrame) -> bool:
    cols = set([c.strip().upper() for c in df.columns])
    required = {"DATA HORA", "LATITUDE", "LONGITUDE"}
    return required.issubset(cols) or ("MOTORISTA" in cols)

def filename_plate_guess(filename: str) -> Optional[str]:
    base = filename.split("/")[-1].split("\\")[-1]
    m = re.match(r"([A-Z0-9\-]{5,10})__.*\.csv$", base, flags=re.IGNORECASE)
    if m: return m.group(1).upper()
    m2 = re.match(r"([A-Z0-9\-]{5,10})\.csv$", base, flags=re.IGNORECASE)
    if m2: return m2.group(1).upper()
    return None

# =============================
# Normalizers
# =============================
def normalize_vfleets(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = _clean_columns(df)
    col = {c.strip().upper(): c for c in df.columns}
    def colget(*names):
        for n in names:
            if n in col: return col[n]
        return None

    c_dt = colget("DATA HORA")
    c_lat = colget("LATITUDE")
    c_lon = colget("LONGITUDE")
    c_plate = colget("PLACA")
    c_driver = colget("MOTORISTA", "CONDUTOR", "DRIVER")
    c_hodo = colget("HODOMETRO", "HODÔMETRO", "HODOMETER", "HODOMETRE")

    if not all([c_dt, c_lat, c_lon]):
        raise ValueError("CSV VFleets sem colunas essenciais: DATA HORA / LATITUDE / LONGITUDE")

    out = pd.DataFrame()

    if c_plate:
        plates = df[c_plate].astype(str).str.strip()
        plates = plates.str.replace(r"^-+\s*", "", regex=True)
        plates = plates.str.extract(r"([A-Z0-9\-]{5,10})", expand=False).fillna(plates)
        out["license_plate"] = plates.str.upper()
    else:
        g = filename_plate_guess(filename)
        out["license_plate"] = g if g else "PLACA_DESCONHECIDA"

    # VFleets -> timestamps locais (dd/mm/yyyy HH:MM:SS) em America/Sao_Paulo
    dt_series = pd.to_datetime(df[c_dt].astype(str).str.strip(), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    out["datetime"] = dt_series.dt.tz_localize("America/Sao_Paulo")

    def to_float_ptbr(s):
        return pd.to_numeric(s.astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    out["lat"] = to_float_ptbr(df[c_lat])
    out["lon"] = to_float_ptbr(df[c_lon])

    c_speed = colget("VELOCIDADE", "SPEED")
    if c_speed:
        sp = df[c_speed].astype(str).str.replace(",", ".").str.replace("km/h", "", regex=False)
        out["speed_kmh"] = pd.to_numeric(sp, errors="coerce")
    else:
        out["speed_kmh"] = np.nan

    c_ign = colget("IGNIÇÃO", "IGNICAO", "IGNITION")
    if c_ign:
        ign = df[c_ign].astype(str).str.lower().str.strip()
        out["ignition"] = ign.isin(["1","true","ligado","on","yes","y","t","sim"])
    else:
        out["ignition"] = (out["speed_kmh"].fillna(0) > 0)

    c_gps = colget("GPS", "GPS STATUS", "GPS_STATUS")
    if c_gps:
        gps = df[c_gps].astype(str).str.upper().str.strip()
        valid_vals = {"OK","VALIDO","VÁLIDO","COM SINAL","WITH_PRECISION","VALID","TRUE","1"}
        out["gps_status"] = gps.isin(valid_vals)
    else:
        out["gps_status"] = ~(((out["lat"]==0) & (out["lon"]==0)) | out["lat"].isna() | out["lon"].isna())

    out["uo"] = ""
    out["group"] = ""
    out["driver"] = df[c_driver].astype(str).str.strip() if c_driver else ""

    # HODÔMETRO (opcional): converte para quilômetros
    if c_hodo:
        hraw = to_float_ptbr(df[c_hodo])
        # Heurística de unidade: valores muito altos costumam estar em metros
        # Se percentil 95 > 100000 -> metros
        try:
            p95 = np.nanpercentile(hraw.values.astype(float), 95)
        except Exception:
            p95 = np.nan
        if pd.notna(p95) and p95 > 100000:
            out["hodometer_km"] = hraw / 1000.0
        else:
            out["hodometer_km"] = hraw  # assume km
    else:
        out["hodometer_km"] = np.nan

    out = out.dropna(subset=["datetime", "lat", "lon"])
    out = out.drop_duplicates(subset=["license_plate","datetime","lat","lon"])
    out = out.sort_values(["license_plate","datetime"]).reset_index(drop=True)
    return out

def normalize_daas(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(df)
    # Robust mapping (como o backup) + nomes exatos do DaaS
    candidates = {
        "license_plate": ["placa", "license_plate", "plate", "placa_veiculo", "vehicle_plate", "licensePlate", "prefix"],
        "uo": ["uo", "unidade_operacional", "client", "cliente", "uo_nome", "organizationUnitName"],
        "group": ["grupo", "grupo_veiculo", "vehicle_group", "group", "vehicleGroup"],
        "datetime": ["datetime", "dateTime", "data_hora", "datahora", "data_hora_gps", "timestamp"],
        "lat": ["lat", "latitude", "lat_gps"],
        "lon": ["lon", "longitude", "lng", "long", "lon_gps"],
        "speed_kmh": ["speed", "speed_kmh", "velocidade", "vel_kmh"],
        "ignition": ["ignition", "ign", "ignicao", "ignicao_ligada", "ignicao_bool"],
        "gps_status": ["gpsStatus", "gps_status", "status_gps", "gps_valid"],
        "driver": ["driver","motorista","condutor"],
        "hodometer": ["hodometer","odometer","odometro"],
    }
    lower = {c.lower(): c for c in df.columns}
    colmap = {}
    for canon, opts in candidates.items():
        for o in opts:
            if o.lower() in lower:
                colmap[canon] = lower[o.lower()]
                break
    if "license_plate" in colmap and colmap["license_plate"] == "prefix" and "licensePlate" in df.columns:
        colmap["license_plate"] = "licensePlate"

    out = df.rename(columns={v:k for k,v in colmap.items()})

    required = ["license_plate", "datetime", "lat", "lon"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        expected = {"licensePlate":"license_plate","dateTime":"datetime","latitude":"lat","longitude":"lon"}
        for raw, canon in expected.items():
            if raw in df.columns and canon not in out.columns:
                out[canon] = df[raw]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"CSV DaaS sem colunas obrigatórias após normalização: {missing}. Colunas lidas: {list(df.columns)}")

    # DaaS -> geralmente UTC; converto para America/Sao_Paulo
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True).dt.tz_convert("America/Sao_Paulo")

    out = out.dropna(subset=["datetime", "lat", "lon"])

    if "ignition" in out.columns:
        if out["ignition"].dtype == object:
            out["ignition"] = out["ignition"].astype(str).str.lower().isin(["1","true","ligado","on","yes","y","t","sim"])
        else:
            out["ignition"] = out["ignition"].astype(bool)
    else:
        out["ignition"] = False

    if "gps_status" in out.columns:
        if out["gps_status"].dtype == object:
            out["gps_status"] = out["gps_status"].astype(str).str.upper().isin(["WITH_PRECISION","VALID","OK","TRUE","1"])
        else:
            out["gps_status"] = out["gps_status"].astype(bool)
    else:
        out["gps_status"] = True

    if "speed_kmh" not in out.columns:
        out["speed_kmh"] = np.nan
    if "uo" not in out.columns:
        out["uo"] = "UO Desconhecida"
    if "group" not in out.columns:
        out["group"] = "Grupo Desconhecido"
    if "driver" not in out.columns:
        out["driver"] = ""

    # DaaS hodometer -> km
    if "hodometer" in out.columns:
        hod = pd.to_numeric(out["hodometer"], errors="coerce")
        try:
            p95 = np.nanpercentile(hod.values.astype(float), 95)
        except Exception:
            p95 = np.nan
        if pd.notna(p95) and p95 > 100000:
            out["hodometer_km"] = hod / 1000.0
        else:
            out["hodometer_km"] = hod
    else:
        out["hodometer_km"] = np.nan

    out = out.drop_duplicates(subset=["license_plate","datetime","lat","lon"])
    out = out.sort_values(["license_plate","datetime"]).reset_index(drop=True)
    return out

def normalize_any(df: pd.DataFrame, read_info: dict) -> pd.DataFrame:
    if looks_like_vfleets(df):
        return normalize_vfleets(df, read_info.get("file","(upload)"))
    else:
        return normalize_daas(df)

# =============================
# Core analytics (off-hour only)
# =============================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

def default_rules_noturno():
    # Seg–Sex: 18–23:59 e 00–06 | Sáb/Dom: 00–23:59
    return {
        0: [("18:00","23:59"), ("00:00","06:00")],
        1: [("18:00","23:59"), ("00:00","06:00")],
        2: [("18:00","23:59"), ("00:00","06:00")],
        3: [("18:00","23:59"), ("00:00","06:00")],
        4: [("18:00","23:59"), ("00:00","06:00")],
        5: [("00:00","23:59"), (None, None)],
        6: [("00:00","23:59"), (None, None)],
    }

def default_rules_comercial():
    return {
        0: [("07:00","19:00"), (None, None)],
        1: [("07:00","19:00"), (None, None)],
        2: [("07:00","19:00"), (None, None)],
        3: [("07:00","19:00"), (None, None)],
        4: [("07:00","19:00"), (None, None)],
        5: [("00:00","23:59"), (None, None)],
        6: [("00:00","23:59"), (None, None)],
    }

def default_rules_24x7():
    return {d: [("00:00","23:59"), (None, None)] for d in range(7)}

def parse_time_str(t):
    if t is None or (isinstance(t, str) and t.strip() == ""): return None
    h, m = map(int, t.split(":")); return time(hour=h, minute=m)

def in_windows(ts: pd.Timestamp, winrules: dict) -> bool:
    wd = ts.weekday()
    dayrules = winrules.get(wd, [(None,None), (None,None)])
    for start_str, end_str in dayrules:
        if start_str is None or end_str is None: continue
        s = parse_time_str(start_str); e = parse_time_str(end_str)
        t = ts.time()
        if s <= t <= e: return True
    return False

def build_effective_rules(uo_rules=None, group_rules=None, plate_rules=None):
    base = default_rules_noturno()  # preset padrão
    eff = {d: list(base[d]) for d in base}
    def apply(rdict):
        if not rdict: return
        for d in range(7):
            if d in rdict:
                pair = rdict[d]
                if len(pair) == 1: pair = [pair[0], (None,None)]
                eff[d] = pair
    apply(uo_rules); apply(group_rules); apply(plate_rules)
    return eff

def segment_and_points(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["license_plate","datetime"]).copy()
    df["dt_shift"] = df.groupby("license_plate")["datetime"].shift(1)
    df["lat_shift"] = df.groupby("license_plate")["lat"].shift(1)
    df["lon_shift"] = df.groupby("license_plate")["lon"].shift(1)
    df["time_gap_min"] = (df["datetime"] - df["dt_shift"]).dt.total_seconds().div(60.0)

    # Segmentos: gap > 10min ou virada de dia
    df["new_seg"] = df["time_gap_min"].isna() | (df["time_gap_min"] > 10) | (df["datetime"].dt.date != df["dt_shift"].dt.date)

    # Distância por Haversine (com salvaguardas)
    df["raw_step_km_hav"] = haversine_km(df["lat_shift"], df["lon_shift"], df["lat"], df["lon"])
    df.loc[df["new_seg"], "raw_step_km_hav"] = 0.0
    df.loc[~df["gps_status"].astype(bool), "raw_step_km_hav"] = 0.0
    df["implied_speed_kmh_hav"] = df["raw_step_km_hav"] / (df["time_gap_min"]/60.0)
    df["step_km_hav"] = df["raw_step_km_hav"].where(df["implied_speed_kmh_hav"] <= 160, 0.0)

    # Distância por Hodômetro (se disponível)
    if "hodometer_km" in df.columns:
        df["hodometer_prev"] = df.groupby("license_plate")["hodometer_km"].shift(1)
        df["hod_step"] = df["hodometer_km"] - df["hodometer_prev"]
        # Invalidar em início de segmento, negativos e saltos improváveis (> 200 km/h)
        df.loc[df["new_seg"] | df["hod_step"].isna() | (df["hod_step"] < 0), "hod_step"] = 0.0
        hours = df["time_gap_min"] / 60.0
        df["implied_speed_kmh_hod"] = df["hod_step"] / hours.replace(0, np.nan)
        df.loc[df["implied_speed_kmh_hod"] > 200, "hod_step"] = 0.0
        df["step_km"] = np.where(df["hod_step"].notna(), df["hod_step"], df["step_km_hav"])
    else:
        df["step_km"] = df["step_km_hav"]

    df["seg_id"] = df["new_seg"].groupby(df["license_plate"]).cumsum()
    seg = df.groupby(["license_plate","seg_id"]).agg(
        uo=("uo","first"),
        group=("group","first"),
        driver=("driver","first"),
        start=("datetime","first"),
        end=("datetime","last"),
        duration_min=("datetime", lambda s: (s.iloc[-1]-s.iloc[0]).total_seconds()/60.0),
        distance_km=("step_km","sum"),
        max_speed=("speed_kmh","max"),
        n_points=("datetime","size")
    ).reset_index()
    return seg, df

def classify_segments(seg: pd.DataFrame, points: pd.DataFrame, rules_by_scope: Dict) -> pd.DataFrame:
    results = []
    for _, row in seg.iterrows():
        plate = row["license_plate"]; uo = row["uo"]; group = row["group"]; seg_id = row["seg_id"]
        plate_rules = rules_by_scope.get(("PLATE", plate))
        group_rules = rules_by_scope.get(("GROUP", group)) if group else None
        uo_rules = rules_by_scope.get(("UO", uo)) if uo else None
        eff = build_effective_rules(uo_rules, group_rules, plate_rules)

        pts = points[(points["license_plate"]==plate) & (points["seg_id"]==seg_id)].copy()
        pts = pts.sort_values("datetime")
        pts["in_window"] = pts["datetime"].apply(lambda ts: in_windows(ts, eff))
        pts["dt_prev"] = pts["datetime"].shift(1)
        pts["gap_min"] = (pts["datetime"] - pts["dt_prev"]).dt.total_seconds().div(60.0).fillna(0)

        # Já temos step_km vindo do segment_and_points (hodômetro preferencial quando existir)
        off_mask = ~pts["in_window"]
        off_duration_min = pts.loc[off_mask, "gap_min"].sum()
        off_distance_km = pts.loc[off_mask, "step_km"].sum()
        off_start = pts.loc[off_mask, "datetime"].min()
        off_end = pts.loc[off_mask, "datetime"].max()

        ign_off = ((pts["ignition"]==True) & off_mask).any()
        pts["idle_flag"] = (pts["speed_kmh"].fillna(0) <= 3) & (pts["ignition"]==True) & off_mask
        idling_minutes = pts.loc[pts["idle_flag"], "gap_min"].sum()
        idling_off = idling_minutes >= 5.0

        movement_off = off_distance_km > 0.5
        reportable = (off_mask.any()) and (movement_off or ign_off or idling_off)

        reason = []
        if off_mask.any(): reason.append("movimentação do veículo fora da janela")
        if ign_off: reason.append("ignição ligada fora da janela")
        if idling_off: reason.append("motor ocioso fora da janela")

        severity = None
        if reportable:
            if (off_duration_min >= 30) or (off_distance_km > 5):
                severity = "GRAVE"
            elif (5 <= off_duration_min <= 15) or (off_distance_km <= 1):
                severity = "LEVE"
            else:
                severity = "MODERADA"

        if severity is not None:
            results.append({
                "uo": uo, "group": group, "driver": row.get("driver",""), "license_plate": plate,
                "start_off": off_start, "end_off": off_end,
                "off_duration_min": off_duration_min, "off_distance_km": off_distance_km,
                "max_speed": row["max_speed"], "reason": ", ".join(reason) if reason else "",
                "severity": severity, "seg_id": seg_id
            })
    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["uo","group","driver","license_plate","start_off"])
    return out

def build_rules_lookup(uos, groups, plates, rules_state):
    rules_lookup = {}
    for uo in [u for u in uos if str(u).strip()!=""]:
        rules_lookup[("UO", uo)] = rules_state["UO"].get(uo, default_rules_noturno())
    for g in [g for g in groups if str(g).strip()!=""]:
        rules_lookup[("GROUP", g)] = rules_state["GROUP"].get(g, None)
    for p in plates:
        rules_lookup[("PLATE", p)] = rules_state["PLATE"].get(p, default_rules_noturno())
    return rules_lookup

def ensure_sp_tz(dt_py):
    ts = pd.Timestamp(dt_py)
    if ts.tz is None:
        return ts.tz_localize("America/Sao_Paulo")
    return ts.tz_convert("America/Sao_Paulo")

def fmt_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%d/%m/%Y %H:%M:%S")

def fmt_duration_minsec(minutes_float: float) -> str:
    total_seconds = int(round(minutes_float * 60))
    mm, ss = divmod(total_seconds, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        return f"{hh:d}h {mm:02d}m {ss:02d}s"
    else:
        return f"{mm:d}m {ss:02d}s"

def fmt_distance_km_m(km: float) -> str:
    if km < 1.0:
        return f"{int(round(km*1000))} m"
    return f"{km:.1f} km"

# =============================
# UI
# =============================
st.set_page_config(page_title="Movimentação Indevida – DaaS & VFleets", layout="wide")
st.title("Análise de Movimentação Indevida (DaaS + VFleets)")
st.caption("Upload de **um ou mais** CSVs. Suporte a DaaS (,) e VFleets (;). Presets de regras: Comercial, Noturno e 24x7.")

uploads = st.file_uploader("CSV(s)", type=["csv"], accept_multiple_files=True)

# Pool de uploads persistente
if "data_pool" not in st.session_state:
    st.session_state["data_pool"] = []  # [{key,file,source,df,rows}]

colA, colB = st.columns(2)
with colA:
    keep_accum = st.checkbox("Acumular uploads nesta sessão (somar VFleets + DaaS)", value=True)
with colB:
    if st.button("Limpar todos os uploads da sessão"):
        st.session_state["data_pool"] = []

if uploads:
    new_entries = []
    for up in uploads:
        df_raw, info = read_any_csv(up)
        try:
            df_norm = normalize_any(df_raw, info)
            src = "VFleets" if looks_like_vfleets(df_raw) else "DaaS"
            key = (info.get("file"), info.get("md5"), str(info.get("sep")))
            new_entries.append({"key": key, "file": info.get("file"), "source": src, "df": df_norm, "rows": len(df_norm)})
        except Exception as e:
            st.warning(f"Falha ao normalizar {info['file']}: {e}")

    if keep_accum:
        keys_existing = {e["key"] for e in st.session_state["data_pool"]}
        for ent in new_entries:
            if ent["key"] not in keys_existing:
                st.session_state["data_pool"].append(ent)
    else:
        st.session_state["data_pool"] = new_entries

# Mesmo sem novos uploads, se houver pool acumulado, usa-o
if st.session_state["data_pool"]:
    with st.expander("Uploads acumulados na sessão", expanded=False):
        pool_df = pd.DataFrame([{"Arquivo": e["file"], "Fonte": e["source"], "Linhas": e["rows"]} for e in st.session_state["data_pool"]])
        st.caption(f"Arquivos no pool: {len(st.session_state['data_pool'])}")
        st.dataframe(pool_df, use_container_width=True, height=260)

    df = pd.concat([e["df"] for e in st.session_state["data_pool"]], ignore_index=True)

    # -------- Filtros com suporte a vazios (VFleets) --------
    # UO (inclui vazios)
    uos_vals = _series_str(df["uo"])
    has_uo_empty = (uos_vals == "").any()
    uos_all = sorted([u for u in uos_vals.unique() if u != ""])
    uo_options = ["(Todos)"] + (["(Sem UO)"] if has_uo_empty else []) + uos_all
    uo_sel = st.selectbox("UO", options=uo_options, index=0)
    if uo_sel == "(Todos)":
        df_uo = df
    elif uo_sel == "(Sem UO)":
        df_uo = df[ uos_vals == "" ]
    else:
        df_uo = df[ uos_vals == uo_sel ]

    # Grupo (inclui vazios)
    grp_vals = _series_str(df_uo["group"])
    has_grp_empty = (grp_vals == "").any()
    groups_all = sorted([g for g in grp_vals.unique() if g != ""])
    grp_options = (["(Sem grupo)"] if has_grp_empty else []) + groups_all
    default_groups = grp_options[:]  # tudo marcado (inclui "(Sem grupo)" se existir)
    groups_sel = st.multiselect("Grupo(s)", options=grp_options, default=default_groups)
    if groups_sel:
        sel_nonempty = [g for g in groups_sel if g != "(Sem grupo)"]
        mask = grp_vals.isin(sel_nonempty)
        if "(Sem grupo)" in groups_sel:
            mask = mask | (grp_vals == "")
        df_grp = df_uo[mask]
    else:
        df_grp = df_uo

    # Motorista (opcional; inclui vazios)
    drv_vals = _series_str(df_grp.get("driver", ""))
    has_drv = (drv_vals != "").any()
    if has_drv:
        has_drv_empty = (drv_vals == "").any()
        drv_all = sorted([d for d in drv_vals.unique() if d != ""])
        drv_options = (["(Sem motorista)"] if has_drv_empty else []) + drv_all
        default_drivers = drv_options[:]
        drivers_sel = st.multiselect("Motorista(s)", options=drv_options, default=default_drivers)
        if drivers_sel:
            sel_nonempty = [d for d in drivers_sel if d != "(Sem motorista)"]
            mask = drv_vals.isin(sel_nonempty)
            if "(Sem motorista)" in drivers_sel:
                mask = mask | (drv_vals == "")
            df_drv = df_grp[mask]
        else:
            df_drv = df_grp
    else:
        df_drv = df_grp

    # Placas
    plates_all = sorted(_series_str(df_drv["license_plate"]).unique().tolist())
    plates_sel = st.multiselect("Placas", options=plates_all, default=plates_all)
    df_scope = df_drv[df_drv["license_plate"].isin(plates_sel)] if plates_sel else df_drv

    # Período
    st.markdown("---")
    st.subheader("Período de análise (America/Sao_Paulo)")
    max_dt = df_scope["datetime"].max()
    default_start = max_dt - pd.Timedelta(hours=72) if pd.notna(max_dt) else None
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Data inicial", value=default_start.date() if default_start is not None else None, key="start_date")
        start_time = st.time_input("Hora inicial", value=default_start.time() if default_start is not None else time(0,0), key="start_time")
        start_dt = datetime.combine(start_date, start_time)
        start_sp = ensure_sp_tz(start_dt)
    with c2:
        end_date = st.date_input("Data final", value=max_dt.date() if pd.notna(max_dt) else None, key="end_date")
        end_time = st.time_input("Hora final", value=max_dt.time() if pd.notna(max_dt) else time(23,59), key="end_time")
        end_dt = datetime.combine(end_date, end_time)
        end_sp = ensure_sp_tz(end_dt)

    # Regras
    st.markdown("---")
    st.subheader("Regras de horário permitido")
    st.caption("Prioridade: Placa > Grupo > UO. Até 2 faixas por dia.")

    plates_all_for_rules = sorted(df["license_plate"].dropna().unique().tolist())
    uos_all_for_rules = sorted([u for u in df["uo"].dropna().unique().tolist() if str(u).strip()!=""])
    groups_all_for_rules = sorted([g for g in df["group"].dropna().unique().tolist() if str(g).strip()!=""])

    if "rules" not in st.session_state:
        st.session_state["rules"] = {"UO": {}, "GROUP": {}, "PLATE": {}}
    # garantir defaults
    for u in uos_all_for_rules:
        st.session_state["rules"]["UO"].setdefault(u, default_rules_noturno())
    for g in groups_all_for_rules:
        st.session_state["rules"]["GROUP"].setdefault(g, default_rules_noturno())
    for p in plates_all_for_rules:
        st.session_state["rules"]["PLATE"].setdefault(p, default_rules_noturno())

    cpr1, cpr2, cpr3 = st.columns(3)
    with cpr1:
        if st.button("Preset: Comercial (Seg–Sex 07:00–19:00)"):
            for scope in st.session_state["rules"].values():
                for key in list(scope.keys()):
                    scope[key] = default_rules_comercial()
    with cpr2:
        if st.button("Preset: Noturno (padrão)"):
            for scope in st.session_state["rules"].values():
                for key in list(scope.keys()):
                    scope[key] = default_rules_noturno()
    with cpr3:
        if st.button("Preset: 24x7 Permitido"):
            for scope in st.session_state["rules"].values():
                for key in list(scope.keys()):
                    scope[key] = default_rules_24x7()

    tabs_rules = st.tabs(["UO", "Grupo", "Placa", "Preview por dia"])

    def rules_editor(scope_name, scope_values):
        if not scope_values:
            st.info(f"Nenhum valor para {scope_name}.")
            return None
        selected = st.selectbox(f"Selecione {scope_name}", options=scope_values, key=f"sel_{scope_name}")
        rules = st.session_state["rules"][scope_name].setdefault(selected, default_rules_noturno())
        weekdays = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        for d in range(7):
            with st.expander(f"{weekdays[d]}"):
                c1, c2, c3, c4 = st.columns(4)
                s1 = c1.text_input(f"Início faixa 1 (HH:MM) dia {d}", value=rules[d][0][0] or "", key=f"{scope_name}_{selected}_d{d}_s1")
                e1 = c2.text_input(f"Fim faixa 1 (HH:MM) dia {d}", value=rules[d][0][1] or "", key=f"{scope_name}_{selected}_d{d}_e1")
                s2 = c3.text_input(f"Início faixa 2 (HH:MM) dia {d}", value=(rules[d][1][0] or ""), key=f"{scope_name}_{selected}_d{d}_s2")
                e2 = c4.text_input(f"Fim faixa 2 (HH:MM) dia {d}", value=(rules[d][1][1] or ""), key=f"{scope_name}_{selected}_d{d}_e2")
                st.session_state["rules"][scope_name][selected][d] = [(s1 or None, e1 or None), (s2 or None, e2 or None)]
        st.button("Resetar este item para preset Noturno", on_click=lambda: st.session_state["rules"][scope_name].update({selected: default_rules_noturno()}), key=f"reset_{scope_name}")
        return selected

    with tabs_rules[0]:
        if uos_all_for_rules: rules_editor("UO", uos_all_for_rules)
        else: st.info("Sem UO (VFleets).")
    with tabs_rules[1]:
        if groups_all_for_rules: rules_editor("GROUP", groups_all_for_rules)
        else: st.info("Sem Grupo (VFleets).")
    with tabs_rules[2]:
        rules_editor("PLATE", plates_all_for_rules)
    with tabs_rules[3]:
        st.caption("Escolha placa e dia da semana para ver as janelas efetivas (prioridade Placa > Grupo > UO).")
        if plates_all_for_rules:
            prev_plate = st.selectbox("Placa", options=plates_all_for_rules, key="prev_plate")
            prev_wd = st.selectbox("Dia da semana", options=[0,1,2,3,4,5,6], format_func=lambda d: ['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'][d])
            # pega grupo/uo amostrados para essa placa
            sample_group = None
            vals = df[df["license_plate"]==prev_plate]["group"].dropna().unique().tolist()
            sample_group = vals[0] if vals else None
            vals2 = df[df["license_plate"]==prev_plate]["uo"].dropna().unique().tolist()
            sample_uo = vals2[0] if vals2 else None
            plate_rules = st.session_state["rules"]["PLATE"].get(prev_plate)
            group_rules = st.session_state["rules"]["GROUP"].get(sample_group) if sample_group else None
            uo_rules = st.session_state["rules"]["UO"].get(sample_uo) if sample_uo else None
            eff = build_effective_rules(uo_rules, group_rules, plate_rules)
            windows = eff.get(prev_wd, [(None,None),(None,None)])
            st.info(f"Janelas efetivas: {windows[0]} e {windows[1]}")

    # Executar Análise
    if st.button("Executar Análise"):
        mask = (df_scope["datetime"] >= start_sp) & (df_scope["datetime"] <= end_sp)
        df_filtered = df_scope.loc[mask].copy()

        seg_all, points = segment_and_points(df_filtered)
        rules_lookup = build_rules_lookup(uos_all_for_rules, groups_all_for_rules, plates_all_for_rules, st.session_state["rules"])
        segments = classify_segments(seg_all, points, rules_lookup)

        st.session_state["segments_result"] = segments
        st.session_state["points_scoped"] = points
        st.session_state["rules_lookup"] = rules_lookup
        uo_label = uo_sel if uo_sel!="(Todos)" else "Todos"
        st.session_state["period_str"] = f"{start_sp.strftime('%d/%b/%Y %H:%M')} – {end_sp.strftime('%d/%b/%Y %H:%M')}"
        st.session_state["uo_sel"] = uo_label

    st.markdown("---")

    # Resultados
    tabs = st.tabs(["KPIs & Tabela", "Mapa (mini por segmento)", "Heatmap (1 km ~)", "Recorrência 7/30 dias"])

    with tabs[0]:
        segments = st.session_state.get("segments_result")
        if segments is None or segments.empty:
            st.info("Execute a análise para visualizar os resultados.")
        else:
            df_disp = segments.copy()
            df_disp["Início (off)"] = df_disp["start_off"].apply(fmt_dt)
            df_disp["Fim (off)"] = df_disp["end_off"].apply(fmt_dt)
            df_disp["Duração (off)"] = df_disp["off_duration_min"].apply(fmt_duration_minsec)
            df_disp["Distância (off)"] = df_disp["off_distance_km"].apply(fmt_distance_km_m)
            df_disp["Velocidade máx. (km/h)"] = df_disp["max_speed"].round(0)
            df_disp["Tipo / Motivo"] = df_disp["reason"].replace("", "movimentação do veículo fora da janela")
            df_disp = df_disp.rename(columns={
                "license_plate":"Placa", "uo":"UO", "group":"Grupo", "driver":"Motorista", "severity":"Severidade"
            })[["UO","Grupo","Motorista","Placa","Início (off)","Fim (off)","Duração (off)","Distância (off)","Velocidade máx. (km/h)","Tipo / Motivo","Severidade"]]

            total_segments = len(df_disp)
            total_km = segments["off_distance_km"].sum()
            plates_occ = segments["license_plate"].nunique()
            ign_count = segments["reason"].str.contains("ignição ligada", na=False).sum()
            idle_count = segments["reason"].str.contains("motor ocioso", na=False).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("# segmentos", total_segments)
            c2.metric("Distância off-hour", fmt_distance_km_m(total_km))
            c3.metric("Placas com ocorrência", plates_occ)
            c4.metric("Eventos (ignição / ocioso)", f"{int(ign_count)}/{int(idle_count)}")

            st.dataframe(df_disp, height=520, use_container_width=True)

            by_km = segments.groupby("license_plate")["off_distance_km"].sum().sort_values(ascending=False).round(1)
            plates_list = ", ".join([f"{p} ({km:.1f} km)" if km>=1 else f"{p} ({int(km*1000)} m)" for p, km in by_km.items()][:20])
            summary_text = f"""[Cliente/UO: {st.session_state.get('uo_sel')}] Período: {st.session_state.get('period_str')} (timezone America/Sao_Paulo)
Total off-hour: {total_segments} segmentos, {total_km:.1f} km, {plates_occ} placas.
Lista de placas (km): {plates_list if plates_list else "-"}
Destaques: {int(ign_count)} segmentos com ignição ligada fora de janela; {int(idle_count)} com motor ocioso fora de janela.
"""
            st.subheader("Resumo copiável")
            st.code(summary_text, language="markdown")

            csv = segments.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV de segmentos off-hour", data=csv, file_name="offhour_segments.csv", mime="text/csv")

    with tabs[1]:
        segments = st.session_state.get("segments_result")
        points = st.session_state.get("points_scoped")
        rules_lookup = st.session_state.get("rules_lookup")
        if segments is None or segments.empty:
            st.info("Execute a análise para visualizar o mapa.")
        else:
            options = segments.apply(lambda r: f"{r['license_plate']} | {r['start_off'].strftime('%d/%m %H:%M')} → {r['end_off'].strftime('%d/%m %H:%M')} | {r['severity']}", axis=1).tolist()
            idx = st.selectbox("Selecione um segmento", options=list(range(len(options))), format_func=lambda i: options[i])
            sel = segments.iloc[idx]
            seg_points = points[(points["license_plate"]==sel["license_plate"]) & (points["seg_id"]==sel["seg_id"])].copy()
            seg_points = seg_points[(seg_points["lat"].between(-90, 90)) & (seg_points["lon"].between(-180, 180)) & (seg_points["lat"]!=0) & (seg_points["lon"]!=0)]
            seg_points = seg_points.sort_values("datetime")
            if seg_points.empty:
                st.warning("Sem pontos válidos para este segmento.")
            else:
                plate = sel["license_plate"]; g = sel.get("group",""); u = sel.get("uo","")
                plate_rules = rules_lookup.get(("PLATE", plate))
                group_rules = rules_lookup.get(("GROUP", g)) if g else None
                uo_rules = rules_lookup.get(("UO", u)) if u else None
                eff = build_effective_rules(uo_rules, group_rules, plate_rules)
                seg_points["in_window"] = seg_points["datetime"].apply(lambda ts: in_windows(ts, eff))

                path = seg_points.apply(lambda r: [float(r["lon"]), float(r["lat"])], axis=1).tolist()
                center_lat = float(seg_points["lat"].mean())
                center_lon = float(seg_points["lon"].mean())

                off_pts = seg_points[~seg_points["in_window"]][["lon","lat"]].rename(columns={"lon":"x","lat":"y"})
                on_pts = seg_points[seg_points["in_window"]][["lon","lat"]].rename(columns={"lon":"x","lat":"y"})

                layers = [
                    pdk.Layer("PathLayer", data=[{"path": path, "name": "trajeto"}], get_path="path", get_width=4, get_color=[120, 120, 255, 200]),
                    pdk.Layer("ScatterplotLayer", data=on_pts.rename(columns={"x":"lon","y":"lat"}), get_position='[lon, lat]', get_radius=40, get_fill_color=[0, 200, 0, 180], pickable=True),
                    pdk.Layer("ScatterplotLayer", data=off_pts.rename(columns={"x":"lon","y":"lat"}), get_position='[lon, lat]', get_radius=40, get_fill_color=[230, 50, 50, 200], pickable=True),
                ]
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, bearing=0, pitch=0)
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "Ponto"}))

    with tabs[2]:
        segments = st.session_state.get("segments_result")
        points = st.session_state.get("points_scoped")
        if segments is None or segments.empty:
            st.info("Execute a análise para visualizar o heatmap.")
        else:
            off = segments[["license_plate","seg_id"]]
            pts = points.merge(off, on=["license_plate","seg_id"], how="inner").copy()
            pts = pts[(pts["lat"].between(-90, 90)) & (pts["lon"].between(-180, 180)) & (pts["lat"]!=0) & (pts["lon"]!=0)]
            pts["lat_bin"] = (pts["lat"] / 0.01).round().astype(int) * 0.01
            pts["lon_bin"] = (pts["lon"] / 0.01).round().astype(int) * 0.01
            heat = pts.groupby(["lat_bin","lon_bin"]).size().reset_index(name="weight")
            if heat.empty:
                st.info("Sem dados suficientes para heatmap.")
            else:
                heat["lat"] = heat["lat_bin"].astype(float)
                heat["lon"] = heat["lon_bin"].astype(float)
                view_state = pdk.ViewState(latitude=float(heat["lat"].mean()), longitude=float(heat["lon"].mean()), zoom=6)
                layer = pdk.Layer("HeatmapLayer", data=heat, get_position='[lon, lat]', get_weight='weight', aggregation=pdk.types.String("SUM"), radius_pixels=40)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    with tabs[3]:
        segments = st.session_state.get("segments_result")
        if segments is None or segments.empty:
            st.info("Execute a análise para ver recorrência.")
        else:
            seg = segments.copy()
            seg["day"] = seg["start_off"].dt.date
            end_date = seg["end_off"].max().date()
            last7 = pd.date_range(end=end_date, periods=7).date
            last30 = pd.date_range(end=end_date, periods=30).date
            rec7 = (seg[seg["day"].isin(last7)].groupby("license_plate")["day"].nunique().reset_index(name="Dias com ocorrência (7d)"))
            rec30 = (seg[seg["day"].isin(last30)].groupby("license_plate")["day"].nunique().reset_index(name="Dias com ocorrência (30d)"))
            rec = rec7.merge(rec30, on="license_plate", how="outer").fillna(0)
            rec["% recorrência (7d)"] = (rec["Dias com ocorrência (7d)"] / 7.0 * 100).round(1)
            rec["% recorrência (30d)"] = (rec["Dias com ocorrência (30d)"] / 30.0 * 100).round(1)
            rec = rec.rename(columns={"license_plate":"Placa"})
            rec = rec.sort_values(["Dias com ocorrência (30d)","Dias com ocorrência (7d)"], ascending=False)
            st.dataframe(rec, use_container_width=True, height=480)

else:
    st.info("Faça upload dos CSVs para iniciar.")
