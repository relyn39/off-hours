
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, time, timedelta
from typing import Dict, Tuple

# ==============================
# Helpers
# ==============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    for canon, src in list(colmap.items()):
        if canon != src:
            df.rename(columns={src: canon}, inplace=True)
    required = ["license_plate", "datetime", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes após normalização: {missing}. Colunas: {list(df.columns)}")
    # Converte para America/Sao_Paulo
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert("America/Sao_Paulo")
    df = df.dropna(subset=["datetime", "lat", "lon"])
    if "ignition" in df.columns:
        if df["ignition"].dtype == object:
            df["ignition"] = df["ignition"].astype(str).str.lower().isin(["1","true","ligado","on","yes","y","t"])
        else:
            df["ignition"] = df["ignition"].astype(bool)
    else:
        df["ignition"] = False
    if "gps_status" in df.columns:
        if df["gps_status"].dtype == object:
            df["gps_status"] = df["gps_status"].astype(str).str.upper().isin(["WITH_PRECISION","VALID","OK","TRUE","1"])
        else:
            df["gps_status"] = df["gps_status"].astype(bool)
    else:
        df["gps_status"] = True
    if "speed_kmh" not in df.columns:
        df["speed_kmh"] = np.nan
    if "uo" not in df.columns:
        df["uo"] = "UO Desconhecida"
    if "group" not in df.columns:
        df["group"] = "Grupo Desconhecido"
    df = df.drop_duplicates(subset=["license_plate","datetime","lat","lon"])
    df = df.sort_values(["license_plate","datetime"]).reset_index(drop=True)
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

# ----- Default rules (pedido) -----
def default_week_rules():
    # Seg(0)–Sex(4): 18:00–23:59 e 00:00–06:00 | Sáb(5), Dom(6): 00:00–23:59
    return {
        0: [("18:00","23:59"), ("00:00","06:00")],
        1: [("18:00","23:59"), ("00:00","06:00")],
        2: [("18:00","23:59"), ("00:00","06:00")],
        3: [("18:00","23:59"), ("00:00","06:00")],
        4: [("18:00","23:59"), ("00:00","06:00")],
        5: [("00:00","23:59"), (None, None)],
        6: [("00:00","23:59"), (None, None)],
    }

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
    base = default_week_rules(); eff = {d: list(base[d]) for d in base}
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
    df["new_seg"] = df["time_gap_min"].isna() | (df["time_gap_min"] > 10) | (df["datetime"].dt.date != df["dt_shift"].dt.date)
    df["raw_step_km"] = haversine_km(df["lat_shift"], df["lon_shift"], df["lat"], df["lon"])
    df.loc[df["new_seg"], "raw_step_km"] = 0.0
    df.loc[~df["gps_status"].astype(bool), "raw_step_km"] = 0.0
    df["implied_speed_kmh"] = df["raw_step_km"] / (df["time_gap_min"]/60.0)
    df["step_km"] = df["raw_step_km"].where(df["implied_speed_kmh"] <= 160, 0.0)
    df["seg_id"] = df["new_seg"].groupby(df["license_plate"]).cumsum()
    seg = df.groupby(["license_plate","seg_id"]).agg(
        uo=("uo","first"),
        group=("group","first"),
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
        group_rules = rules_by_scope.get(("GROUP", group))
        uo_rules = rules_by_scope.get(("UO", uo))
        eff = build_effective_rules(uo_rules, group_rules, plate_rules)
        pts = points[(points["license_plate"]==plate) & (points["seg_id"]==seg_id)].copy()
        pts["in_window"] = pts["datetime"].apply(lambda ts: in_windows(ts, eff))
        off_any = (~pts["in_window"]).any()
        ign_off = ((pts["ignition"]==True) & (~pts["in_window"])).any()
        pts["idle_flag"] = (pts["speed_kmh"].fillna(0) <= 3) & (pts["ignition"]==True) & (~pts["in_window"])
        pts["dt_shift2"] = pts["datetime"].shift(1)
        pts["gap_min2"] = (pts["datetime"] - pts["dt_shift2"]).dt.total_seconds().div(60.0).fillna(0)
        idling_minutes = pts.loc[pts["idle_flag"], "gap_min2"].sum()
        idling_off = idling_minutes >= 5.0
        movement = row["distance_km"] > 0.5
        reportable = off_any and (movement or ign_off or idling_off)
        reason = []
        if off_any: reason.append("movimentação do veículo fora da janela")
        if ign_off: reason.append("ignição ligada fora da janela")
        if idling_off: reason.append("motor ocioso fora da janela")
        severity = None
        if reportable:
            if (row["duration_min"] >= 30) or (row["distance_km"] > 5):
                severity = "GRAVE"
            elif (5 <= row["duration_min"] <= 15) or (row["distance_km"] <= 1):
                severity = "LEVE"
            else:
                severity = "MODERADA"
        results.append({
            "uo": uo, "group": group, "license_plate": plate,
            "start": row["start"], "end": row["end"],
            "duration_min": row["duration_min"], "distance_km": row["distance_km"],
            "max_speed": row["max_speed"], "reason": ", ".join(reason) if reason else "",
            "severity": severity, "seg_id": seg_id
        })
    out = pd.DataFrame(results)
    out = out.dropna(subset=["severity"]).sort_values(["uo","group","license_plate","start"])
    return out

def build_rules_lookup(uos, groups, plates, rules_state):
    rules_lookup = {}
    for uo in uos:
        rules_lookup[("UO", uo)] = rules_state["UO"].get(uo, default_week_rules())
    for g in groups:
        rules_lookup[("GROUP", g)] = rules_state["GROUP"].get(g, None)
    for p in plates:
        rules_lookup[("PLATE", p)] = rules_state["PLATE"].get(p, None)
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

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Análise de Movimentação Indevida", layout="wide")
st.title("Análise de Movimentação Indevida (DaaS – Consulta de Posições)")
st.caption("Faça upload de **um ou mais** CSVs de posições (schema DaaS). Não é preciso colocar arquivos em pasta do projeto.")

uploaded = st.file_uploader("CSV(s) de posições", type=["csv"], accept_multiple_files=True)

if uploaded:
    # Load & normalize
    dfs = [pd.read_csv(f) for f in uploaded]
    raw = pd.concat(dfs, ignore_index=True)
    df = normalize_columns(raw)

    # Filtros: UO → Grupo → Placas
    uos = sorted(df["uo"].dropna().unique().tolist())
    uo_sel = st.selectbox("UO", options=uos, index=0 if uos else None)
    df_uo = df[df["uo"] == uo_sel] if uo_sel else df

    groups = sorted(df_uo["group"].dropna().unique().tolist())
    groups_sel = st.multiselect("Grupo(s)", options=groups, default=groups)

    df_grp = df_uo[df_uo["group"].isin(groups_sel)] if groups_sel else df_uo
    plates = sorted(df_grp["license_plate"].dropna().unique().tolist())
    plates_sel = st.multiselect("Placas", options=plates, default=plates)

    df_scope = df_grp[df_grp["license_plate"].isin(plates_sel)] if plates_sel else df_grp

    # Período (date + time) com timezone-aware
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

    # Regras (Editor)
    st.markdown("---")
    st.subheader("Regras de horário permitido")
    st.caption("Prioridade: Placa > Grupo > UO. Até 2 faixas por dia. Horários no fuso America/Sao_Paulo.")
    tabs_rules = st.tabs(["UO", "Grupo", "Placa"])

    def init_rules_for(scope_values):
        return {sv: default_week_rules() for sv in scope_values}

    if "rules" not in st.session_state:
        st.session_state["rules"] = {
            "UO": init_rules_for(uos),
            "GROUP": init_rules_for(groups),
            "PLATE": init_rules_for(plates),
        }

    def rules_editor(scope_name, scope_values):
        if not scope_values:
            st.info(f"Nenhum valor para {scope_name}.")
            return None
        selected = st.selectbox(f"Selecione {scope_name}", options=scope_values, key=f"sel_{scope_name}")
        rules = st.session_state["rules"][scope_name].setdefault(selected, default_week_rules())
        weekdays = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        for d in range(7):
            with st.expander(f"{weekdays[d]}"):
                c1, c2, c3, c4 = st.columns(4)
                s1 = c1.text_input(f"Início faixa 1 (HH:MM) dia {d}", value=rules[d][0][0] or "", key=f"{scope_name}_{selected}_d{d}_s1")
                e1 = c2.text_input(f"Fim faixa 1 (HH:MM) dia {d}", value=rules[d][0][1] or "", key=f"{scope_name}_{selected}_d{d}_e1")
                s2 = c3.text_input(f"Início faixa 2 (HH:MM) dia {d}", value=(rules[d][1][0] or ""), key=f"{scope_name}_{selected}_d{d}_s2")
                e2 = c4.text_input(f"Fim faixa 2 (HH:MM) dia {d}", value=(rules[d][1][1] or ""), key=f"{scope_name}_{selected}_d{d}_e2")
                st.session_state["rules"][scope_name][selected][d] = [(s1 or None, e1 or None), (s2 or None, e2 or None)]
        st.button("Resetar para padrão solicitado", on_click=lambda: st.session_state["rules"][scope_name].update({selected: default_week_rules()}), key=f"reset_{scope_name}")
        return selected

    with tabs_rules[0]:
        rules_editor("UO", uos)
    with tabs_rules[1]:
        rules_editor("GROUP", groups)
    with tabs_rules[2]:
        rules_editor("PLATE", plates)

    # Executar Análise
    if st.button("Executar Análise"):
        mask = (df_scope["datetime"] >= start_sp) & (df_scope["datetime"] <= end_sp)
        df_filtered = df_scope.loc[mask].copy()

        seg_all, points = segment_and_points(df_filtered)
        rules_lookup = build_rules_lookup(uos, groups, plates, st.session_state["rules"])
        segments = classify_segments(seg_all, points, rules_lookup)

        st.session_state["segments_result"] = segments
        st.session_state["points_scoped"] = points
        st.session_state["period_str"] = f"{start_sp.strftime('%d/%b/%Y %H:%M')} – {end_sp.strftime('%d/%b/%Y %H:%M')}"
        st.session_state["uo_sel"] = uo_sel

    st.markdown("---")

    # Resultados
    tabs = st.tabs(["KPIs & Tabela", "Mapa (mini por segmento)", "Heatmap (1 km ~)", "Recorrência 7/30 dias"])

    with tabs[0]:
        segments = st.session_state.get("segments_result")
        if segments is None or segments.empty:
            st.info("Execute a análise para visualizar os resultados.")
        else:
            # Tabela mais amigável
            df_disp = segments.copy()
            df_disp["Início"] = df_disp["start"].apply(fmt_dt)
            df_disp["Fim"] = df_disp["end"].apply(fmt_dt)
            df_disp["Duração"] = df_disp["duration_min"].apply(fmt_duration_minsec)
            df_disp["Distância"] = df_disp["distance_km"].apply(fmt_distance_km_m)
            df_disp["Velocidade máx. (km/h)"] = df_disp["max_speed"].round(0)
            df_disp["Tipo / Motivo"] = df_disp["reason"].replace("", "movimentação do veículo fora da janela")
            df_disp = df_disp.rename(columns={
                "license_plate":"Placa", "uo":"UO", "group":"Grupo", "severity":"Severidade"
            })[["UO","Grupo","Placa","Início","Fim","Duração","Distância","Velocidade máx. (km/h)","Tipo / Motivo","Severidade"]]

            # KPIs
            total_segments = len(df_disp)
            total_km = segments["distance_km"].sum()
            plates_occ = segments["license_plate"].nunique()
            ign_count = segments["reason"].str.contains("ignição ligada", na=False).sum()
            idle_count = segments["reason"].str.contains("motor ocioso", na=False).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("# segmentos", total_segments)
            c2.metric("Distância off-hour", fmt_distance_km_m(total_km))
            c3.metric("Placas com ocorrência", plates_occ)
            c4.metric("Eventos (ignição / ocioso)", f"{int(ign_count)}/{int(idle_count)}")

            st.dataframe(df_disp, height=520, use_container_width=True)

            # Resumo copiável
            by_km = segments.groupby("license_plate")["distance_km"].sum().sort_values(ascending=False).round(1)
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
        if segments is None or segments.empty:
            st.info("Execute a análise para visualizar o mapa.")
        else:
            options = segments.apply(lambda r: f"{r['license_plate']} | {r['start'].strftime('%d/%m %H:%M')} → {r['end'].strftime('%d/%m %H:%M')} | {r['severity']}", axis=1).tolist()
            idx = st.selectbox("Selecione um segmento", options=list(range(len(options))), format_func=lambda i: options[i])
            sel = segments.iloc[idx]
            seg_points = points[(points["license_plate"]==sel["license_plate"]) & (points["seg_id"]==sel["seg_id"])].copy()
            # filtros de coordenadas válidas
            seg_points = seg_points[(seg_points["lat"].between(-90, 90)) & (seg_points["lon"].between(-180, 180)) & (seg_points["lat"]!=0) & (seg_points["lon"]!=0)]
            seg_points = seg_points.sort_values("datetime")
            if seg_points.empty:
                st.warning("Sem pontos válidos para este segmento.")
            else:
                path = seg_points.apply(lambda r: [float(r["lon"]), float(r["lat"])], axis=1).tolist()
                center_lat = float(seg_points["lat"].mean())
                center_lon = float(seg_points["lon"].mean())

                layers = [
                    pdk.Layer(
                        "PathLayer",
                        data=[{"path": path, "name": "trajeto"}],
                        get_path="path",
                        get_width=5,
                        get_color=[0, 180, 255, 200],
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"lon": path[0][0], "lat": path[0][1]}],
                        get_position='[lon, lat]',
                        get_radius=60,
                        get_fill_color=[0, 255, 0, 200],
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"lon": path[-1][0], "lat": path[-1][1]}],
                        get_position='[lon, lat]',
                        get_radius=60,
                        get_fill_color=[255, 0, 0, 200],
                    )
                ]
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, bearing=0, pitch=0)
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"}))

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
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=heat,
                    get_position='[lon, lat]',
                    get_weight='weight',
                    aggregation=pdk.types.String("SUM"),
                    radius_pixels=40,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    with tabs[3]:
        segments = st.session_state.get("segments_result")
        if segments is None or segments.empty:
            st.info("Execute a análise para ver recorrência.")
        else:
            seg = segments.copy()
            seg["day"] = seg["start"].dt.date
            end_date = seg["end"].max().date()
            last7 = pd.date_range(end=end_date, periods=7).date
            last30 = pd.date_range(end=end_date, periods=30).date

            rec7 = (seg[seg["day"].isin(last7)]
                    .groupby("license_plate")["day"].nunique()
                    .reset_index(name="Dias com ocorrência (7d)"))
            rec30 = (seg[seg["day"].isin(last30)]
                    .groupby("license_plate")["day"].nunique()
                    .reset_index(name="Dias com ocorrência (30d)"))
            rec = rec7.merge(rec30, on="license_plate", how="outer").fillna(0)
            rec["% recorrência (7d)"] = (rec["Dias com ocorrência (7d)"] / 7.0 * 100).round(1)
            rec["% recorrência (30d)"] = (rec["Dias com ocorrência (30d)"] / 30.0 * 100).round(1)
            rec = rec.rename(columns={"license_plate":"Placa"})
            rec = rec.sort_values(["Dias com ocorrência (30d)","Dias com ocorrência (7d)"], ascending=False)

            st.dataframe(rec, use_container_width=True, height=480)

else:
    st.info("Faça upload dos CSVs para iniciar.")
