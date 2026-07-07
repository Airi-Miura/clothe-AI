from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from clothing_classifier import (
    CLOTHING_CATEGORY_CANDIDATES,
    classify_clothing_image,
    load_clip_classifier,
)
from closet_db import (
    CLOSET_DB_PATH,
    CLOSET_IMAGE_DIR,
    delete_closet_item,
    load_closet_items,
    save_closet_items,
    save_uploaded_closet_image,
)

from model import (
    ACTIVITY_OPTIONS,
    CLO_DB,
    DELTA_T_COEFFS,
    DELTA_T_MAX_MINUTES,
    ENVIRONMENT_OPTIONS,
    FEEDBACK_OPTIONS,
    FEEDBACK_SCORE_MAP,
    SEASON_PRESETS,
    THERMAL_TYPE_OPTIONS,
    TRANSPORT_OPTIONS,
    aggregate_minutes_to_segments,
    apply_cz_from_pmv,
    apply_first_order_lag_to_minutes,
    apply_open_meteo_to_outdoor_segments,
    build_daily_segments_from_trips,
    calculate_pmv_series,
    calculate_total_clo,
    compare_clo_recommendation,
    compare_pmv_transition,
    expand_segments_to_time_level,
    get_initial_personal_clo_offset,
    get_recommendation_message,
    infer_environment_from_place,
    summarize_results,
    update_personal_clo_offset,
    validate_trip_df,
)

st.set_page_config(
    page_title="環境温度推定 & CZ判定アプリ",
    page_icon="🌡️",
    layout="wide",
)

SEASON_LABELS = {
    "spring": "春",
    "summer": "夏",
    "autumn": "秋",
    "winter": "冬",
}

DEFAULT_PLACES = {
    "自宅": {"lat": 35.6580, "lon": 139.7016, "default_env": "屋内"},
    "学校": {"lat": 35.6550, "lon": 139.7957, "default_env": "屋内"},
    "渋谷": {"lat": 35.6580, "lon": 139.7016, "default_env": "屋外"},
    "豊洲": {"lat": 35.6550, "lon": 139.7957, "default_env": "屋内"},
    "大宮": {"lat": 35.9062, "lon": 139.6238, "default_env": "屋外"},
}

DEFAULT_TRIPS = [
    {
        "depart_time": "07:30",
        "arrive_time": "09:00",
        "origin": "自宅",
        "destination": "学校",
        "transport": "電車",
        "destination_environment": "学校(屋内)",
        "destination_activity": "座位",
        "notes": "登校",
    },
    {
        "depart_time": "17:00",
        "arrive_time": "18:00",
        "origin": "学校",
        "destination": "自宅",
        "transport": "電車",
        "destination_environment": "自宅(屋内)",
        "destination_activity": "座位",
        "notes": "帰宅",
    },
]


def geocode_place_name(query: str):
    if not query.strip():
        return None

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?"
            + urllib.parse.urlencode({"q": query, "format": "jsonv2", "limit": 1})
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "streamlit-cz-app/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            raw = res.read().decode("utf-8")
            data = json.loads(raw)
            if not data:
                return None
            item = data[0]
            return {
                "display_name": item.get("display_name", query),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            }
    except Exception:
        return None


if "places" not in st.session_state:
    st.session_state.places = DEFAULT_PLACES.copy()

if "draft_trip_rows" not in st.session_state:
    st.session_state.draft_trip_rows = [row.copy() for row in DEFAULT_TRIPS]

if "geocode_result" not in st.session_state:
    st.session_state.geocode_result = None

if "closet_items" not in st.session_state:
    st.session_state.closet_items = load_closet_items()

if "closet_classification_result" not in st.session_state:
    st.session_state.closet_classification_result = None

if "closet_selected_category" not in st.session_state:
    st.session_state.closet_selected_category = CLOTHING_CATEGORY_CANDIDATES[0]

if "closet_category_method" not in st.session_state:
    st.session_state.closet_category_method = "manual"

if "closet_last_image_name" not in st.session_state:
    st.session_state.closet_last_image_name = None


@st.cache_resource(show_spinner=False)
def get_clothing_classifier_resource():
    return load_clip_classifier()


def add_trip_row():
    place_options = sorted(list(st.session_state.places.keys()))
    default_origin = place_options[0] if place_options else ""
    default_destination = place_options[0] if place_options else ""
    default_dest_env = infer_environment_from_place(default_destination, st.session_state.places) if default_destination else "屋外"

    st.session_state.draft_trip_rows.append({
        "depart_time": "18:00",
        "arrive_time": "19:00",
        "origin": default_origin,
        "destination": default_destination,
        "transport": "電車",
        "destination_environment": default_dest_env,
        "destination_activity": "座位",
        "notes": "",
    })


def remove_trip_row(index: int):
    if 0 <= index < len(st.session_state.draft_trip_rows):
        st.session_state.draft_trip_rows.pop(index)


def move_trip_up(index: int):
    if index > 0:
        rows = st.session_state.draft_trip_rows
        rows[index - 1], rows[index] = rows[index], rows[index - 1]


def move_trip_down(index: int):
    rows = st.session_state.draft_trip_rows
    if index < len(rows) - 1:
        rows[index + 1], rows[index] = rows[index], rows[index + 1]


def reset_trips():
    st.session_state.draft_trip_rows = [row.copy() for row in DEFAULT_TRIPS]


def generate_input_image_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows:
        lines.append(
            f"{row.get('depart_time','')} {row.get('origin','')} を出る → "
            f"{row.get('arrive_time','')} {row.get('destination','')} に着く（{row.get('transport','')}）"
        )
    return "\n".join(lines)


def plot_research_figure(minute_df: pd.DataFrame, title: str):
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    x = minute_df["minute_of_day"]

    axes[0].plot(x, minute_df["estimated_temp"], label="Environment Temp")
    axes[0].plot(x, minute_df["teff"], label="Teff")
    axes[0].set_ylabel("Temp [°C]")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(x, minute_df["rh"], label="RH")
    axes[1].set_ylabel("RH [%]")
    axes[1].legend()

    if minute_df["pmv"].notna().any():
        axes[2].plot(x, minute_df["pmv"], label="PMV")
        axes[2].axhline(0.5, linestyle="--", label="PMV +0.5")
        axes[2].axhline(-0.5, linestyle="--", label="PMV -0.5")
        axes[2].set_ylabel("PMV")
        axes[2].legend()
    else:
        axes[2].text(
            0.5, 0.5,
            "PMV unavailable\n(pythermalcomfort not installed or calculation failed)",
            ha="center", va="center", transform=axes[2].transAxes
        )
        axes[2].set_ylabel("PMV")

    axes[3].plot(x, minute_df["met"], label="MET")
    axes[3].plot(x, minute_df["clo"], label="CLO")
    axes[3].set_ylabel("MET / CLO")
    axes[3].legend()

    axes[4].plot(x, minute_df["v"], label="Air Speed")
    axes[4].set_ylabel("m/s")
    axes[4].set_xlabel("Minute of day")
    axes[4].legend()

    plt.tight_layout()
    return fig


def _evaluation_debug_anomalies(minute_df: pd.DataFrame, recommendation_comparison: dict, clo_min: float, clo_max: float) -> list[str]:
    warnings = []
    if minute_df is None or minute_df.empty:
        return warnings

    checks = [
        ("Teff", "teff", 0.0, 50.0),
        ("ΔT", "delta_t", -20.0, 30.0),
    ]
    if "estimated_temp" in minute_df.columns and "delta_t" in minute_df.columns:
        ttarget = pd.to_numeric(minute_df["estimated_temp"], errors="coerce") + pd.to_numeric(minute_df["delta_t"], errors="coerce")
        if ttarget.min() < 0 or ttarget.max() > 50:
            warnings.append(f"Ttarget が通常範囲外です: min={ttarget.min():.3f}, max={ttarget.max():.3f}")

    for label, col, lower, upper in checks:
        if col in minute_df.columns:
            values = pd.to_numeric(minute_df[col], errors="coerce").dropna()
            if not values.empty and (values.min() < lower or values.max() > upper):
                warnings.append(f"{label} が通常範囲外です: min={values.min():.3f}, max={values.max():.3f}")

    for method_key, method_name in [("conventional", "従来手法"), ("proposed", "提案手法")]:
        result = recommendation_comparison.get(method_key, {})
        best_clo = float(result.get("best_clo", 0.0))
        best_pmv = result.get("best_pmv")
        if abs(best_clo - float(clo_max)) < 1e-9:
            warnings.append(f"{method_name}: 推奨CLOが探索上限 {clo_max:.2f} に張り付いています。")
        if abs(best_clo - float(clo_min)) < 1e-9:
            warnings.append(f"{method_name}: 推奨CLOが探索下限 {clo_min:.2f} に張り付いています。")
        if best_pmv is not None and abs(float(best_pmv)) > 3.0:
            warnings.append(f"{method_name}: PMV が -3〜3 を大きく超えています（PMV={float(best_pmv):.3f}）。")

    return warnings


def _build_teff_step_debug_df(minute_df: pd.DataFrame, alpha_value: float, dt_min: float = 1.0, rows: int = 10) -> pd.DataFrame:
    debug_rows = []
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()

    for i in range(min(rows, len(minute_df))):
        tenv = float(minute_df["estimated_temp"].iloc[i])
        delta_t = float(minute_df["delta_t"].iloc[i]) if "delta_t" in minute_df.columns else 0.0
        ttarget = tenv + delta_t
        teff_after = float(minute_df["teff"].iloc[i])
        teff_before = teff_after if i == 0 else float(minute_df["teff"].iloc[i - 1])
        debug_rows.append({
            "step": i,
            "時刻": minute_df["datetime"].iloc[i] if "datetime" in minute_df.columns else i,
            "Tenv": tenv,
            "ΔT": delta_t,
            "Ttarget": ttarget,
            "Teff_before": teff_before,
            "α": float(alpha_value),
            "Δt": float(dt_min),
            "Teff_after": teff_after,
            "温度差": minute_df["transition_temp_diff"].iloc[i] if "transition_temp_diff" in minute_df.columns else None,
            "遷移方向": minute_df["transition_direction"].iloc[i] if "transition_direction" in minute_df.columns else None,
            "代表温度差": minute_df["delta_t_selected_temp_diff"].iloc[i] if "delta_t_selected_temp_diff" in minute_df.columns else None,
            "使用ΔT式": minute_df["delta_t_formula"].iloc[i] if "delta_t_formula" in minute_df.columns else None,
            "t_since_transition": minute_df["transition_elapsed_min"].iloc[i] if "transition_elapsed_min" in minute_df.columns else None,
            "t_clip": minute_df["delta_t_elapsed_clipped_min"].iloc[i] if "delta_t_elapsed_clipped_min" in minute_df.columns else None,
        })
    return pd.DataFrame(debug_rows)


def _build_pmv_input_debug_df(minute_df: pd.DataFrame, pmv_df: pd.DataFrame, method_name: str, input_temp_col: str, clo_value: float, rows: int = 20) -> pd.DataFrame:
    debug_df = pd.DataFrame({
        "手法": method_name,
        "時刻": minute_df["datetime"] if "datetime" in minute_df.columns else minute_df.index,
        "tdb": pd.to_numeric(minute_df[input_temp_col], errors="coerce"),
        "tr": pd.to_numeric(minute_df[input_temp_col], errors="coerce"),
        "rh": pd.to_numeric(minute_df["rh"], errors="coerce"),
        "v": pd.to_numeric(minute_df["v"], errors="coerce"),
        "met": pd.to_numeric(minute_df["met"], errors="coerce"),
        "clo": float(clo_value),
        "PMV計算結果": pmv_df["pmv"] if "pmv" in pmv_df.columns else None,
    })
    return debug_df.head(rows)


def _summarize_search_results(search_results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(search_results)
    if df.empty:
        return df
    ok_df = df[df["status"] == "ok"].copy()
    head_df = df.head(10)
    tail_df = df.tail(10)
    best_df = ok_df.sort_values("error").head(10) if not ok_df.empty else pd.DataFrame()
    summary = pd.concat([
        head_df.assign(表示区分="最初の10件"),
        tail_df.assign(表示区分="最後の10件"),
        best_df.assign(表示区分="error上位10件"),
    ], ignore_index=True)
    return summary.drop_duplicates(subset=["clo", "表示区分"])


st.title("🌡️ 環境温度推定 & CZ判定アプリ")
st.caption("主要な移動だけ入力し、徒歩・電車などの細かい区間はアプリ側で自動補完します。")

st.sidebar.header("全体設定")

season = st.sidebar.selectbox(
    "季節",
    options=["spring", "summer", "autumn", "winter"],
    format_func=lambda x: SEASON_LABELS.get(x, x),
    index=1,
)
target_date = st.sidebar.date_input("対象日", value=datetime.today().date())

preset = SEASON_PRESETS[season].copy()

st.sidebar.header("個人設定")
thermal_type = st.sidebar.selectbox(
    "体感タイプ",
    options=THERMAL_TYPE_OPTIONS,
    index=THERMAL_TYPE_OPTIONS.index("普通"),
)

if "last_thermal_type" not in st.session_state:
    st.session_state.last_thermal_type = thermal_type
    st.session_state.personal_clo_offset = get_initial_personal_clo_offset(thermal_type)
elif st.session_state.last_thermal_type != thermal_type:
    st.session_state.last_thermal_type = thermal_type
    st.session_state.personal_clo_offset = get_initial_personal_clo_offset(thermal_type)
    st.session_state.last_feedback_message = None

if "personal_clo_offset" not in st.session_state:
    st.session_state.personal_clo_offset = get_initial_personal_clo_offset(thermal_type)

st.sidebar.subheader("1日の基本設定")
day_start_time = st.sidebar.text_input("1日の開始時刻", value="06:00")
day_end_time = st.sidebar.text_input("1日の終了時刻", value="23:00")

place_options_sidebar = sorted(list(st.session_state.places.keys()))
default_start_place_idx = place_options_sidebar.index("自宅") if "自宅" in place_options_sidebar else 0

start_place = st.sidebar.selectbox(
    "開始時点でいる場所",
    options=place_options_sidebar if place_options_sidebar else [""],
    index=default_start_place_idx if place_options_sidebar else 0
)
start_environment_default = infer_environment_from_place(start_place, st.session_state.places) if start_place else "屋外"
start_environment = st.sidebar.selectbox(
    "開始時点の環境",
    options=ENVIRONMENT_OPTIONS,
    index=ENVIRONMENT_OPTIONS.index(start_environment_default) if start_environment_default in ENVIRONMENT_OPTIONS else 0
)
start_activity = st.sidebar.selectbox(
    "開始時点の活動量",
    options=ACTIVITY_OPTIONS,
    index=ACTIVITY_OPTIONS.index("座位")
)

st.sidebar.subheader("移動自動補完設定")
walk_buffer_min = st.sidebar.slider("電車移動の前後徒歩時間 [分]", 3, 20, 10, 1)

st.sidebar.subheader("温熱モデル設定")
alpha = st.sidebar.slider("一次遅れ係数 α", 0.01, 1.00, 0.15, 0.01)
clo = st.sidebar.slider("Clo値", 0.0, 2.0, 0.8, 0.1)
use_teff_for_tr = st.sidebar.checkbox("放射温度にも Teff を使う", value=False)
use_open_meteo = st.sidebar.checkbox("屋外に Open-Meteo 実気象データを使う", value=True)

st.sidebar.subheader("快適判定設定")
pmv_lower = st.sidebar.slider("PMV下限", -3.0, 0.0, -0.5, 0.1)
pmv_upper = st.sidebar.slider("PMV上限", 0.0, 3.0, 0.5, 0.1)

st.sidebar.subheader("代表温度設定（屋内用）")
preset["outdoor_min"] = st.sidebar.number_input("外気温(最低) [℃]", value=float(preset["outdoor_min"]), step=0.5)
preset["outdoor_max"] = st.sidebar.number_input("外気温(最高) [℃]", value=float(preset["outdoor_max"]), step=0.5)
preset["indoor_home"] = st.sidebar.number_input("自宅室内 [℃]", value=float(preset["indoor_home"]), step=0.5)
preset["indoor_school"] = st.sidebar.number_input("学校室内 [℃]", value=float(preset["indoor_school"]), step=0.5)
preset["indoor_office"] = st.sidebar.number_input("オフィス室内 [℃]", value=float(preset["indoor_office"]), step=0.5)
preset["indoor_shop"] = st.sidebar.number_input("店舗室内 [℃]", value=float(preset["indoor_shop"]), step=0.5)
preset["train_temp"] = st.sidebar.number_input("電車内 [℃]", value=float(preset["train_temp"]), step=0.5)
preset["car_temp"] = st.sidebar.number_input("車内 [℃]", value=float(preset["car_temp"]), step=0.5)
preset["other_vehicle_temp"] = st.sidebar.number_input("その他乗り物内 [℃]", value=float(preset["other_vehicle_temp"]), step=0.5)

st.sidebar.subheader("湿度設定（屋内用）")
preset["outdoor_rh"] = st.sidebar.number_input("屋外RH [%]", value=float(preset["outdoor_rh"]), step=1.0)
preset["indoor_rh"] = st.sidebar.number_input("屋内RH [%]", value=float(preset["indoor_rh"]), step=1.0)
preset["train_rh"] = st.sidebar.number_input("電車RH [%]", value=float(preset["train_rh"]), step=1.0)
preset["car_rh"] = st.sidebar.number_input("車RH [%]", value=float(preset["car_rh"]), step=1.0)

st.sidebar.subheader("風速設定（屋内用）")
preset["outdoor_v"] = st.sidebar.number_input("屋外風速 [m/s]", value=float(preset["outdoor_v"]), step=0.05, format="%.2f")
preset["indoor_v"] = st.sidebar.number_input("屋内風速 [m/s]", value=float(preset["indoor_v"]), step=0.05, format="%.2f")
preset["train_v"] = st.sidebar.number_input("電車風速 [m/s]", value=float(preset["train_v"]), step=0.05, format="%.2f")
preset["car_v"] = st.sidebar.number_input("車風速 [m/s]", value=float(preset["car_v"]), step=0.05, format="%.2f")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 主要移動の入力",
    "② 地点マスタ",
    "③ 推定結果",
    "④ 研究者向け可視化",
    "⑤ クローゼット登録",
    "⑥ 提案手法評価",
])

with tab1:
    st.subheader("主要移動の入力")
    st.info("電車を選ぶと、前後の徒歩時間は自動で補完されます。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("移動を追加", use_container_width=True):
            add_trip_row()
            st.rerun()
    with col2:
        if st.button("初期例に戻す", use_container_width=True):
            reset_trips()
            st.rerun()

    place_options = sorted(list(st.session_state.places.keys()))
    if not place_options:
        st.warning("地点マスタが空です。")
    else:
        for i, row in enumerate(st.session_state.draft_trip_rows):
            with st.container(border=True):
                st.markdown(f"### 移動 {i + 1}")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    depart_time = st.text_input("出発時刻", value=row["depart_time"], key=f"depart_time_{i}")
                with c2:
                    arrive_time = st.text_input("到着時刻", value=row["arrive_time"], key=f"arrive_time_{i}")
                with c3:
                    origin = st.selectbox("出発地", place_options, index=place_options.index(row["origin"]) if row["origin"] in place_options else 0, key=f"origin_{i}")
                with c4:
                    destination = st.selectbox("到着地", place_options, index=place_options.index(row["destination"]) if row["destination"] in place_options else 0, key=f"destination_{i}")

                c5, c6, c7 = st.columns(3)
                with c5:
                    transport = st.selectbox("移動手段", TRANSPORT_OPTIONS, index=TRANSPORT_OPTIONS.index(row["transport"]) if row["transport"] in TRANSPORT_OPTIONS else 1, key=f"transport_{i}")
                with c6:
                    destination_environment = st.selectbox("到着後の環境", ENVIRONMENT_OPTIONS, index=ENVIRONMENT_OPTIONS.index(row["destination_environment"]) if row["destination_environment"] in ENVIRONMENT_OPTIONS else 0, key=f"dest_env_{i}")
                with c7:
                    destination_activity = st.selectbox("到着後の活動量", ACTIVITY_OPTIONS, index=ACTIVITY_OPTIONS.index(row["destination_activity"]) if row["destination_activity"] in ACTIVITY_OPTIONS else 0, key=f"dest_act_{i}")

                notes = st.text_input("メモ", value=row.get("notes", ""), key=f"notes_{i}")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("↑ 上へ", key=f"up_{i}", use_container_width=True):
                        move_trip_up(i)
                        st.rerun()
                with b2:
                    if st.button("↓ 下へ", key=f"down_{i}", use_container_width=True):
                        move_trip_down(i)
                        st.rerun()
                with b3:
                    if st.button("削除", key=f"del_{i}", use_container_width=True):
                        remove_trip_row(i)
                        st.rerun()
                with b4:
                    if st.button("到着地から環境補完", key=f"auto_env_{i}", use_container_width=True):
                        st.session_state.draft_trip_rows[i]["destination_environment"] = infer_environment_from_place(destination, st.session_state.places)
                        st.rerun()

                st.session_state.draft_trip_rows[i] = {
                    "depart_time": depart_time,
                    "arrive_time": arrive_time,
                    "origin": origin,
                    "destination": destination,
                    "transport": transport,
                    "destination_environment": destination_environment,
                    "destination_activity": destination_activity,
                    "notes": notes,
                }

        st.markdown("#### 入力中のプレビュー")
        st.dataframe(pd.DataFrame(st.session_state.draft_trip_rows), use_container_width=True)

        st.markdown("#### 入力イメージ")
        input_image_text = generate_input_image_text(st.session_state.draft_trip_rows)
        if input_image_text:
            st.code(input_image_text, language="text")

with tab2:
    st.subheader("地点マスタ")
    places_df = pd.DataFrame([
        {"place_name": name, "lat": info["lat"], "lon": info["lon"], "default_env": info["default_env"]}
        for name, info in st.session_state.places.items()
    ]).sort_values("place_name").reset_index(drop=True)
    st.dataframe(places_df, use_container_width=True)

    left, right = st.columns(2)

    with left:
        with st.form("manual_place_form"):
            new_place_name = st.text_input("地点名")
            new_lat = st.number_input("緯度", value=35.6580, format="%.6f")
            new_lon = st.number_input("経度", value=139.7016, format="%.6f")
            new_default_env = st.selectbox("既定環境", ["屋外", "屋内"])
            submit_manual = st.form_submit_button("地点を追加")

            if submit_manual and new_place_name.strip():
                st.session_state.places[new_place_name.strip()] = {
                    "lat": float(new_lat),
                    "lon": float(new_lon),
                    "default_env": new_default_env,
                }
                st.rerun()

    with right:
        geo_query = st.text_input("地点名で検索", placeholder="例: 埼玉県大宮駅")
        geo_env = st.selectbox("検索追加時の既定環境", ["屋外", "屋内"], key="geo_env")

        g1, g2 = st.columns(2)
        with g1:
            if st.button("検索する"):
                st.session_state.geocode_result = geocode_place_name(geo_query)
        with g2:
            if st.button("検索結果を追加") and st.session_state.geocode_result is not None:
                result = st.session_state.geocode_result
                label = geo_query.strip() if geo_query.strip() else result["display_name"]
                st.session_state.places[label] = {
                    "lat": float(result["lat"]),
                    "lon": float(result["lon"]),
                    "default_env": geo_env,
                }
                st.session_state.geocode_result = None
                st.rerun()

        if st.session_state.geocode_result:
            result = st.session_state.geocode_result
            st.info(f"{result['display_name']} / lat={result['lat']:.6f}, lon={result['lon']:.6f}")

base_clo = float(clo)
adjusted_clo = base_clo + float(st.session_state.personal_clo_offset)

trip_df = pd.DataFrame(st.session_state.draft_trip_rows)
result_ready = False
minute_df = None
segment_summary_df = None
summary = None
pmv_available = False
errors = []

if not trip_df.empty:
    errors = validate_trip_df(trip_df)
    if not errors:
        segments_df = build_daily_segments_from_trips(
            trip_df=trip_df,
            target_date=target_date,
            season=season,
            preset=preset,
            day_start_time=day_start_time,
            day_end_time=day_end_time,
            start_place=start_place,
            start_environment=start_environment,
            start_activity=start_activity,
            walk_buffer_min=walk_buffer_min,
        )

        minute_df = expand_segments_to_time_level(segments_df)

        if use_open_meteo:
            minute_df = apply_open_meteo_to_outdoor_segments(
                minute_df=minute_df,
                places=st.session_state.places,
                trip_df=trip_df,
                target_date=target_date,
            )

        minute_df = apply_first_order_lag_to_minutes(minute_df, alpha=alpha)
        minute_df, pmv_available = calculate_pmv_series(
            minute_df,
            clo=adjusted_clo,
            use_teff_for_tdb=True,
            use_teff_for_tr=use_teff_for_tr,
        )
        minute_df = apply_cz_from_pmv(minute_df, pmv_lower=pmv_lower, pmv_upper=pmv_upper)

        summary = summarize_results(minute_df)
        segment_summary_df = aggregate_minutes_to_segments(minute_df)
        result_ready = True

with tab3:
    st.subheader("推定結果")

    st.markdown("### 個人適応設定")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("体感タイプ", thermal_type)
    p2.metric("基本CLO", f"{base_clo:.2f}")
    p3.metric("個人補正値", f"{st.session_state.personal_clo_offset:+.2f}")
    p4.metric("補正後CLO", f"{adjusted_clo:.2f}")
    st.info(get_recommendation_message(st.session_state.personal_clo_offset))

    st.markdown("### フィードバック")
    feedback_label = st.radio(
        "推薦された服装はどうでしたか？",
        options=FEEDBACK_OPTIONS,
        index=FEEDBACK_OPTIONS.index("ちょうどよかった"),
        horizontal=True,
        key="feedback_label",
    )
    if st.button("フィードバックを反映"):
        score = FEEDBACK_SCORE_MAP[feedback_label]
        st.session_state.personal_clo_offset = update_personal_clo_offset(
            st.session_state.personal_clo_offset, score
        )
        st.session_state.last_feedback_message = (
            f"フィードバックを反映しました。次回以降の推薦では、個人補正値 "
            f"{st.session_state.personal_clo_offset:+.2f} を使用します。"
        )
        st.rerun()

    if st.session_state.get("last_feedback_message"):
        st.success(st.session_state.last_feedback_message)

    if trip_df.empty:
        st.warning("主要移動データが空です。")
    else:
        if errors:
            for err in errors:
                st.error(err)
        elif result_ready:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("平均環境温度", f"{summary['mean_temp']} ℃")
            m2.metric("平均Teff", f"{summary['mean_teff']} ℃")
            m3.metric("快適時間割合", f"{summary['comfort_ratio']} %")
            m4.metric("寒い時間割合", f"{summary['cold_ratio']} %")
            m5.metric("暑い時間割合", f"{summary['hot_ratio']} %")

            if summary["mean_pmv"] is not None:
                st.metric("平均PMV", f"{summary['mean_pmv']}")

            st.markdown("### 区間別要約")
            st.dataframe(
                segment_summary_df.rename(columns={
                    "label": "区間名",
                    "start_time": "開始",
                    "end_time": "終了",
                    "duration_min": "時間[分]",
                    "environment": "環境",
                    "transport": "移動手段",
                    "activity": "活動量",
                    "mean_tenv": "平均環境温度[℃]",
                    "mean_teff": "平均Teff[℃]",
                    "mean_rh": "平均RH[%]",
                    "mean_v": "平均風速[m/s]",
                    "mean_met": "平均MET",
                    "clo": "CLO",
                    "mean_pmv": "平均PMV",
                    "cz_result_majority": "CZ判定",
                    "notes": "メモ",
                }),
                use_container_width=True
            )

            st.markdown("### 5分粒度データ（先頭）")
            display_cols = [
                "datetime", "label", "environment", "transport", "activity",
                "estimated_temp", "teff", "rh", "v", "met", "clo", "pmv", "cz_result"
            ]
            st.dataframe(minute_df[display_cols].head(300), use_container_width=True)

with tab4:
    st.subheader("研究者向け可視化")
    if trip_df.empty:
        st.warning("主要移動データが空です。")
    else:
        if errors:
            for err in errors:
                st.error(err)
        elif result_ready:
            if not pmv_available:
                st.warning("PMVは未計算です。`pip install pythermalcomfort` を実行すると表示できます。")

            st.markdown("### 背景で何が計算されているか")
            fig = plot_research_figure(
                minute_df,
                title=f"Research View / {target_date} / {SEASON_LABELS.get(season, season)}"
            )
            st.pyplot(fig)

            st.markdown("### 可視化対象の説明")
            st.write("・Environment Temp: 各時刻の環境温度（屋外はOpen-Meteo実データ、屋内は代表温度）")
            st.write("・Teff: 環境温度に対して一次遅れで変化する人体側の有効温度")
            st.write("・PMV: Teff を tdb に入力して計算した快適指標")
            st.write("・PMV:その場所の熱い、寒いの感じ方の数値")
            st.write("  →0.5の点線は快適ライン")
            st.write("・PMV上下限: 快適判定に使う閾値")
            st.write("・MET / CLO: PMVに入れた活動量と着衣量")
            st.write("・Air Speed / RH: PMVに入れた風速と湿度")

            st.markdown("### PMV入力値の一覧")
            pmv_input_df = minute_df[[
                "minute_of_day", "label", "estimated_temp", "teff",
                "delta_t", "transition_direction", "transition_elapsed_min",
                "rh", "v", "met", "clo", "pmv", "cz_result"
            ]].copy()
            pmv_input_df.columns = [
                "minute_of_day", "区間", "環境温度[℃]", "Teff[℃]",
                "ΔT", "遷移方向", "遷移経過[min]",
                "RH[%]", "風速[m/s]", "MET", "CLO", "PMV", "CZ判定"
            ]
            st.dataframe(pmv_input_df.head(500), use_container_width=True)
with tab5:
    st.subheader("クローゼット登録")

    uploaded_file = st.file_uploader(
        "服画像",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="closet_image_uploader",
    )
    item_name = st.text_input("登録名", key="closet_item_name")

    if uploaded_file is not None:
        if st.session_state.closet_last_image_name != uploaded_file.name:
            st.session_state.closet_last_image_name = uploaded_file.name
            st.session_state.closet_classification_result = None
            st.session_state.closet_selected_category = CLOTHING_CATEGORY_CANDIDATES[0]
            st.session_state.closet_category_method = "manual"

        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption=uploaded_file.name, width=260)

        if st.button("画像から自動判定", key="classify_closet_image"):
            try:
                classifier_resource = get_clothing_classifier_resource()
                if classifier_resource.get("warning"):
                    st.warning(classifier_resource["warning"])
                result = classify_clothing_image(
                    image_bytes,
                    model_bundle=classifier_resource,
                    categories=CLOTHING_CATEGORY_CANDIDATES,
                    clo_dict=CLO_DB,
                )
                st.session_state.closet_classification_result = result
                st.session_state.closet_selected_category = result["category"]
                st.session_state.closet_category_method = result.get("method", "fashionclip")
                st.rerun()
            except Exception as exc:
                st.session_state.closet_classification_result = None
                st.session_state.closet_category_method = "manual"
                st.error(f"ローカル画像分類に失敗しました。手動でカテゴリを選択してください。詳細: {exc}")

        result = st.session_state.closet_classification_result
        if result:
            st.markdown("### AI判定結果")
            r1, r2, r3 = st.columns(3)
            r1.metric("判定カテゴリ", result["category"])
            r2.metric("CLO値", f"{float(result['clo']):.2f}")
            r3.metric("信頼度", f"{float(result['confidence']) * 100:.1f}%")

            st.markdown("#### 上位3候補")
            st.dataframe(pd.DataFrame(result["top_candidates"]), use_container_width=True)

            if st.button("この判定を採用", key="adopt_closet_prediction"):
                st.session_state.closet_selected_category = result["category"]
                st.session_state.closet_category_method = result.get("method", "fashionclip")
                st.success("AI判定を採用しました。")

        manual_category = st.selectbox(
            "手動で修正する",
            options=CLOTHING_CATEGORY_CANDIDATES,
            index=CLOTHING_CATEGORY_CANDIDATES.index(st.session_state.closet_selected_category)
            if st.session_state.closet_selected_category in CLOTHING_CATEGORY_CANDIDATES else 0,
            key="closet_manual_category_select",
        )
        if manual_category != st.session_state.closet_selected_category:
            st.session_state.closet_selected_category = manual_category
            st.session_state.closet_category_method = "manual"

        final_category = st.session_state.closet_selected_category
        final_clo = float(CLO_DB.get(final_category, 0.0))
        confidence = 0.0
        if result and result.get("category") == final_category:
            confidence = float(result.get("confidence", 0.0))

        st.metric("登録されるCLO値", f"{final_clo:.2f}")
        if st.button("登録する", key="register_closet_item"):
            if not item_name.strip():
                st.error("登録名を入力してください。")
            else:
                image_path = save_uploaded_closet_image(image_bytes, uploaded_file.name)
                st.session_state.closet_items.append({
                    "name": item_name.strip(),
                    "image_path": image_path,
                    "image_name": uploaded_file.name,
                    "image_type": uploaded_file.type,
                    "category": final_category,
                    "categories": [final_category],
                    "clo": final_clo,
                    "total_clo": final_clo,
                    "method": st.session_state.closet_category_method,
                    "confidence": confidence,
                    "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                save_closet_items(st.session_state.closet_items)
                st.success("クローゼットに登録しました。")
    else:
        st.info("服画像をアップロードすると、FashionCLIP/CLIPでカテゴリを推定できます。")

    st.markdown("### 登録済みクローゼット一覧")
    st.caption(f"保存先: {CLOSET_DB_PATH}")
    if not st.session_state.closet_items:
        st.info("登録済みの服はまだありません。")
    else:
        for i, item in enumerate(st.session_state.closet_items):
            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.3, 1.5, 0.8, 1.0, 0.8])
                with c1:
                    image_source = item.get("image_path") or item.get("image")
                    if image_source:
                        st.image(image_source, caption=item.get("image_name", "服画像"), use_container_width=True)
                    else:
                        st.caption("画像なし")
                with c2:
                    st.write(item.get("name", ""))
                    st.caption(item.get("registered_at", ""))
                with c3:
                    st.write(item.get("category", "、".join(item.get("categories", []))))
                    st.caption(f"判定方法: {item.get('method', 'manual')}")
                with c4:
                    st.metric("CLO", f"{float(item.get('clo', item.get('total_clo', 0.0))):.2f}")
                with c5:
                    st.metric("信頼度", f"{float(item.get('confidence', 0.0)) * 100:.1f}%")
                with c6:
                    if st.button("削除", key=f"delete_closet_item_{i}", use_container_width=True):
                        st.session_state.closet_items = delete_closet_item(st.session_state.closet_items, i)
                        st.rerun()


with tab6:
    st.subheader("提案手法評価")
    st.caption("従来手法は PMV入力温度を Tenv、提案手法は PMV入力温度を Teff として比較します。")

    missing_message = "評価に必要なデータが不足しています。予定入力・環境温度取得・温熱状態推定を先に実行してください。"
    required_eval_cols = {"estimated_temp", "teff", "delta_t", "delta_t_elapsed_clipped_min", "rh", "v", "met"}

    if trip_df.empty or errors or not result_ready or minute_df is None or minute_df.empty:
        st.warning(missing_message)
        if errors:
            for err in errors:
                st.error(err)
    elif not required_eval_cols.issubset(set(minute_df.columns)):
        st.warning(missing_message)
    else:
        eval_clo_min, eval_clo_max, eval_clo_step = st.columns(3)
        with eval_clo_min:
            clo_min_eval = st.number_input("CLO探索下限", value=0.10, min_value=0.0, max_value=5.0, step=0.01, format="%.2f")
        with eval_clo_max:
            clo_max_eval = st.number_input("CLO探索上限", value=2.00, min_value=0.1, max_value=5.0, step=0.01, format="%.2f")
        with eval_clo_step:
            clo_step_eval = st.number_input("CLO探索刻み", value=0.01, min_value=0.01, max_value=0.50, step=0.01, format="%.2f")

        try:
            recommendation_comparison = compare_clo_recommendation(
                minute_df,
                clo_min=clo_min_eval,
                clo_max=clo_max_eval,
                clo_step=clo_step_eval,
            )
            clo_base = float(recommendation_comparison["proposed"]["best_clo"])

            st.markdown("### 1. 推奨CLO比較")
            rec_df = pd.DataFrame([
                {
                    "手法": "従来手法",
                    "入力温度": "Tenv",
                    "推奨CLO": recommendation_comparison["conventional"]["best_clo"],
                    "最終PMV": recommendation_comparison["conventional"]["best_pmv"],
                    "誤差": recommendation_comparison["conventional"]["best_error"],
                },
                {
                    "手法": "提案手法",
                    "入力温度": "Teff",
                    "推奨CLO": recommendation_comparison["proposed"]["best_clo"],
                    "最終PMV": recommendation_comparison["proposed"]["best_pmv"],
                    "誤差": recommendation_comparison["proposed"]["best_error"],
                },
            ])
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

            for method_key, method_name in [("conventional", "従来手法"), ("proposed", "提案手法")]:
                warnings = recommendation_comparison[method_key].get("warnings", [])
                for warning in warnings:
                    st.warning(f"{method_name}: {warning}")

            debug_df = pd.DataFrame([
                {
                    "手法": "従来手法",
                    "入力温度": "Tenv",
                    "入力温度最小": recommendation_comparison["conventional"]["debug"]["temperature_min"],
                    "入力温度最大": recommendation_comparison["conventional"]["debug"]["temperature_max"],
                    "湿度": recommendation_comparison["conventional"]["debug"]["humidity"],
                    "風速": recommendation_comparison["conventional"]["debug"]["air_speed"],
                    "代謝量": recommendation_comparison["conventional"]["debug"]["met"],
                    "探索された推奨CLO": recommendation_comparison["conventional"]["debug"]["best_clo"],
                    "最終PMV": recommendation_comparison["conventional"]["debug"]["best_pmv"],
                },
                {
                    "手法": "提案手法",
                    "入力温度": "Teff",
                    "入力温度最小": recommendation_comparison["proposed"]["debug"]["temperature_min"],
                    "入力温度最大": recommendation_comparison["proposed"]["debug"]["temperature_max"],
                    "湿度": recommendation_comparison["proposed"]["debug"]["humidity"],
                    "風速": recommendation_comparison["proposed"]["debug"]["air_speed"],
                    "代謝量": recommendation_comparison["proposed"]["debug"]["met"],
                    "探索された推奨CLO": recommendation_comparison["proposed"]["debug"]["best_clo"],
                    "最終PMV": recommendation_comparison["proposed"]["debug"]["best_pmv"],
                },
            ])
            for warning in _evaluation_debug_anomalies(minute_df, recommendation_comparison, clo_min_eval, clo_max_eval):
                st.warning(warning)

            with st.expander("Teff計算デバッグ情報"):
                teff_debug_df = minute_df.copy()
                teff_debug_df["Ttarget"] = pd.to_numeric(teff_debug_df["estimated_temp"], errors="coerce") + pd.to_numeric(teff_debug_df["delta_t"], errors="coerce")
                teff_summary_df = pd.DataFrame([
                    {"項目": "α", "値": float(alpha)},
                    {"項目": "Δt", "値": 1.0},
                    {"項目": "ΔT式のt上限[min]", "値": float(DELTA_T_MAX_MINUTES)},
                    {"項目": "初期Teff", "値": float(teff_debug_df["teff"].iloc[0])},
                    {"項目": "使用可能なΔT式", "値": str(DELTA_T_COEFFS)},
                    {"項目": "使用している遷移方向", "値": "、".join([str(v) for v in teff_debug_df["transition_direction"].dropna().unique()])},
                    {"項目": "使用している代表温度差", "値": "、".join([str(v) for v in teff_debug_df["delta_t_selected_temp_diff"].dropna().unique()]) if "delta_t_selected_temp_diff" in teff_debug_df.columns else ""},
                    {"項目": "Tenv 最小値", "値": float(teff_debug_df["estimated_temp"].min())},
                    {"項目": "Tenv 最大値", "値": float(teff_debug_df["estimated_temp"].max())},
                    {"項目": "ΔT 最小値", "値": float(teff_debug_df["delta_t"].min())},
                    {"項目": "ΔT 最大値", "値": float(teff_debug_df["delta_t"].max())},
                    {"項目": "Ttarget 最小値", "値": float(teff_debug_df["Ttarget"].min())},
                    {"項目": "Ttarget 最大値", "値": float(teff_debug_df["Ttarget"].max())},
                    {"項目": "Teff 最小値", "値": float(teff_debug_df["teff"].min())},
                    {"項目": "Teff 最大値", "値": float(teff_debug_df["teff"].max())},
                ])
                st.dataframe(teff_summary_df, use_container_width=True, hide_index=True)
                st.line_chart(teff_debug_df[["datetime", "estimated_temp", "delta_t", "Ttarget", "teff"]].set_index("datetime"))
                # ΔT式確認用の列は、古いセッションデータが残っている場合に備えて存在する列だけ表示する。
                teff_debug_cols = [
                    "datetime", "estimated_temp", "delta_t", "Ttarget", "teff",
                    "transition_temp_diff", "transition_direction", "delta_t_selected_temp_diff",
                    "delta_t_formula", "transition_elapsed_min", "delta_t_elapsed_clipped_min"
                ]
                teff_debug_cols = [col for col in teff_debug_cols if col in teff_debug_df.columns]
                st.dataframe(
                    teff_debug_df[teff_debug_cols].head(300),
                    use_container_width=True,
                )
                st.markdown("#### Teff計算ステップ確認（先頭10行）")
                st.dataframe(_build_teff_step_debug_df(minute_df, alpha), use_container_width=True, hide_index=True)

            with st.expander("PMV計算デバッグ情報"):
                conventional_pmv_debug_df, _ = calculate_pmv_series(
                    minute_df,
                    clo=clo_base,
                    use_teff_for_tdb=False,
                    use_teff_for_tr=False,
                )
                proposed_pmv_debug_df, _ = calculate_pmv_series(
                    minute_df,
                    clo=clo_base,
                    use_teff_for_tdb=True,
                    use_teff_for_tr=True,
                )
                pmv_debug_df = pd.concat([
                    _build_pmv_input_debug_df(
                        minute_df,
                        conventional_pmv_debug_df,
                        "従来手法",
                        "estimated_temp",
                        clo_base,
                    ),
                    _build_pmv_input_debug_df(
                        minute_df,
                        proposed_pmv_debug_df,
                        "提案手法",
                        "teff",
                        clo_base,
                    ),
                ], ignore_index=True)
                st.dataframe(pmv_debug_df, use_container_width=True, hide_index=True)

            with st.expander("CLO探索デバッグ情報", expanded=True):
                st.dataframe(debug_df, use_container_width=True, hide_index=True)
                st.markdown("#### 探索条件")
                st.json({
                    "clo_min": clo_min_eval,
                    "clo_max": clo_max_eval,
                    "clo_step": clo_step_eval,
                    "target_pmv": 0.0,
                    "従来手法_best_clo": recommendation_comparison["conventional"]["best_clo"],
                    "従来手法_best_pmv": recommendation_comparison["conventional"]["best_pmv"],
                    "従来手法_best_error": recommendation_comparison["conventional"]["best_error"],
                    "提案手法_best_clo": recommendation_comparison["proposed"]["best_clo"],
                    "提案手法_best_pmv": recommendation_comparison["proposed"]["best_pmv"],
                    "提案手法_best_error": recommendation_comparison["proposed"]["best_error"],
                })
                st.markdown("#### 従来手法 探索結果")
                st.dataframe(_summarize_search_results(recommendation_comparison["conventional"]["search_results"]), use_container_width=True, hide_index=True)
                st.markdown("#### 提案手法 探索結果")
                st.dataframe(_summarize_search_results(recommendation_comparison["proposed"]["search_results"]), use_container_width=True, hide_index=True)

            d1, d2 = st.columns(2)
            d1.metric("CLO差分（提案 - 従来）", f"{recommendation_comparison['clo_diff']:+.2f}")
            d2.metric("厚着を推薦した手法", recommendation_comparison["heavier_method"])

            if recommendation_comparison["heavier_method"] == "同程度":
                st.info("従来手法と提案手法で推奨CLOは同程度でした。今回の予定条件では、温熱状態の時間遅れが推奨CLOに与える影響は小さいと考えられます。")
            else:
                st.info(
                    f"環境遷移を考慮した結果、{recommendation_comparison['heavier_method']}の方が厚い服装を推薦しました。"
                    f"CLO差分は {recommendation_comparison['clo_diff']:+.2f} clo です。"
                )

            st.markdown("### 2. PMV判定誤差")
            st.caption("評価条件: 提案手法で推薦されたCLO値を共通に使用し、入力温度のみを Tenv / Teff で変更します。")
            st.metric("評価に使用したCLO値（CLO_base）", f"{clo_base:.2f}")
            transition_df, transition_metrics = compare_pmv_transition(
                minute_df,
                fixed_clo=clo_base,
                comfort_lower=-0.5,
                comfort_upper=0.5,
            )

            valid_transition_df = transition_df.dropna(subset=["pmv_conventional", "pmv_proposed"]).copy()
            valid_transition_df["temp_diff_teff_tenv"] = (
                pd.to_numeric(valid_transition_df["teff"], errors="coerce")
                - pd.to_numeric(valid_transition_df["estimated_temp"], errors="coerce")
            )
            mean_conventional_pmv = float(valid_transition_df["pmv_conventional"].mean())
            mean_proposed_pmv = float(valid_transition_df["pmv_proposed"].mean())
            final_conventional_pmv = float(valid_transition_df["pmv_conventional"].iloc[-1])
            final_proposed_pmv = float(valid_transition_df["pmv_proposed"].iloc[-1])
            final_pmv_diff = final_proposed_pmv - final_conventional_pmv
            mean_temp_diff = float(valid_transition_df["temp_diff_teff_tenv"].mean())
            max_abs_temp_diff = float(valid_transition_df["temp_diff_teff_tenv"].abs().max())

            if max_abs_temp_diff < 0.001:
                st.info("Tenv と Teff がほぼ同じため、PMV差分も0に近くなっています。予定の環境遷移、ΔT、Teff計算結果を確認してください。")
            elif transition_metrics["max_abs_pmv_diff"] < 0.001:
                st.info("Tenv と Teff に差はありますが、PMV差分が非常に小さいため、丸め表示では0.000に見えています。")

            plot_df = transition_df[["datetime", "pmv_conventional", "pmv_proposed"]].set_index("datetime")
            plot_df = plot_df.rename(columns={
                "pmv_conventional": "従来手法PMV（Tenv）",
                "pmv_proposed": "提案手法PMV（Teff）",
            })
            st.line_chart(plot_df)

            diff_plot_df = transition_df[["datetime", "pmv_diff"]].set_index("datetime").rename(columns={
                "pmv_diff": "PMV差分（提案 - 従来）"
            })
            st.line_chart(diff_plot_df)

            st.markdown("#### 最終時刻のPMV")
            f1, f2, f3 = st.columns(3)
            f1.metric("従来手法PMV（最終）", f"{final_conventional_pmv:+.4f}")
            f2.metric("提案手法PMV（最終）", f"{final_proposed_pmv:+.4f}")
            f3.metric("PMV差（最終）", f"{final_pmv_diff:+.4f}")

            st.markdown("#### 一日平均のPMV")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("従来手法PMV平均", f"{mean_conventional_pmv:+.4f}")
            m2.metric("提案手法PMV平均", f"{mean_proposed_pmv:+.4f}")
            m3.metric("平均PMV差", f"{transition_metrics['mean_pmv_diff']:+.4f}")
            m4.metric("最大PMV差", f"{transition_metrics['max_abs_pmv_diff']:.4f}")
            m5.metric(
                "快適範囲内割合",
                f"{transition_metrics['conventional_comfort_ratio']:.1f} / {transition_metrics['proposed_comfort_ratio']:.1f} %",
                help="左が従来手法、右が提案手法です。",
            )
            t1, t2 = st.columns(2)
            t1.metric("平均温度差（Teff - Tenv）", f"{mean_temp_diff:+.3f} ℃")
            t2.metric("最大温度差（絶対値）", f"{max_abs_temp_diff:.3f} ℃")

            st.markdown("### 自動考察")
            st.write(
                f"・PMV判定誤差では、提案手法で推薦された CLO_base={clo_base:.2f} を両手法で共通に使用しています。"
            )
            st.write(
                f"・最終時刻のPMV差は {final_pmv_diff:+.3f}、一日平均PMV差は {transition_metrics['mean_pmv_diff']:+.3f} でした。"
            )
            st.write(
                "・これは、PMVへ入力する温度を環境温度そのものではなく、人体温熱状態の時間遅れを反映した Teff に変更したためと考えられます。"
            )

            with st.expander("比較データを確認する"):
                transition_df["temp_diff_teff_tenv"] = (
                    pd.to_numeric(transition_df["teff"], errors="coerce")
                    - pd.to_numeric(transition_df["estimated_temp"], errors="coerce")
                )
                display_eval_cols = [
                    "datetime", "estimated_temp", "teff", "temp_diff_teff_tenv", "rh", "v", "met",
                    "pmv_conventional", "pmv_proposed", "pmv_diff"
                ]
                st.dataframe(transition_df[display_eval_cols].head(500), use_container_width=True)
        except Exception as exc:
            st.warning(missing_message)
            st.error(str(exc))
