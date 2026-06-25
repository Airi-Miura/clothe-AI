from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model import (
    ACTIVITY_OPTIONS,
    CLO_DB,
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
    classify_clothing_categories_from_image,
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
    st.session_state.closet_items = []

if "closet_selected_categories" not in st.session_state:
    st.session_state.closet_selected_categories = []

if "detected_clothing_categories" not in st.session_state:
    st.session_state.detected_clothing_categories = []

if "gemini_detection_message" not in st.session_state:
    st.session_state.gemini_detection_message = None


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 主要移動の入力",
    "② 地点マスタ",
    "③ 推定結果",
    "④ 研究者向け可視化",
    "⑤ クローゼット登録",
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

    # Apply the pending Gemini result before the multiselect widget is created.
    if st.session_state.detected_clothing_categories:
        st.session_state.closet_selected_categories = st.session_state.detected_clothing_categories
        st.session_state.detected_clothing_categories = []

    with st.form("closet_register_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "服画像",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
        item_name = st.text_input("登録名")
        selected_categories = st.multiselect(
            "衣服カテゴリ",
            options=list(CLO_DB.keys()),
            key="closet_selected_categories",
        )

        clothing_categories = list(selected_categories)
        total_clo = calculate_total_clo(clothing_categories)
        st.metric("総CLO値", f"{total_clo:.2f}")

        if st.session_state.gemini_detection_message:
            st.info(st.session_state.gemini_detection_message)

        auto_detected = st.form_submit_button("画像から自動判定")
        submitted = st.form_submit_button("登録する")

        if auto_detected:
            if uploaded_file is None:
                st.error("服画像をアップロードしてください。")
            else:
                try:
                    detected_categories, _ = classify_clothing_categories_from_image(
                        uploaded_file.getvalue(),
                        mime_type=uploaded_file.type,
                    )
                    st.session_state.detected_clothing_categories = detected_categories
                    if detected_categories:
                        st.session_state.gemini_detection_message = "Gemini判定: " + "、".join(detected_categories)
                    else:
                        st.session_state.gemini_detection_message = "Geminiで該当カテゴリを判定できませんでした。手動で選択してください。"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        if submitted:
            if uploaded_file is None:
                st.error("服画像をアップロードしてください。")
            elif not item_name.strip():
                st.error("登録名を入力してください。")
            elif not clothing_categories:
                st.error("衣服カテゴリを1つ以上選択してください。")
            else:
                st.session_state.closet_items.append({
                    "name": item_name.strip(),
                    "image": uploaded_file.getvalue(),
                    "image_name": uploaded_file.name,
                    "image_type": uploaded_file.type,
                    "categories": clothing_categories,
                    "total_clo": total_clo,
                })
                st.success("クローゼットに登録しました。")
    st.markdown("### 登録済みクローゼット一覧")
    if not st.session_state.closet_items:
        st.info("登録済みの服はまだありません。")
    else:
        for i, item in enumerate(st.session_state.closet_items):
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1.2, 1.4, 2.4, 1.0, 0.9])
                with c1:
                    st.image(item["image"], caption=item.get("image_name", "服画像"), use_container_width=True)
                with c2:
                    st.write(item["name"])
                with c3:
                    st.write("、".join(item["categories"]))
                with c4:
                    st.metric("総CLO", f"{float(item['total_clo']):.2f}")
                with c5:
                    if st.button("削除", key=f"delete_closet_item_{i}", use_container_width=True):
                        st.session_state.closet_items.pop(i)
                        st.rerun()
