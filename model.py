from __future__ import annotations

import math
import urllib.parse
import urllib.request
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

SEASON_PRESETS = {
    "spring": {
        "outdoor_min": 10.0,
        "outdoor_max": 20.0,
        "outdoor_rh": 55.0,
        "outdoor_v": 0.30,
        "indoor_home": 22.0,
        "indoor_school": 22.0,
        "indoor_office": 22.0,
        "indoor_shop": 23.0,
        "train_temp": 22.0,
        "car_temp": 22.0,
        "other_vehicle_temp": 22.0,
        "indoor_rh": 50.0,
        "train_rh": 50.0,
        "car_rh": 50.0,
        "other_vehicle_rh": 50.0,
        "indoor_v": 0.10,
        "train_v": 0.10,
        "car_v": 0.10,
        "other_vehicle_v": 0.10,
    },
    "summer": {
        "outdoor_min": 25.0,
        "outdoor_max": 34.0,
        "outdoor_rh": 65.0,
        "outdoor_v": 0.30,
        "indoor_home": 27.0,
        "indoor_school": 25.0,
        "indoor_office": 24.5,
        "indoor_shop": 24.0,
        "train_temp": 25.0,
        "car_temp": 24.0,
        "other_vehicle_temp": 26.0,
        "indoor_rh": 50.0,
        "train_rh": 50.0,
        "car_rh": 50.0,
        "other_vehicle_rh": 50.0,
        "indoor_v": 0.10,
        "train_v": 0.10,
        "car_v": 0.10,
        "other_vehicle_v": 0.10,
    },
    "autumn": {
        "outdoor_min": 12.0,
        "outdoor_max": 22.0,
        "outdoor_rh": 60.0,
        "outdoor_v": 0.25,
        "indoor_home": 23.0,
        "indoor_school": 23.0,
        "indoor_office": 23.0,
        "indoor_shop": 24.0,
        "train_temp": 23.0,
        "car_temp": 23.0,
        "other_vehicle_temp": 23.0,
        "indoor_rh": 50.0,
        "train_rh": 50.0,
        "car_rh": 50.0,
        "other_vehicle_rh": 50.0,
        "indoor_v": 0.10,
        "train_v": 0.10,
        "car_v": 0.10,
        "other_vehicle_v": 0.10,
    },
    "winter": {
        "outdoor_min": 2.0,
        "outdoor_max": 12.0,
        "outdoor_rh": 45.0,
        "outdoor_v": 0.25,
        "indoor_home": 21.0,
        "indoor_school": 22.0,
        "indoor_office": 22.0,
        "indoor_shop": 23.0,
        "train_temp": 21.0,
        "car_temp": 20.0,
        "other_vehicle_temp": 19.0,
        "indoor_rh": 40.0,
        "train_rh": 40.0,
        "car_rh": 40.0,
        "other_vehicle_rh": 40.0,
        "indoor_v": 0.10,
        "train_v": 0.10,
        "car_v": 0.10,
        "other_vehicle_v": 0.10,
    },
}

TRANSPORT_OPTIONS = ["徒歩", "電車", "車", "その他"]

ENVIRONMENT_OPTIONS = [
    "屋外",
    "自宅(屋内)",
    "学校(屋内)",
    "オフィス(屋内)",
    "店舗(屋内)",
    "電車内",
    "車内",
    "その他乗り物内",
]

ACTIVITY_OPTIONS = ["座位", "立位", "徒歩", "軽い運動", "強い運動"]

THERMAL_TYPE_OPTIONS = ["寒がり", "やや寒がり", "普通", "やや暑がり", "暑がり"]

THERMAL_TYPE_TO_OFFSET = {
    "寒がり": 0.20,
    "やや寒がり": 0.10,
    "普通": 0.00,
    "やや暑がり": -0.10,
    "暑がり": -0.20,
}

FEEDBACK_OPTIONS = ["寒かった", "少し寒かった", "ちょうどよかった", "少し暑かった", "暑かった"]

FEEDBACK_SCORE_MAP = {
    "寒かった": -2,
    "少し寒かった": -1,
    "ちょうどよかった": 0,
    "少し暑かった": 1,
    "暑かった": 2,
}

PERSONAL_CLO_OFFSET_MIN = -0.5
PERSONAL_CLO_OFFSET_MAX = 0.5
CLO_DB = {
    "半袖Tシャツ": 0.08,
    "長袖Tシャツ": 0.20,
    "シャツ": 0.25,
    "ブラウス": 0.20,
    "セーター": 0.30,
    "カーディガン": 0.30,
    "パーカー": 0.35,
    "フリース": 0.40,
    "薄手ジャケット": 0.35,
    "厚手ジャケット": 0.60,
    "コート": 0.70,
    "ダウンジャケット": 0.80,
    "ジーンズ": 0.25,
    "スラックス": 0.25,
    "ショートパンツ": 0.10,
    "スカート": 0.15,
    "ワンピース": 0.35,
}
CLO_DICT = CLO_DB


def get_clo_value(category: str) -> float:
    return float(CLO_DB.get(category, 0.0))


def calculate_total_clo(categories: list[str]) -> float:
    return round(sum(get_clo_value(category) for category in categories), 2)


def get_initial_personal_clo_offset(thermal_type: str) -> float:
    return float(THERMAL_TYPE_TO_OFFSET.get(thermal_type, 0.0))


def update_personal_clo_offset(old_offset: float, feedback_score: float, learning_rate: float = 0.05) -> float:
    new_offset = float(old_offset) - float(learning_rate) * float(feedback_score)
    if new_offset < PERSONAL_CLO_OFFSET_MIN:
        new_offset = PERSONAL_CLO_OFFSET_MIN
    if new_offset > PERSONAL_CLO_OFFSET_MAX:
        new_offset = PERSONAL_CLO_OFFSET_MAX
    return float(new_offset)


def get_recommendation_message(offset: float) -> str:
    if offset > 0.1:
        return "寒がり傾向があるため、標準よりやや暖かめの服装を推薦します。"
    if offset < -0.1:
        return "暑がり傾向があるため、標準よりやや軽めの服装を推薦します。"
    return "標準的な体感として服装を推薦します。"


DELTA_T_COEFFS = {
    "in_to_out": {"a": -0.185750, "b": 7.741123},
    "out_to_in": {"a": 0.212578, "b": -0.925903},
}


def calculate_delta_t(elapsed_min: float, transition_direction: Optional[str]) -> float:
    if transition_direction not in DELTA_T_COEFFS:
        return 0.0
    coef = DELTA_T_COEFFS[transition_direction]
    return float(coef["a"]) * float(elapsed_min) + float(coef["b"])


def _is_outdoor_env(environment: str) -> bool:
    return str(environment) == "屋外"


def parse_hhmm(value: str):
    try:
        return datetime.strptime(str(value).strip(), "%H:%M").time()
    except Exception:
        return None


def combine_date_and_time(base_date, t):
    return datetime.combine(base_date, t)


def minutes_between(start_dt: datetime, end_dt: datetime) -> float:
    return (end_dt - start_dt).total_seconds() / 60.0


def safe_round(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


def outdoor_temp_by_hour(hour_float: float, season: str, preset: dict) -> float:
    t_min = preset["outdoor_min"]
    t_max = preset["outdoor_max"]
    avg = (t_min + t_max) / 2
    amp = (t_max - t_min) / 2
    value = avg + amp * math.cos(2 * math.pi * (hour_float - 14) / 24)
    return safe_round(value, 1)


def get_environment_temperature(hour_float: float, season: str, environment: str, preset: dict) -> float:
    if environment == "屋外":
        return outdoor_temp_by_hour(hour_float, season, preset)
    if environment == "自宅(屋内)":
        return preset["indoor_home"]
    if environment == "学校(屋内)":
        return preset["indoor_school"]
    if environment == "オフィス(屋内)":
        return preset["indoor_office"]
    if environment == "店舗(屋内)":
        return preset["indoor_shop"]
    if environment == "電車内":
        return preset["train_temp"]
    if environment == "車内":
        return preset["car_temp"]
    if environment == "その他乗り物内":
        return preset["other_vehicle_temp"]
    return outdoor_temp_by_hour(hour_float, season, preset)


def get_environment_rh(environment: str, preset: dict) -> float:
    if environment == "屋外":
        return preset["outdoor_rh"]
    if environment in ["自宅(屋内)", "学校(屋内)", "オフィス(屋内)", "店舗(屋内)"]:
        return preset["indoor_rh"]
    if environment == "電車内":
        return preset["train_rh"]
    if environment == "車内":
        return preset["car_rh"]
    if environment == "その他乗り物内":
        return preset["other_vehicle_rh"]
    return preset["indoor_rh"]


def get_environment_v(environment: str, preset: dict) -> float:
    if environment == "屋外":
        return preset["outdoor_v"]
    if environment in ["自宅(屋内)", "学校(屋内)", "オフィス(屋内)", "店舗(屋内)"]:
        return preset["indoor_v"]
    if environment == "電車内":
        return preset["train_v"]
    if environment == "車内":
        return preset["car_v"]
    if environment == "その他乗り物内":
        return preset["other_vehicle_v"]
    return preset["indoor_v"]


def activity_to_met(activity: str) -> float:
    mapping = {
        "座位": 1.0,
        "立位": 1.2,
        "徒歩": 2.0,
        "軽い運動": 3.0,
        "強い運動": 4.0,
    }
    return mapping.get(activity, 1.0)


def judge_cz_by_pmv(pmv_value: Optional[float], lower: float = -0.5, upper: float = 0.5) -> str:
    if pmv_value is None or pd.isna(pmv_value):
        return "不明"
    if pmv_value < lower:
        return "寒い"
    if pmv_value > upper:
        return "暑い"
    return "快適"


def infer_environment_from_place(place_name: str, places: dict) -> str:
    info = places.get(place_name, {})
    default_env = info.get("default_env", "屋外")

    if default_env == "屋外":
        return "屋外"

    lowered = str(place_name).lower()
    if "家" in place_name or "自宅" in place_name or "home" in lowered:
        return "自宅(屋内)"
    if "学校" in place_name or "大学" in place_name or "campus" in lowered:
        return "学校(屋内)"
    if "店" in place_name or "shop" in lowered or "mall" in lowered:
        return "店舗(屋内)"
    if "会社" in place_name or "office" in lowered:
        return "オフィス(屋内)"

    return "オフィス(屋内)"


def get_place_coords(place_name: str, places: dict) -> Tuple[Optional[float], Optional[float]]:
    info = places.get(place_name)
    if info:
        return info.get("lat"), info.get("lon")
    return None, None


def validate_trip_df(df: pd.DataFrame):
    errors = []

    required_cols = [
        "depart_time",
        "arrive_time",
        "origin",
        "destination",
        "transport",
        "destination_environment",
        "destination_activity",
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"列 {col} がありません。")
            return errors

    parsed_rows = []
    for i, row in df.iterrows():
        depart_t = parse_hhmm(row["depart_time"])
        arrive_t = parse_hhmm(row["arrive_time"])

        if depart_t is None:
            errors.append(f"{i + 1}行目: depart_time は HH:MM 形式で入力してください。")
        if arrive_t is None:
            errors.append(f"{i + 1}行目: arrive_time は HH:MM 形式で入力してください。")
        if depart_t and arrive_t and arrive_t <= depart_t:
            errors.append(f"{i + 1}行目: arrive_time は depart_time より後にしてください。")

        origin = str(row["origin"]).strip()
        destination = str(row["destination"]).strip()
        if not origin:
            errors.append(f"{i + 1}行目: origin が空です。")
        if not destination:
            errors.append(f"{i + 1}行目: destination が空です。")

        parsed_rows.append((i, depart_t, arrive_t))

    valid_rows = [r for r in parsed_rows if r[1] is not None and r[2] is not None]
    for idx in range(1, len(valid_rows)):
        _, _, prev_arrive = valid_rows[idx - 1]
        cur_i, cur_depart, _ = valid_rows[idx]

        if cur_depart < valid_rows[idx - 1][1]:
            errors.append(f"{cur_i + 1}行目: depart_time が前の移動より早いです。時間順に並べてください。")
        if cur_depart < prev_arrive:
            errors.append(f"{cur_i + 1}行目: depart_time が前の arrive_time より前です。移動が重なっています。")

    return errors


def _mid_hour(start_dt: datetime, end_dt: datetime) -> float:
    mid = start_dt + (end_dt - start_dt) / 2
    return mid.hour + mid.minute / 60


def _make_segment(
    segment_type: str,
    label: str,
    start_dt: datetime,
    end_dt: datetime,
    place_name: str,
    environment: str,
    transport: str,
    activity: str,
    season: str,
    preset: dict,
    notes: str = "",
):
    duration_min = minutes_between(start_dt, end_dt)
    hour_float = _mid_hour(start_dt, end_dt)
    estimated_temp = get_environment_temperature(hour_float, season, environment, preset)
    rh = get_environment_rh(environment, preset)
    v = get_environment_v(environment, preset)
    met = activity_to_met(activity)

    return {
        "segment_type": segment_type,
        "label": label,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "duration_min": safe_round(duration_min, 1),
        "place_name": place_name,
        "environment": environment,
        "transport": transport,
        "activity": activity,
        "estimated_temp": estimated_temp,
        "rh": rh,
        "v": v,
        "met": met,
        "notes": notes,
    }


def split_transport_segments(
    origin: str,
    destination: str,
    depart_dt: datetime,
    arrive_dt: datetime,
    transport: str,
    season: str,
    preset: dict,
    walk_buffer_min: int = 10,
):
    total_min = minutes_between(depart_dt, arrive_dt)
    if total_min <= 0:
        return []

    if transport == "徒歩":
        return [
            _make_segment(
                "移動", f"{origin}→{destination}（徒歩）",
                depart_dt, arrive_dt,
                f"{origin}→{destination}", "屋外", "徒歩", "徒歩",
                season, preset
            )
        ]

    if transport == "車":
        return [
            _make_segment(
                "移動", f"{origin}→{destination}（車）",
                depart_dt, arrive_dt,
                "車内", "車内", "車", "座位",
                season, preset
            )
        ]

    if transport == "その他":
        return [
            _make_segment(
                "移動", f"{origin}→{destination}（その他）",
                depart_dt, arrive_dt,
                "その他乗り物内", "その他乗り物内", "その他", "座位",
                season, preset
            )
        ]

    if total_min >= (walk_buffer_min * 2 + 1):
        walk_before = walk_buffer_min
        walk_after = walk_buffer_min
        train_min = total_min - walk_before - walk_after
    else:
        walk_before = max(int(total_min * 0.25), 1)
        walk_after = max(int(total_min * 0.25), 1)
        train_min = total_min - walk_before - walk_after
        if train_min < 1:
            train_min = 1
            remainder = total_min - train_min
            walk_before = max(remainder / 2, 0)
            walk_after = max(remainder - walk_before, 0)

    t1_end = depart_dt + timedelta(minutes=walk_before)
    t2_end = t1_end + timedelta(minutes=train_min)

    segments = []

    if walk_before > 0:
        segments.append(
            _make_segment(
                "移動", f"{origin}→駅（徒歩）",
                depart_dt, t1_end,
                f"{origin}周辺", "屋外", "徒歩", "徒歩",
                season, preset
            )
        )
    if train_min > 0:
        segments.append(
            _make_segment(
                "移動", "駅→駅（電車）",
                t1_end, t2_end,
                "電車内", "電車内", "電車", "座位",
                season, preset
            )
        )
    if walk_after > 0:
        segments.append(
            _make_segment(
                "移動", f"駅→{destination}（徒歩）",
                t2_end, arrive_dt,
                f"{destination}周辺", "屋外", "徒歩", "徒歩",
                season, preset
            )
        )

    return segments


def build_daily_segments_from_trips(
    trip_df: pd.DataFrame,
    target_date,
    season: str,
    preset: dict,
    day_start_time: str,
    day_end_time: str,
    start_place: str,
    start_environment: str,
    start_activity: str,
    end_place: Optional[str] = None,
    end_environment: Optional[str] = None,
    end_activity: Optional[str] = None,
    walk_buffer_min: int = 10,
):
    trip_df = trip_df.copy()
    trip_df["sort_key"] = trip_df["depart_time"].astype(str)
    trip_df = trip_df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    rows = []

    day_start_t = parse_hhmm(day_start_time)
    day_end_t = parse_hhmm(day_end_time)
    if day_start_t is None or day_end_t is None:
        raise ValueError("day_start_time / day_end_time は HH:MM 形式で指定してください。")

    current_place = start_place
    current_environment = start_environment
    current_activity = start_activity
    current_dt = combine_date_and_time(target_date, day_start_t)

    for _, row in trip_df.iterrows():
        depart_t = parse_hhmm(row["depart_time"])
        arrive_t = parse_hhmm(row["arrive_time"])

        depart_dt = combine_date_and_time(target_date, depart_t)
        arrive_dt = combine_date_and_time(target_date, arrive_t)

        origin = str(row["origin"]).strip()
        destination = str(row["destination"]).strip()
        transport = str(row["transport"]).strip()
        destination_environment = str(row["destination_environment"]).strip()
        destination_activity = str(row["destination_activity"]).strip()
        notes = str(row.get("notes", "")).strip()

        if depart_dt > current_dt:
            rows.append(
                _make_segment(
                    "滞在", f"{current_place}（滞在）",
                    current_dt, depart_dt,
                    current_place, current_environment, "なし", current_activity,
                    season, preset
                )
            )

        current_place = origin

        move_segments = split_transport_segments(
            origin=origin,
            destination=destination,
            depart_dt=depart_dt,
            arrive_dt=arrive_dt,
            transport=transport,
            season=season,
            preset=preset,
            walk_buffer_min=walk_buffer_min,
        )

        if move_segments:
            move_segments[-1]["notes"] = notes

        rows.extend(move_segments)

        current_place = destination
        current_environment = destination_environment
        current_activity = destination_activity
        current_dt = arrive_dt

    final_place = end_place if end_place is not None else current_place
    final_environment = end_environment if end_environment is not None else current_environment
    final_activity = end_activity if end_activity is not None else current_activity
    day_end_dt = combine_date_and_time(target_date, day_end_t)

    if day_end_dt > current_dt:
        rows.append(
            _make_segment(
                "滞在", f"{final_place}（滞在）",
                current_dt, day_end_dt,
                final_place, final_environment, "なし", final_activity,
                season, preset
            )
        )

    return pd.DataFrame(rows)


def expand_segments_to_time_level(segment_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in segment_df.iterrows():
        start_dt = row["start_dt"]
        end_dt = row["end_dt"]
        current = start_dt

        while current < end_dt:
            rows.append({
                "datetime": current,
                "minute_of_day": current.hour * 60 + current.minute,
                "segment_type": row["segment_type"],
                "label": row["label"],
                "environment": row["environment"],
                "transport": row["transport"],
                "activity": row["activity"],
                "estimated_temp": row["estimated_temp"],
                "rh": row["rh"],
                "v": row["v"],
                "met": row["met"],
                "notes": row["notes"],
            })
            current += timedelta(minutes=5)

    return pd.DataFrame(rows)


def fetch_open_meteo_hourly_weather(lat: float, lon: float, target_date) -> pd.DataFrame:
    """
    Open-Meteo から指定地点・指定日の時間別予報を取得する。
    返す列:
      - datetime_hour
      - temperature_2m
      - relative_humidity_2m
      - wind_speed_10m
    """
    date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Tokyo",
        "start_date": date_str,
        "end_date": date_str,
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "streamlit-cz-app/1.0"})
    with urllib.request.urlopen(req, timeout=20) as res:
        raw = res.read().decode("utf-8")
        data = json.loads(raw)

    hourly = data.get("hourly", {})
    if not hourly:
        raise ValueError("Open-Meteo から hourly データを取得できませんでした。")

    df = pd.DataFrame({
        "datetime_hour": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "wind_speed_10m": hourly["wind_speed_10m"],
    })
    return df


def apply_open_meteo_to_outdoor_segments(
    minute_df: pd.DataFrame,
    places: dict,
    trip_df: pd.DataFrame,
    target_date,
):
    """
    屋外区間だけ Open-Meteo の実予報に置き換える。
    方針:
      - 屋外の「滞在」区間は、その区間 label から place_name を使って地点特定
      - 屋外の「移動」区間は、origin/destination から地点を決めたいが、
        まずは origin 側の座標を優先して使う
      - 取得失敗時は元の代表値をそのまま残す
    """
    df = minute_df.copy()

    trip_lookup = []
    for _, row in trip_df.iterrows():
        trip_lookup.append({
            "origin": str(row["origin"]).strip(),
            "destination": str(row["destination"]).strip(),
            "depart_time": str(row["depart_time"]).strip(),
            "arrive_time": str(row["arrive_time"]).strip(),
        })

    weather_cache = {}

    def get_weather_for_place(place_name: str) -> Optional[pd.DataFrame]:
        lat, lon = get_place_coords(place_name, places)
        if lat is None or lon is None:
            return None
        cache_key = (place_name, float(lat), float(lon), str(target_date))
        if cache_key not in weather_cache:
            try:
                weather_cache[cache_key] = fetch_open_meteo_hourly_weather(lat, lon, target_date)
            except Exception:
                weather_cache[cache_key] = None
        return weather_cache[cache_key]

    def match_weather_at_time(weather_df: pd.DataFrame, dt: pd.Timestamp):
        if weather_df is None or weather_df.empty:
            return None
        hour_dt = dt.floor("H")
        matched = weather_df.loc[weather_df["datetime_hour"] == hour_dt]
        if matched.empty:
            return None
        return matched.iloc[0]

    def infer_place_for_row(row):
        label = str(row["label"])

        if label.endswith("（滞在）"):
            base_place = label.replace("（滞在）", "")
            if base_place in places:
                return base_place

        if "→駅（徒歩）" in label:
            origin = label.split("→駅（徒歩）")[0]
            if origin in places:
                return origin

        if "駅→" in label and "（徒歩）" in label:
            dest = label.split("駅→", 1)[1].replace("（徒歩）", "")
            if dest in places:
                return dest

        return None

    outdoor_mask = df["environment"] == "屋外"

    for idx in df.index[outdoor_mask]:
        row = df.loc[idx]
        place_name = infer_place_for_row(row)
        if place_name is None:
            continue

        weather_df = get_weather_for_place(place_name)
        matched = match_weather_at_time(weather_df, pd.to_datetime(row["datetime"]))
        if matched is None:
            continue

        df.at[idx, "estimated_temp"] = float(matched["temperature_2m"])
        df.at[idx, "rh"] = float(matched["relative_humidity_2m"])
        df.at[idx, "v"] = float(matched["wind_speed_10m"])

    return df


def apply_first_order_lag_to_minutes(minute_df: pd.DataFrame, alpha: float, dt_min: float = 1.0) -> pd.DataFrame:
    df = minute_df.copy().reset_index(drop=True)

    if df.empty:
        df["teff"] = []
        df["delta_t"] = []
        df["transition_direction"] = []
        df["transition_elapsed_min"] = []
        return df

    n = len(df)
    transition_direction = [None] * n
    transition_elapsed_min = [0.0] * n
    delta_t_values = [0.0] * n

    current_direction: Optional[str] = None
    transition_start_dt = None

    for i in range(n):
        if i > 0:
            prev_outdoor = _is_outdoor_env(df.loc[i - 1, "environment"])
            cur_outdoor = _is_outdoor_env(df.loc[i, "environment"])
            if not prev_outdoor and cur_outdoor:
                current_direction = "in_to_out"
                transition_start_dt = df.loc[i, "datetime"]
            elif prev_outdoor and not cur_outdoor:
                current_direction = "out_to_in"
                transition_start_dt = df.loc[i, "datetime"]

        if current_direction is not None and transition_start_dt is not None:
            elapsed = (df.loc[i, "datetime"] - transition_start_dt).total_seconds() / 60.0
            if elapsed < 0:
                elapsed = 0.0
        else:
            elapsed = 0.0

        transition_direction[i] = current_direction
        transition_elapsed_min[i] = elapsed
        delta_t_values[i] = calculate_delta_t(elapsed, current_direction)

    teff_values = [float(df.loc[0, "estimated_temp"])]
    for i in range(1, n):
        prev_teff = teff_values[-1]
        current_tenv = float(df.loc[i - 1, "estimated_temp"])
        dt_val = delta_t_values[i - 1]
        next_teff = prev_teff + alpha * ((current_tenv + dt_val) - prev_teff) * float(dt_min)
        teff_values.append(next_teff)

    df["teff"] = [round(float(v), 3) for v in teff_values]
    df["delta_t"] = [round(float(v), 4) for v in delta_t_values]
    df["transition_direction"] = transition_direction
    df["transition_elapsed_min"] = [round(float(v), 2) for v in transition_elapsed_min]
    return df


def calculate_pmv_series(
    minute_df: pd.DataFrame,
    clo: float,
    use_teff_for_tdb: bool = True,
    use_teff_for_tr: bool = False,
):
    try:
        from pythermalcomfort.models import pmv_ppd_iso
    except Exception:
        df = minute_df.copy()
        df["clo"] = clo
        df["pmv"] = None
        return df, False

    df = minute_df.copy()
    df["clo"] = clo

    pmv_values = []

    for _, row in df.iterrows():
        try:
            tdb_input = float(row["teff"]) if use_teff_for_tdb else float(row["estimated_temp"])
            tr_input = float(row["teff"]) if use_teff_for_tr else float(row["estimated_temp"])

            res = pmv_ppd_iso(
                tdb=tdb_input,
                tr=tr_input,
                vr=float(row["v"]),
                rh=float(row["rh"]),
                met=float(row["met"]),
                clo=float(clo),
                wme=0,
                limit_inputs=False,
            )
            pmv = getattr(res, "pmv", None)
            if pmv is None and isinstance(res, dict):
                pmv = res.get("pmv")
            pmv_values.append(pmv)
        except Exception:
            pmv_values.append(None)

    df["pmv"] = pmv_values
    return df, True


def apply_cz_from_pmv(minute_df: pd.DataFrame, pmv_lower: float = -0.5, pmv_upper: float = 0.5) -> pd.DataFrame:
    df = minute_df.copy()
    df["pmv_lower"] = pmv_lower
    df["pmv_upper"] = pmv_upper
    df["cz_result"] = df["pmv"].apply(lambda x: judge_cz_by_pmv(x, pmv_lower, pmv_upper))
    return df


def summarize_results(minute_df: pd.DataFrame):
    if minute_df.empty:
        return {
            "mean_temp": None,
            "mean_teff": None,
            "comfort_ratio": None,
            "cold_ratio": None,
            "hot_ratio": None,
            "mean_pmv": None,
        }

    total = len(minute_df)
    comfort = (minute_df["cz_result"] == "快適").sum()
    cold = (minute_df["cz_result"] == "寒い").sum()
    hot = (minute_df["cz_result"] == "暑い").sum()

    mean_pmv = None
    if "pmv" in minute_df.columns and minute_df["pmv"].notna().any():
        mean_pmv = round(float(minute_df["pmv"].dropna().mean()), 3)

    return {
        "mean_temp": round(float(minute_df["estimated_temp"].mean()), 1),
        "mean_teff": round(float(minute_df["teff"].mean()), 1) if "teff" in minute_df.columns else None,
        "comfort_ratio": round(comfort / total * 100, 1),
        "cold_ratio": round(cold / total * 100, 1),
        "hot_ratio": round(hot / total * 100, 1),
        "mean_pmv": mean_pmv,
    }


def aggregate_minutes_to_segments(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame()

    rows = []
    for label, g in minute_df.groupby("label", sort=False):
        row = {
            "label": label,
            "start_time": g["datetime"].iloc[0].strftime("%H:%M"),
            "end_time": (g["datetime"].iloc[-1] + timedelta(minutes=1)).strftime("%H:%M"),
            "duration_min": len(g),
            "environment": g["environment"].iloc[0],
            "transport": g["transport"].iloc[0],
            "activity": g["activity"].iloc[0],
            "mean_tenv": round(float(g["estimated_temp"].mean()), 2),
            "mean_teff": round(float(g["teff"].mean()), 2) if "teff" in g.columns else None,
            "mean_rh": round(float(g["rh"].mean()), 2),
            "mean_v": round(float(g["v"].mean()), 2),
            "mean_met": round(float(g["met"].mean()), 2),
            "clo": round(float(g["clo"].mean()), 2) if "clo" in g.columns else None,
            "mean_pmv": round(float(g["pmv"].dropna().mean()), 3) if "pmv" in g.columns and g["pmv"].notna().any() else None,
            "cz_result_majority": g["cz_result"].mode().iloc[0] if "cz_result" in g.columns and not g["cz_result"].mode().empty else None,
            "notes": g["notes"].iloc[0] if "notes" in g.columns else "",
        }
        rows.append(row)

    return pd.DataFrame(rows)