import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pymongo import MongoClient

def run_aggregator_from_file(config_path: str) -> pd.DataFrame:
    """
    Loads YAML config from disk, then calls run_aggregator().
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return run_aggregator(config)

def run_aggregator(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main aggregator entry point, reading config to:
      1) Connect to Mongo (with default db = 'EscapeBase' if not in config)
      2) Fetch from user-specified collections
      3) Unify duplicates
      4) Apply mapping & formulas
      5) Apply rolling logic if enabled
      6) Pick last N rows
      7) Output if requested
      8) Return final DataFrame
    """
    # --------------------------------------------------
    # 1) Basic Mongo info from config or fallback
    # --------------------------------------------------
    mongo_uri = config.get("mongodb_uri", "mongodb://localhost:27017")
    db_name   = config.get("mongo_database", "EscapeBase")  # default

    # The user can define specific collections in config, e.g.
    #   "collections": [ config["mongo_collection_raw"], config["mongo_collection_nhRaw"] ]
    # or pass them directly. We'll just read from 'collections' key.
    collections = config.get("collections", [])
    if not collections:
        # fallback if user didn't specify
        # for example, they might define 'mongo_collection_raw' in config
        # so let's gather them if present, else default to "Raw"
        fallback_col = config.get("mongo_collection_raw", "Raw")
        collections = [fallback_col]

    # Filters & aggregator settings
    filters  = config.get("filters", {})
    fields   = config.get("fields", [])
    mapping  = config.get("mapping", {})
    formulas = config.get("formulas", {})
    rolling  = config.get("rolling", {})
    output   = config.get("output", {})

    # --------------------------------------------------
    # 2) Fetch & unify data
    # --------------------------------------------------
    df = _fetch_data(mongo_uri, db_name, collections, filters, fields)
    if df.empty:
        return df  # no data

    df = _unify_duplicates(df)

    # --------------------------------------------------
    # 3) Apply mapping
    # --------------------------------------------------
    df = _apply_mapping(df, mapping)

    # --------------------------------------------------
    # 4) Apply formulas
    # --------------------------------------------------
    df = _apply_formulas(df, formulas)

    # --------------------------------------------------
    # 5) Rolling logic
    # --------------------------------------------------
    if rolling.get("enabled", False):
        df = _apply_rolling(df, rolling)

    # --------------------------------------------------
    # 6) Pick last N rows
    # --------------------------------------------------
    n_val = filters.get("n", 10)  # e.g., default to 10
    date_field = filters.get("date_field", "date")
    df = _pick_last_n(df, n_val, date_field)

    # --------------------------------------------------
    # Final sort
    # --------------------------------------------------
    if date_field in df.columns:
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce")
        df.sort_values(["symbol", date_field], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------
    # 7) Output if requested
    # --------------------------------------------------
    _handle_output(output, df, mongo_uri, db_name)

    return df


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def _fetch_data(mongo_uri: str, db_name: str, collections: List[str],
                filters: Dict[str,Any], fields: List[str]) -> pd.DataFrame:
    date_field = filters.get("date_field", "date")
    # build projection
    proj = {"_id":0}
    for must_have in [date_field, "symbol", "freq"]:
        proj[must_have] = 1
    for f in fields:
        proj[f] = 1

    query = _build_query(filters)

    all_frames = []
    with MongoClient(mongo_uri) as cli:
        db = cli[db_name]
        for coll in collections:
            docs = list(db[coll].find(query, proj))
            if docs:
                df_part = pd.DataFrame(docs)
                if not df_part.empty:
                    all_frames.append(df_part)
    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)

def _build_query(filters: Dict[str,Any]) -> dict:
    q = {}
    freq_ = filters.get("freq")
    if freq_:
        if isinstance(freq_, str):
            freq_ = [freq_]
        freq_ = [f.upper() for f in freq_]
        q["freq"] = {"$in": freq_}

    syms = filters.get("symbol")
    if syms:
        q["symbol"] = {"$in": syms}

    inds = filters.get("assigned_industry")
    if inds:
        q["assigned_industry"] = {"$in": inds}

    # date range, etc. can be extended here

    return q

def _unify_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = []
    for c in ["symbol","date","freq","period"]:
        if c in df.columns:
            group_cols.append(c)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def combine_rows(g: pd.DataFrame) -> pd.Series:
        row0 = g.iloc[0].copy()
        for col in numeric_cols:
            if pd.isna(row0[col]):
                for _, row in g.iterrows():
                    if pd.notnull(row[col]):
                        row0[col] = row[col]
                        break
        return row0

    out = df.groupby(group_cols, as_index=False).apply(combine_rows)
    out.reset_index(drop=True, inplace=True)
    return out

def _apply_mapping(df: pd.DataFrame, mapping: Dict[str,str]) -> pd.DataFrame:
    if df.empty or not mapping:
        return df
    rename_map = {}
    for old_col, new_col in mapping.items():
        if old_col in df.columns:
            rename_map[old_col] = new_col
    return df.rename(columns=rename_map)

def _apply_formulas(df: pd.DataFrame, formulas: Dict[str,str]) -> pd.DataFrame:
    if df.empty or not formulas:
        return df
    df = df.copy()
    for new_col, expr in formulas.items():
        tokens = _tokenize(expr)
        expr_parsed = expr
        for t in tokens:
            if t in df.columns:
                expr_parsed = expr_parsed.replace(t, f"df['{t}']")
        try:
            result = eval(expr_parsed, {"df": df, "np": np}, {})
            df[new_col] = result
        except:
            df[new_col] = np.nan
    return df

def _apply_rolling(df: pd.DataFrame, rolling: Dict[str,Any]) -> pd.DataFrame:
    """
    Example rolling config:
      rolling:
        enabled: true
        window: 4
        method: "mean"  # or "sum"
        columns: ["revenue","net_inc"]
    """
    if df.empty:
        return df

    df = df.copy()
    window = rolling.get("window", 4)
    method = rolling.get("method", "mean")
    cols   = rolling.get("columns", [])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["symbol","date"], ascending=True, inplace=True)

    out_frames = []
    for sym, grp in df.groupby("symbol", group_keys=False):
        grp = grp.sort_values("date")
        for c in cols:
            if c in grp.columns:
                roll_obj = grp[c].rolling(window=window, min_periods=1)
                if method == "sum":
                    grp[f"{c}_roll{window}"] = roll_obj.sum()
                else:
                    grp[f"{c}_roll{window}"] = roll_obj.mean()
        out_frames.append(grp)

    new_df = pd.concat(out_frames, ignore_index=True)
    new_df.sort_values(["symbol","date"], ascending=True, inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def _pick_last_n(df: pd.DataFrame, n: int, date_field: str) -> pd.DataFrame:
    if df.empty or not date_field in df.columns:
        return df
    out_list = []
    for sym, grp in df.groupby("symbol", group_keys=False):
        grp_desc = grp.sort_values(date_field, ascending=False)
        picked = grp_desc.head(n)
        out_list.append(picked)
    final = pd.concat(out_list, ignore_index=True)
    final.sort_values(["symbol", date_field], ascending=True, inplace=True)
    final.reset_index(drop=True, inplace=True)
    return final

def _handle_output(output: Dict[str,Any], df: pd.DataFrame, mongo_uri: str, db_name: str):
    if df.empty or not output:
        return

    excel_path = output.get("excel_path")
    if excel_path:
        folder = os.path.dirname(excel_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        df.to_excel(excel_path, index=False)

    base_coll = output.get("base_collection")
    if base_coll:
        with MongoClient(mongo_uri) as cli:
            c = cli[db_name][base_coll]
            c.delete_many({})
            c.insert_many(df.to_dict("records"))

def _tokenize(expr: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z_]\w*", expr)
