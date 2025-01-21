import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pymongo import MongoClient

def run_aggregator_from_file(config_path: str) -> pd.DataFrame:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return run_aggregator(config)

def run_aggregator(config: Dict[str, Any]) -> pd.DataFrame:
    """
    1) Fetch data from Mongo
    2) Unify duplicates by (symbol,date,freq,period)
    3) Apply mapping & formulas
    4) Apply rolling if enabled
    5) Pick last n rows if specified
    6) Sort ascending by date
    7) Reorder columns: [symbol,date,freq], then mapped fields, then formula columns, then rolling
    8) Output (Excel/Mongo) if requested
    """
    # ------------------------------------------------------------------
    # 1) Basic setup
    # ------------------------------------------------------------------
    mongo_uri = config.get("mongodb_uri", "mongodb://localhost:27017")
    db_name   = config.get("mongo_database", "EscapeBase")
    colls     = config.get("collections", ["Raw"])
    filters   = config.get("filters", {})
    fields    = config.get("fields", [])
    mapping   = config.get("mapping", {})
    formulas  = config.get("formulas", {})
    rolling   = config.get("rolling", {})
    output    = config.get("output", {})

    # ------------------------------------------------------------------
    # 2) Fetch + unify
    # ------------------------------------------------------------------
    df = _fetch_data(mongo_uri, db_name, colls, filters, fields)
    if df.empty:
        return df
    df = _unify_duplicates(df)

    # ------------------------------------------------------------------
    # 3) Apply mapping & formulas
    # ------------------------------------------------------------------
    df = _apply_mapping(df, mapping)
    df = _apply_formulas(df, formulas)

    # ------------------------------------------------------------------
    # 4) Rolling logic (if enabled)
    # ------------------------------------------------------------------
    if rolling.get("enabled", False):
        df = _apply_rolling(df, rolling)

    # ------------------------------------------------------------------
    # 5) Possibly pick last N
    # ------------------------------------------------------------------
    n_val = filters.get("n")
    date_field = filters.get("date_field", "date")
    if n_val and n_val > 0:
        df = _pick_last_n(df, n_val, date_field)

    # ------------------------------------------------------------------
    # 6) Sort ascending by date
    # ------------------------------------------------------------------
    if date_field in df.columns:
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce")
        df.sort_values(["symbol", date_field], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 7) Reorder columns in a clean order
    # ------------------------------------------------------------------
    df = _reorder_columns(df, fields, mapping, formulas, rolling)

    # ------------------------------------------------------------------
    # 8) Output
    # ------------------------------------------------------------------
    _handle_output(output, df, mongo_uri, db_name)

    return df

# ----------------------------------------------------------------------
# Internal functions
# ----------------------------------------------------------------------
def _fetch_data(mongo_uri: str, db_name: str, collections: List[str],
                filters: Dict[str,Any], fields: List[str]) -> pd.DataFrame:
    date_field = filters.get("date_field", "date")
    proj = {"_id":0, date_field:1, "symbol":1, "freq":1}
    for f in fields:
        proj[f] = 1

    query = _build_query(filters)

    frames = []
    with MongoClient(mongo_uri) as cli:
        db = cli[db_name]
        for c in collections:
            docs = list(db[c].find(query, proj))
            if docs:
                df_part = pd.DataFrame(docs)
                if not df_part.empty:
                    frames.append(df_part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

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
        if isinstance(syms, str):
            syms = [syms]
        q["symbol"] = {"$in": syms}

    inds = filters.get("assigned_industry")
    if inds:
        if isinstance(inds, str):
            inds = [inds]
        q["assigned_industry"] = {"$in": inds}

    ep = filters.get("endpoint")
    if ep:
        if isinstance(ep, str):
            ep = [ep]
        q["endpoint"] = {"$in": ep}

    # date range => start_date / end_date
    start_date = filters.get("start_date")
    end_date   = filters.get("end_date")
    if start_date or end_date:
        date_query = {}
        if start_date:
            date_query["$gte"] = pd.to_datetime(start_date, errors="coerce")
        if end_date:
            date_query["$lte"] = pd.to_datetime(end_date, errors="coerce")
        date_field = filters.get("date_field", "date")
        q[date_field] = date_query

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
    rolling config example:
      enabled: true
      window: 4
      method: "mean"
      columns: ["some_field","another_field"]
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
    if df.empty or date_field not in df.columns:
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

def _reorder_columns(df: pd.DataFrame, fields: List[str], mapping: Dict[str, str],
                     formulas: Dict[str, str], rolling: Dict[str,Any]) -> pd.DataFrame:
    """
    Reorders columns so that final layout is:
      1) symbol, date, freq (if present, in that order)
      2) The mapped fields (in the order they appear in 'fields' list)
      3) The formula columns (in the order they appear in 'formulas')
      4) Rolling columns that end with _rollX, sorted alpha
      5) Any leftover columns
    """
    if df.empty:
        return df

    # We'll build a list of columns in the desired order,
    # then do df = df[that_order] if col in df
    desired_order = []

    # 1) symbol, date, freq
    for c in ["symbol", "date", "freq"]:
        if c in df.columns and c not in desired_order:
            desired_order.append(c)

    # 2) The mapped fields in user-specified order
    #    For each field in 'fields', see if mapping -> new_name
    #    else it might remain as old name if not mapped
    for f in fields:
        new_name = mapping.get(f, f)  # if f in mapping => mapped name, else f
        if new_name in df.columns and new_name not in desired_order:
            desired_order.append(new_name)

    # 3) The formula columns (in the order they appear in 'formulas')
    for new_col in formulas.keys():
        if new_col in df.columns and new_col not in desired_order:
            desired_order.append(new_col)

    # 4) Rolling columns => if rolling enabled
    rolling_cols = []
    if rolling.get("enabled", False):
        for col in df.columns:
            if "_roll" in col and col not in desired_order:
                rolling_cols.append(col)
        rolling_cols.sort()  # alphabetical
    desired_order.extend(rolling_cols)

    # 5) Add leftover columns
    #    e.g. anything that wasn't in the above sets
    leftover = []
    for col in df.columns:
        if col not in desired_order:
            leftover.append(col)

    final_order = desired_order + leftover

    # Now reorder
    # only keep columns that exist in df
    final_order = [c for c in final_order if c in df.columns]

    return df[final_order]

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
