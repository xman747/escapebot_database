import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pymongo import MongoClient

def run_aggregator_from_file(config_path: str) -> pd.DataFrame:
    """
    Convenience wrapper: loads YAML config from disk, then calls run_aggregator().
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return run_aggregator(config)

def run_aggregator(config: Dict[str, Any]) -> pd.DataFrame:
    """
    1) Fetch base columns from Mongo (per config["fields"]).
    2) Apply mapping to rename columns.
    3) Apply formulas to create new columns.
    4) If rolling.enabled, compute rolling averages/sums on the entire historical dataset.
    5) Pick last N rows if filters.n is defined.
    6) Sort ascending by date and return the final DataFrame.
    7) Optionally save to Excel / store in Mongo if config["output"] is set.
    """
    # -------------------------------------------------------------------------
    # 1. Fetch / unify data
    # -------------------------------------------------------------------------
    df_merged = _fetch_and_unify(config)

    if df_merged.empty:
        return df_merged

    # -------------------------------------------------------------------------
    # 2. Apply mapping (base columns first)
    # -------------------------------------------------------------------------
    df_merged = _apply_mapping(df_merged, config.get("mapping", {}))

    # -------------------------------------------------------------------------
    # 3. Apply custom formulas
    # -------------------------------------------------------------------------
    df_merged = _apply_formulas(df_merged, config.get("formulas", {}))

    # -------------------------------------------------------------------------
    # 4. If rolling is enabled, compute rolling aggregates across all historical data
    # -------------------------------------------------------------------------
    roll_cfg = config.get("rolling", {})
    if roll_cfg.get("enabled", False):
        df_merged = _apply_rolling(df_merged, roll_cfg)

    # -------------------------------------------------------------------------
    # 5. Pick last N rows (e.g., 10 years)
    # -------------------------------------------------------------------------
    filters = config.get("filters", {})
    n_val = filters.get("n")
    date_field = filters.get("date_field", "date")

    if n_val and n_val > 0:
        df_merged = _pick_last_n(df_merged, n_val, date_field)

    # -------------------------------------------------------------------------
    # 6. Sort ascending by date & symbol
    # -------------------------------------------------------------------------
    if date_field in df_merged.columns:
        df_merged[date_field] = pd.to_datetime(df_merged[date_field], errors="coerce")
        df_merged.sort_values(["symbol", date_field], ascending=True, inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------------------------
    # 7. Output if requested
    # -------------------------------------------------------------------------
    _handle_output(config, df_merged)

    return df_merged

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _fetch_and_unify(config: Dict[str,Any]) -> pd.DataFrame:
    """
    Fetch from Mongo using config["collections"], config["filters"], etc.
    Merge duplicates by symbol+date+freq.
    """
    mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
    mongo_db  = config.get("mongo_db", "FinanceDB")
    colls     = config.get("collections", ["Raw"])
    filters   = config.get("filters", {})
    fields    = config.get("fields", [])
    date_field= filters.get("date_field", "date")

    # Build projection
    proj = {"_id":0}
    # Always keep date, symbol, freq for grouping/sorting
    for must_have in [date_field, "symbol", "freq"]:
        proj[must_have] = 1
    for f in fields:
        proj[f] = 1

    # Build the query from config
    query = _build_query(filters)

    # Fetch from Mongo
    df_list = []
    with MongoClient(mongo_uri) as cli:
        db = cli[mongo_db]
        for c_name in colls:
            docs = list(db[c_name].find(query, proj))
            if docs:
                df_part = pd.DataFrame(docs)
                if not df_part.empty:
                    df_list.append(df_part)

    if not df_list:
        return pd.DataFrame()

    df_merged = pd.concat(df_list, ignore_index=True)

    # unify duplicates => group by (symbol, date, freq, period?) to fill numeric
    df_merged = _unify_duplicates(df_merged)

    return df_merged

def _build_query(filters: Dict[str,Any]) -> dict:
    """
    Construct a Mongo query from the filters in config.
    e.g. freq, assigned_industry, symbol, date range, etc.
    """
    q = {}
    # freq
    freq_ = filters.get("freq")
    if freq_:
        if isinstance(freq_, str):
            freq_ = [freq_]
        freq_ = [f.upper() for f in freq_]
        q["freq"] = {"$in": freq_}

    # symbol
    syms = filters.get("symbol")
    if syms:
        q["symbol"] = {"$in": syms}

    # assigned_industry
    inds = filters.get("assigned_industry")
    if inds:
        q["assigned_industry"] = {"$in": inds}

    # date range can be done similarly if desired

    return q

def _unify_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicates by grouping on (symbol, date, freq, period),
    filling numeric columns from first non-NaN in any row of the group.
    """
    if df.empty:
        return df

    group_cols = []
    for c in ["symbol","date","freq","period"]:
        if c in df.columns:
            group_cols.append(c)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def combine_rows(g: pd.DataFrame) -> pd.Series:
        row0 = g.iloc[0].copy()
        for col in num_cols:
            if pd.isna(row0[col]):
                for _, row in g.iterrows():
                    if pd.notnull(row[col]):
                        row0[col] = row[col]
                        break
        return row0

    out = df.groupby(group_cols, as_index=False).apply(combine_rows)
    if isinstance(out, pd.DataFrame):
        out.reset_index(drop=True, inplace=True)

    return out

def _apply_mapping(df: pd.DataFrame, mapping: Dict[str,str]) -> pd.DataFrame:
    """
    Overwrite columns with new names if in `mapping`.
    """
    if df.empty or not mapping:
        return df

    rename_map = {}
    for old_col, new_col in mapping.items():
        if old_col in df.columns:
            rename_map[old_col] = new_col
    return df.rename(columns=rename_map)

def _apply_formulas(df: pd.DataFrame, formulas: Dict[str,str]) -> pd.DataFrame:
    """
    Evaluate each formula row-by-row, storing results in new columns.
    Example:  grossProfit: "(revenue - cost)"
    We'll parse tokens -> replace them with df["col"] -> eval expression.
    """
    if df.empty or not formulas:
        return df.copy()

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

def _apply_rolling(df: pd.DataFrame, roll_cfg: Dict[str,Any]) -> pd.DataFrame:
    """
    For each symbol, date-sorted ascending, compute a rolling average/sum for the
    specified columns. The aggregator creates new columns like col_roll4 if window=4.

    Example roll_cfg:
      enabled: true
      window: 4
      method: "mean"   # or "sum", etc.
      columns: ["revenue","net_inc"]
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["symbol","date"], ascending=True, inplace=True)

    window = roll_cfg.get("window", 4)
    method = roll_cfg.get("method", "mean")  # "mean" or "sum"
    cols   = roll_cfg.get("columns", [])

    out_list = []
    for sym, grp in df.groupby("symbol", group_keys=False):
        grp = grp.sort_values("date")
        for c in cols:
            if c in grp.columns:
                # rolling operation
                rolled = grp[c].rolling(window=window, min_periods=1)
                if method == "sum":
                    grp[f"{c}_roll{window}"] = rolled.sum()
                else:
                    grp[f"{c}_roll{window}"] = rolled.mean()
        out_list.append(grp)

    df_out = pd.concat(out_list, ignore_index=True)
    df_out.sort_values(["symbol","date"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out

def _tokenize(expr: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z_]\w*", expr)

def _pick_last_n(df: pd.DataFrame, n: int, date_field: str) -> pd.DataFrame:
    """
    For each symbol, pick the last n rows by date (descending), then re-sort ascending.
    """
    if df.empty or date_field not in df.columns:
        return df

    out_parts = []
    for sym, grp in df.groupby("symbol", group_keys=False):
        grp_desc = grp.sort_values(date_field, ascending=False)
        out_parts.append(grp_desc.head(n))

    df_new = pd.concat(out_parts, ignore_index=True)
    df_new.sort_values(["symbol", date_field], ascending=True, inplace=True)
    df_new.reset_index(drop=True, inplace=True)
    return df_new

def _handle_output(config: Dict[str,Any], df: pd.DataFrame):
    """
    If config["output"] has instructions, e.g. excel_path or base_collection, handle them.
    """
    if df.empty:
        return
    out_cfg = config.get("output", {})
    excel_path = out_cfg.get("excel_path")
    if excel_path:
        folder = os.path.dirname(excel_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        df.to_excel(excel_path, index=False)

    base_coll = out_cfg.get("base_collection")
    if base_coll:
        mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
        mongo_db  = config.get("mongo_db", "FinanceDB")
        with MongoClient(mongo_uri) as cli:
            coll = cli[mongo_db][base_coll]
            coll.delete_many({})
            coll.insert_many(df.to_dict("records"))
