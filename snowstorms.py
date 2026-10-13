#!/usr/bin/env python3
"""
Largest snowstorm per season from xmACIS / ACIS station data.

Storm definition
----------------
* A "storm" is a run of consecutive days with measurable snowfall
  (>= 0.1"; trace, zero, and missing days break the run).
* If the run is <= 3 days, the storm total is the run total.
* If the run is > 3 days, only the 3-day window with the most snow is
  kept as the storm (e.g. Feb 3-8 2010 -> Feb 4-6 is the storm).
* Seasons run July 1 of year Y through June 30 of Y+1 ("2009-10").
  A storm is assigned to the season in which its window starts.

Outputs
-------
  <sid>_largest_storm_by_season.png   bar chart, one bar per season
  <sid>_largest_storm_distribution.png histogram + box plot + stats table
  <sid>_largest_storm_by_season.csv   the per-season table
  <sid>_all_storms.csv                every storm found

Usage
-----
  python snowstorms.py BWIthr
  python snowstorms.py DCAthr --climo 1991 2020 --climo-label end
  python snowstorms.py IADthr --save-raw iad.csv    # cache the pull
  python snowstorms.py IADthr --from-csv iad.csv    # re-run offline

Station IDs are anything ACIS accepts: threaded records like "BWIthr",
3-letter IDs like "BWI", COOP IDs like "180465", GHCN IDs, etc.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # noqa: F401  (handy if you extend it)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ACIS_URL = "https://data.rcc-acis.org/StnData"

# ---- palette (light) -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"        # 1991-2020 seasons / climo
BLUE_LT = "#9ec5f4"     # seasons outside the normals period
ORANGE = "#eb6834"      # mean
AQUA = "#1baf7a"        # median
BAND = "#cde2fb"        # Q1-Q3 band


# =============================================================================
# Data
# =============================================================================
def parse_snow(v):
    """ACIS snow string -> (inches as float or NaN, is_trace)."""
    if v is None:
        return np.nan, False
    v = str(v).strip()
    if v in ("", "M", "S"):          # missing / "subsequent" (folded into a later 'A' value)
        return np.nan, False
    if v == "T":
        return 0.0, True
    v = v.rstrip("ASMTabcdefghijklmnopqrstuvwxyzBCDEFGHIJKLNOPQRUVWXYZ")
    try:
        return float(v), False
    except ValueError:
        return np.nan, False


def fetch_acis(sid):
    import requests
    params = {
        "sid": sid,
        "sdate": "por",
        "edate": "por",
        "elems": [{"name": "snow"}],
        "meta": ["name", "state", "sids", "valid_daterange"],
    }
    r = requests.post(ACIS_URL, data={"params": json.dumps(params)}, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        sys.exit(f"ACIS error: {js['error']}")
    meta = js.get("meta", {})
    rows = js.get("data", [])
    if not rows:
        sys.exit(f"No data returned for {sid}")
    df = pd.DataFrame(rows, columns=["date", "raw"])
    df["raw"] = df["raw"].astype(str)
    name = meta.get("name", sid)
    if meta.get("state"):
        name = f"{name}, {meta['state']}"
    return df, name


def build_daily(df):
    """Raw (date, raw) -> continuous daily frame with snow (in) + trace flag."""
    parsed = df["raw"].map(parse_snow)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "snow": [p[0] for p in parsed],
        "trace": [p[1] for p in parsed],
    }).set_index("date").sort_index()
    # Reindex so days absent from the record count as missing.
    first = out.index.min()
    start = pd.Timestamp(first.year if first.month >= 7 else first.year - 1, 7, 1)
    full = pd.date_range(start, out.index.max(), freq="D")
    out = out.reindex(full)
    out["trace"] = out["trace"].fillna(False).astype(bool)
    out.index.name = "date"
    return out


def season_start_year(ts):
    return ts.year if ts.month >= 7 else ts.year - 1


def season_label(y):
    return f"{y}-{(y + 1) % 100:02d}"


# =============================================================================
# Storm detection
# =============================================================================
def find_storms(daily, threshold=0.1, window=3):
    vals = daily["snow"].to_numpy()
    dates = daily.index
    measurable = np.where(np.isnan(vals), False, vals >= threshold - 1e-9)

    storms = []
    n = len(vals)
    i = 0
    while i < n:
        if not measurable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and measurable[j + 1]:
            j += 1
        run = vals[i:j + 1]
        L = len(run)
        if L <= window:
            k0, k1 = i, j
        else:
            sums = np.convolve(run, np.ones(window), mode="valid")
            k = int(np.argmax(sums))          # first window with the max total
            k0, k1 = i + k, i + k + window - 1
        total = float(np.round(vals[k0:k1 + 1].sum(), 2))
        storms.append({
            "start": dates[k0], "end": dates[k1],
            "days": k1 - k0 + 1,
            "total": total,
            "run_start": dates[i], "run_end": dates[j],
            "run_days": L,
            "run_total": float(np.round(run.sum(), 2)),
            "daily": " / ".join(f"{v:.1f}" for v in vals[k0:k1 + 1]),
        })
        i = j + 1

    s = pd.DataFrame(storms)
    if not s.empty:
        s["season"] = s["start"].map(season_start_year)
    return s


def largest_by_season(daily, storms, max_missing_pct=10.0):
    """One row per complete season with its largest storm (0 if none)."""
    last_date = daily["snow"].last_valid_index()
    first_y = season_start_year(daily.index.min())
    last_y = season_start_year(last_date)

    rows = []
    for y in range(first_y, last_y + 1):
        s0, s1 = pd.Timestamp(y, 7, 1), pd.Timestamp(y + 1, 6, 30)
        if s1 > last_date:
            continue  # season not finished
        # completeness judged on the Nov 1 - Apr 30 core of the season
        core = daily.loc[pd.Timestamp(y, 11, 1):pd.Timestamp(y + 1, 4, 30), "snow"]
        miss_pct = 100.0 * core.isna().mean()
        row = {"season": y, "label": season_label(y),
               "missing_pct_nov_apr": round(miss_pct, 1),
               "complete": miss_pct <= max_missing_pct,
               "largest": 0.0, "start": pd.NaT, "end": pd.NaT,
               "days": 0, "daily": "", "season_total": float(daily.loc[s0:s1, "snow"].sum())}
        if not storms.empty:
            ss = storms[storms["season"] == y]
            if not ss.empty:
                top = ss.loc[ss["total"].idxmax()]
                row.update(largest=top["total"], start=top["start"], end=top["end"],
                           days=int(top["days"]), daily=top["daily"])
        rows.append(row)
    return pd.DataFrame(rows)


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean": x.mean(),
        "median": np.median(x),
        "q1": np.percentile(x, 25),
        "q3": np.percentile(x, 75),
        "std": x.std(ddof=1) if len(x) > 1 else np.nan,
        "min": x.min(),
        "max": x.max(),
    }


# =============================================================================
# Plotting
# =============================================================================
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_window(r):
    if pd.isna(r["start"]):
        return ""
    s, e = r["start"], r["end"]
    if s == e:
        return s.strftime("%b %-d, %Y")
    if s.month == e.month:
        return f"{s:%b} {s.day}–{e.day}, {e.year}"
    return f"{s:%b} {s.day}–{e:%b} {e.day}, {e.year}"


def plot_timeseries(seas, climo, cs, name, sid, climo_txt, out):
    fig, ax = plt.subplots(figsize=(15, 6.2), facecolor=SURFACE)
    style_axes(ax)

    in_c = seas["season"].isin(climo["season"])
    colors = np.where(in_c, BLUE, BLUE_LT)
    ax.bar(seas["season"], seas["largest"], width=0.8, color=colors, zorder=3)

    # Climatology overlay
    ax.axhspan(cs["q1"], cs["q3"], color=BAND, alpha=0.55, zorder=1, lw=0)
    ax.axhline(cs["mean"], color=ORANGE, lw=2, ls="--", zorder=4)
    ax.axhline(cs["median"], color=AQUA, lw=2, zorder=4)

    # Label the five biggest storms on record
    top = seas.nlargest(5, "largest").sort_values("season")
    placed = []   # (season, y_top_in_points-ish) to stagger neighbours
    for _, r in top.iterrows():
        dy = 4
        for s_prev, dy_prev in placed:
            if abs(r["season"] - s_prev) <= 4:
                dy = max(dy, dy_prev + 26)
        placed.append((r["season"], dy))
        ax.annotate(f"{r['largest']:.1f}\"\n{fmt_window(r)}",
                    (r["season"], r["largest"]), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_2, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if dy > 4 else None)

    ax.set_ylabel("Largest storm total (in)", color=INK_2)
    ax.set_xlabel("Season (July–June), labeled by starting year", color=INK_2)
    ax.set_xlim(seas["season"].min() - 1, seas["season"].max() + 1)
    ax.set_ylim(0, seas["largest"].max() * 1.25)

    handles = [
        Patch(color=BLUE, label=f"Seasons in {climo_txt}"),
        Patch(color=BLUE_LT, label="Other seasons"),
        Patch(color=BAND, alpha=0.55, label=f"{climo_txt} Q1–Q3 ({cs['q1']:.1f}–{cs['q3']:.1f}\")"),
        Line2D([], [], color=ORANGE, lw=2, ls="--", label=f"{climo_txt} mean ({cs['mean']:.1f}\")"),
        Line2D([], [], color=AQUA, lw=2, label=f"{climo_txt} median ({cs['median']:.1f}\")"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              ncol=3, labelcolor=INK_2)

    fig.suptitle(f"Largest snowstorm each season — {name} ({sid})",
                 x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold")
    ax.set_title("Storm = consecutive days with ≥0.1\" snow; runs longer than 3 days "
                 "keep only their snowiest 3-day window.",
                 loc="left", fontsize=9.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_distribution(seas, climo, cs, ps, name, sid, climo_txt, out, binw=2.0):
    fig = plt.figure(figsize=(14, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[3, 1.15],
                          hspace=0.32, wspace=0.18)
    axh = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, :])
    style_axes(axh)
    style_axes(axb)

    # --- histogram: stacked climo seasons + other seasons = full POR
    allv = seas["largest"].to_numpy()
    cv = climo["largest"].to_numpy()
    ov = seas.loc[~seas["season"].isin(climo["season"]), "largest"].to_numpy()
    top = binw * np.ceil((allv.max() + 1e-9) / binw)
    bins = np.arange(0, top + binw, binw)
    axh.hist([cv, ov], bins=bins, stacked=True, color=[BLUE, BLUE_LT],
             edgecolor=SURFACE, linewidth=1.5, zorder=3,
             label=[f"Seasons in {climo_txt} (n={len(cv)})",
                    f"Other seasons (n={len(ov)})"])
    axh.axvspan(cs["q1"], cs["q3"], color=BAND, alpha=0.55, zorder=1, lw=0,
                label=f"{climo_txt} Q1–Q3")
    axh.axvline(cs["mean"], color=ORANGE, lw=2, ls="--", zorder=4,
                label=f"{climo_txt} mean {cs['mean']:.1f}\"")
    axh.axvline(cs["median"], color=AQUA, lw=2, zorder=4,
                label=f"{climo_txt} median {cs['median']:.1f}\"")
    # ±1σ bracket and min/max ticks near the top of the panel
    ymax = axh.get_ylim()[1] * 1.12
    axh.set_ylim(0, ymax)
    yb = ymax * 0.93
    lo, hi = max(cs["mean"] - cs["std"], 0), cs["mean"] + cs["std"]
    axh.annotate("", (lo, yb), (hi, yb),
                 arrowprops=dict(arrowstyle="|-|", color=INK_2, lw=1.2, mutation_scale=4))
    axh.text(cs["mean"], yb * 1.015, f"mean ± 1σ ({cs['std']:.1f}\")",
             ha="center", va="bottom", fontsize=8.5, color=INK_2, zorder=6,
             bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    for v, lab in ((cs["min"], "min"), (cs["max"], "max")):
        axh.plot([v], [ymax * 0.82], marker="v", ms=8, color=INK, zorder=5)
        axh.text(v, ymax * 0.84, f"{climo_txt} {lab}\n{v:.1f}\"", ha="center",
                 va="bottom", fontsize=8, color=INK_2)
    axh.set_xlabel("Largest storm of the season (in)", color=INK_2)
    axh.set_ylabel("Number of seasons", color=INK_2)
    axh.set_xlim(0, top)
    axh.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK_2)

    # --- box plots: POR vs climo, whiskers = min/max, diamond = mean, bar = ±1σ
    data = [allv, cv]
    labels = [f"Period of record\n{seas['label'].iloc[0]} to {seas['label'].iloc[-1]}",
              f"{climo_txt}"]
    bp = axb.boxplot(data, whis=(0, 100), widths=0.5, patch_artist=True,
                     medianprops=dict(color=AQUA, lw=2.2),
                     whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2),
                     showfliers=False)
    for patch, c in zip(bp["boxes"], [BLUE_LT, BLUE]):
        patch.set_facecolor(c)
        patch.set_edgecolor(INK_2)
    for x, st in ((1, ps), (2, cs)):
        axb.errorbar(x + 0.36, st["mean"], yerr=st["std"], fmt="D", color=ORANGE,
                     ms=7, capsize=4, lw=1.5, zorder=5)
    axb.set_xticks([1, 2])
    axb.set_xticklabels(labels, fontsize=8.5, color=INK_2)
    axb.set_ylabel("Largest storm (in)", color=INK_2)
    axb.set_title("Box = Q1–Q3, line = median,\nwhiskers = min/max, ◆ = mean ± 1σ",
                  fontsize=8.5, color=INK_2)

    # --- stats table
    axt.axis("off")
    seas_i = seas.set_index("season")

    def who(st, frame):
        mn = frame.loc[frame["largest"].idxmin()]
        mx = frame.loc[frame["largest"].idxmax()]
        return mn["label"], f"{mx['label']} ({fmt_window(mx)})"

    pmin, pmax = who(ps, seas)
    cmin, cmax = who(cs, climo)
    rows = [
        ["Seasons (n)", f"{ps['n']}", f"{cs['n']}"],
        ["Mean", f"{ps['mean']:.1f}\"", f"{cs['mean']:.1f}\""],
        ["Median", f"{ps['median']:.1f}\"", f"{cs['median']:.1f}\""],
        ["Q1 – Q3", f"{ps['q1']:.1f}\" – {ps['q3']:.1f}\"", f"{cs['q1']:.1f}\" – {cs['q3']:.1f}\""],
        ["Std. deviation", f"{ps['std']:.1f}\"", f"{cs['std']:.1f}\""],
        ["Minimum", f"{ps['min']:.1f}\"  ({pmin})", f"{cs['min']:.1f}\"  ({cmin})"],
        ["Maximum", f"{ps['max']:.1f}\"  {pmax}", f"{cs['max']:.1f}\"  {cmax}"],
    ]
    tbl = axt.table(cellText=rows, colLabels=["", "Period of record", climo_txt],
                    loc="center", cellLoc="left", colWidths=[0.16, 0.42, 0.42])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if r == 0 or c == 0 else INK_2)
        if r == 0:
            cell.get_text().set_fontweight("bold")

    fig.suptitle(f"Distribution of each season's largest snowstorm — {name} ({sid})",
                 x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.935, f"Compared with the {climo_txt} climatology. "
             "Seasons with no measurable storm count as 0\".",
             fontsize=9.5, color=INK_2)
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sid", help="ACIS/xmACIS station id, e.g. BWIthr, DCAthr, IADthr")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="measurable snow threshold in inches (default 0.1)")
    ap.add_argument("--window", type=int, default=3, help="max storm window in days (default 3)")
    ap.add_argument("--climo", type=int, nargs=2, default=[1991, 2020], metavar=("Y1", "Y2"))
    ap.add_argument("--climo-label", choices=["end", "start"], default="end",
                    help="match the climo years to the season's ending year (default: "
                         "1991-2020 = 1990-91..2019-20) or its starting year "
                         "(1991-92..2020-21)")
    ap.add_argument("--max-missing", type=float, default=10.0,
                    help="drop seasons with more than this %% of Nov-Apr days missing")
    ap.add_argument("--start-season", type=int, default=None,
                    help="ignore seasons starting before this year")
    ap.add_argument("--from-csv", help="load raw data (date,raw) instead of calling ACIS")
    ap.add_argument("--save-raw", help="save the raw ACIS pull to this CSV")
    ap.add_argument("--name", help="override the station name in titles")
    ap.add_argument("--out", default=".", help="output directory")
    a = ap.parse_args()

    if a.from_csv:
        raw = pd.read_csv(a.from_csv, dtype=str)
        name = a.name or a.sid
    else:
        raw, name = fetch_acis(a.sid)
        name = a.name or name
    if a.save_raw:
        raw.to_csv(a.save_raw, index=False)

    daily = build_daily(raw)
    storms = find_storms(daily, a.threshold, a.window)
    seas_all = largest_by_season(daily, storms, a.max_missing)
    dropped = seas_all[~seas_all["complete"]]
    seas = seas_all[seas_all["complete"]].copy()
    if a.start_season is not None:
        seas = seas[seas["season"] >= a.start_season]
    seas = seas.reset_index(drop=True)
    if seas.empty:
        sys.exit("No complete seasons found.")

    y1, y2 = a.climo
    if a.climo_label == "end":
        climo = seas[(seas["season"] + 1 >= y1) & (seas["season"] + 1 <= y2)]
    else:
        climo = seas[(seas["season"] >= y1) & (seas["season"] <= y2)]
    climo_txt = f"{y1}–{y2}"
    expected = y2 - y1 + 1
    if len(climo) < expected:
        print(f"WARNING: only {len(climo)}/{expected} complete seasons in {climo_txt}.")
    if climo.empty:
        sys.exit("No seasons in the climatology period.")

    cs, ps = describe(climo["largest"]), describe(seas["largest"])

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = a.sid.replace("/", "_")
    plot_timeseries(seas, climo, cs, name, a.sid, climo_txt,
                    outdir / f"{tag}_largest_storm_by_season.png")
    plot_distribution(seas, climo, cs, ps, name, a.sid, climo_txt,
                      outdir / f"{tag}_largest_storm_distribution.png")

    seas_out = seas.assign(window=seas.apply(fmt_window, axis=1),
                           in_climo=seas["season"].isin(climo["season"]))
    seas_out[["label", "largest", "window", "days", "daily", "season_total",
              "missing_pct_nov_apr", "in_climo"]].to_csv(
        outdir / f"{tag}_largest_storm_by_season.csv", index=False)
    if not storms.empty:
        storms.to_csv(outdir / f"{tag}_all_storms.csv", index=False)

    # Console summary
    print(f"\n{name} ({a.sid}) — {len(seas)} complete seasons "
          f"({seas['label'].iloc[0]} to {seas['label'].iloc[-1]})")
    if len(dropped):
        print(f"Dropped {len(dropped)} incomplete season(s): "
              + ", ".join(dropped["label"].tolist()))
    print(f"\n{'':16}{'Period of record':>18}{climo_txt:>14}")
    for k, lab in [("n", "Seasons"), ("mean", "Mean"), ("median", "Median"),
                   ("q1", "Q1"), ("q3", "Q3"), ("std", "Std dev"),
                   ("min", "Min"), ("max", "Max")]:
        f = "{:>18d}{:>14d}" if k == "n" else "{:>18.1f}{:>14.1f}"
        print(f"{lab:16}" + f.format(ps[k], cs[k]))
    print("\nTop 10 storms:")
    for _, r in seas.nlargest(10, "largest").iterrows():
        print(f"  {r['label']}  {r['largest']:5.1f}\"  {fmt_window(r)}  [{r['daily']}]")
    print(f"\nWrote plots and CSVs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
