#!/usr/bin/env python3
"""
SkyShade Scenario Performance Report
=====================================
Reads reports/run_metrics_history.json and prints a human-readable
breakdown of how each scenario is performing, what's going well,
and what still needs work.

Usage:
    python3 scenario_report.py
    python3 scenario_report.py --json        # machine-readable output
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

# ── paths ────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_HISTORY_PATH = os.path.join(_HERE, "reports", "run_metrics_history.json")

# ── scenario metadata ─────────────────────────────────────────────────────────

SCENARIOS = {
    "park": {
        "label": "Park",
        "description": (
            "Open grass loop with light obstacles and a figure-8 walking path. "
            "Good unobstructed sightlines and flat terrain make this the easiest "
            "environment for the tracker and PPO hover policy."
        ),
        "challenges": [
            "Few distractors so umbrella SVM sees simple weather only",
            "Flat ground means hover error is the dominant metric",
            "Battery drains evenly — straightforward MDP test",
        ],
    },
    "forest": {
        "label": "Forest Trail",
        "description": (
            "A long wooded trail where the human walks under dense tree canopy. "
            "Branches and leaves frequently occlude the target, and low-contrast "
            "bark/foliage makes the Sub-1 tracker lose confidence. "
            "The PPO hover policy must maintain altitude above the canopy."
        ),
        "challenges": [
            "Heavy occlusion drops Sub-1 tracker lock severely",
            "Canopy blocks direct line-of-sight, making hover positioning unreliable",
            "Variable lighting under the trees confuses the Sub-3 umbrella SVM",
        ],
    },
    "buildings": {
        "label": "Building District",
        "description": (
            "City plaza with buildings, roads, vehicles, and pedestrians. "
            "The drone must navigate around tall structures and avoid near-misses "
            "while the Sub-4 nav SAC decides when to deviate vs hold course."
        ),
        "challenges": [
            "Building proximity triggers avoidance, pulling the drone off the hover target",
            "Multiple pedestrians act as distractors for Sub-1 tracker",
            "Wind between buildings causes larger hover error",
        ],
    },
    "trail": {
        "label": "Urban Trail",
        "description": (
            "Urban trail with a bridge/underpass the drone must fly over and "
            "crowd pedestrians acting as deliberate distractors. "
            "The tracker must distinguish the followed human from a crowd. "
            "Battery reward is high when the drone reaches the far end efficiently."
        ),
        "challenges": [
            "Crowd distractors cause frequent Sub-1 track loss",
            "Bridge/underpass shadow drops HSV tracker confidence (fixed in v2)",
            "Umbrella decision shifts rapidly as the human passes through shade/sun",
        ],
    },
    "night": {
        "label": "Night Park",
        "description": (
            "The same park loop after dark with streetlamps as the only illumination. "
            "The Sub-1 tracker relies on its shadow-tolerant HSV band when the target "
            "moves between lamp pools. Tests the low-light detection path and the "
            "umbrella SVM on night-sky conditions (no rain, possible mist/fog)."
        ),
        "challenges": [
            "Between-lamp shadow zones test the low-value HSV fallback band",
            "No rain → umbrella SVM should consistently output STOW",
            "Slower walk means longer hover episodes — tests PPO patience",
        ],
    },
    "rooftop": {
        "label": "Rooftop",
        "description": (
            "A confined rooftop platform surrounded by low parapet walls. "
            "The human walks a slow elliptical lap while constant wind (3–8 m/s) "
            "buffets the drone. The PPO hover policy must reject wind disturbances "
            "and the nav SAC must avoid the wall edges."
        ),
        "challenges": [
            "Persistent wind (3–8 m/s) is the dominant hover stressor",
            "Parapet walls create a tight avoidance boundary for Sub-4 nav",
            "Open sky means tracker lock should be high — baseline check",
        ],
    },
    "stadium": {
        "label": "Stadium",
        "description": (
            "A full athletics stadium with an oval tartan track, tiered stands, "
            "four floodlight towers, an electronic scoreboard, and infield equipment "
            "(long-jump pit, high-jump mat, hurdles). The athlete jogs a lap of the "
            "400 m oval at a steady pace — testing whether the PPO circular-walk "
            "training mode generalises to a real curved trajectory."
        ),
        "challenges": [
            "Circular walk at 0.08 rad/s exercises the circular curriculum mode",
            "Open sky — tracker lock should be near-perfect; any loss flags a regression",
            "Light ambient wind + changing heading tests hover in low-wind curved flight",
        ],
    },
    "snow": {
        "label": "Snowy Field",
        "description": (
            "A snow-covered winter field with a frozen pond, pine trees, snowmen, "
            "a wooden fence, and a sled. The human walks a slow figure-8 while "
            "gusty omnidirectional wind pushes the drone. Overcast weather means "
            "the umbrella SVM mostly sees low-lux/no-rain conditions — a useful "
            "contrast to the rain-heavy other scenarios."
        ),
        "challenges": [
            "Omnidirectional gusting wind (shifts direction every ~15 s) stresses hover",
            "Low lux + zero rain tests the SVM boundary between 'cold overcast' and 'deploy'",
            "Open field — tracker lock should be near-perfect; any loss is a regression signal",
        ],
    },
    "vineyard": {
        "label": "Vineyard",
        "description": (
            "A vineyard with three parallel rows of trellis posts and foliage canopy. "
            "The human walks an out-and-back path along the central row. "
            "The foliage creates regular partial occlusion similar to forest canopy "
            "but in a structured, predictable geometry — tests whether the drone "
            "can maintain hover despite repetitive occlusion pulses from the vine posts."
        ),
        "challenges": [
            "Regular vine-post occlusion creates rhythmic confidence dips (unlike random forest)",
            "Adjacent rows at y=±3 m are nav obstacles the SAC must avoid",
            "Out-and-back walk with U-turn tests the velocity-biased gimbal recovery",
        ],
    },
    "parking": {
        "label": "Parking Lot",
        "description": (
            "A two-row car park where the human walks a serpentine path between "
            "parked vehicles. The drone must track through a grid of rectangular "
            "box obstacles — a different shape from the cylinders in other scenarios, "
            "testing whether the Sub-4 nav SAC generalises to boxy layouts."
        ),
        "challenges": [
            "Box-shaped car obstacles test nav SAC generalisation beyond cylinders",
            "Tight aisle width constrains hover accuracy when avoiding cars",
            "Serpentine path creates frequent 90° heading changes for the walker",
        ],
    },
    "beach": {
        "label": "Coastal Beach",
        "description": (
            "An open sandy beach where the human strolls along the shore. "
            "A steady 5 m/s lateral sea-breeze (plus weather wind) pushes the "
            "drone sideways continuously. No structural obstacles, so tracker lock "
            "should be near-perfect — this scenario isolates crosswind hover "
            "performance and is the cleanest stress-test of the PPO policy."
        ),
        "challenges": [
            "Constant lateral sea-breeze (5 m/s) requires sustained crosswind rejection",
            "Linear walk with drift means the velocity-biased gimbal search is exercised",
            "Open sky → tracker should be perfect; any lock loss is a regression flag",
        ],
    },
}

# ── ANSI colours ──────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _col(val, good, ok):
    if val >= good: return GREEN
    if val >= ok:   return YELLOW
    return RED

def _pct(val, good, ok, decimals=1):
    c = _col(val, good, ok)
    return f"{c}{val:.{decimals}f}%{RESET}"

def _grade_col(g):
    return {
        "A": GREEN, "B": CYAN, "C": YELLOW, "D": RED
    }.get(g, RESET)

# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def _trend(vals):
    """Linear slope of a series, normalised to [-1, 1] range of the data."""
    n = len(vals)
    if n < 3:
        return 0.0
    xs = list(range(n))
    mx = _mean(xs)
    my = _mean(vals)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, vals))
    den = sum((x - mx) ** 2 for x in xs)
    return (num / den) if den else 0.0

def _trend_str(slope, unit="pp/run"):
    if abs(slope) < 0.3:
        return f"{DIM}stable{RESET}"
    direction = "+" if slope > 0 else ""
    col = GREEN if slope > 0 else RED
    return f"{col}{direction}{slope:.1f} {unit}{RESET}"

def _bar(val, width=20, good=70, ok=40):
    filled = max(0, min(width, round(val / 100 * width)))
    c = _col(val, good, ok)
    return f"{c}{'█' * filled}{'░' * (width - filled)}{RESET}"

def _mini_sparkline(vals, width=30):
    """ASCII sparkline of a list of 0-100 values."""
    if not vals:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    result = []
    for v in vals[-width:]:
        idx = max(0, min(8, round(v / 100 * 8)))
        result.append(blocks[idx])
    return "".join(result)

# ── per-scenario analysis ─────────────────────────────────────────────────────

def _analyse(runs):
    """Return an analysis dict for a list of run records."""
    if not runs:
        return None

    hover   = [r["hover_pct"]          for r in runs]
    lock    = [r["tracker_lock_pct"]   for r in runs]
    umb     = [r["umbrella_correct_pct"] for r in runs]
    bat     = [r["battery_end_pct"]    for r in runs]
    overall = [r["overall_pct"]        for r in runs]
    grades  = [r["grade"]              for r in runs]

    return {
        "n": len(runs),
        "hover":   {"mean": _mean(hover),   "vals": hover,   "trend": _trend(hover)},
        "lock":    {"mean": _mean(lock),     "vals": lock,    "trend": _trend(lock)},
        "umb":     {"mean": _mean(umb),      "vals": umb,     "trend": _trend(umb)},
        "bat":     {"mean": _mean(bat),      "vals": bat,     "trend": _trend(bat)},
        "overall": {"mean": _mean(overall),  "vals": overall, "trend": _trend(overall)},
        "grades":  grades,
        "latest":  runs[-1],
        "best":    max(runs, key=lambda r: r["overall_pct"]),
    }

def _verdict(a):
    """Return (going_right, going_wrong) lists for an analysis dict."""
    right, wrong = [], []
    m = a["hover"]["mean"]
    if m >= 70:
        right.append(f"Hover accuracy is solid ({m:.0f}% mean) — PPO policy tracking well")
    elif m >= 40:
        right.append(f"Hover accuracy is developing ({m:.0f}% mean) — showing progress")
    else:
        wrong.append(
            f"Hover accuracy is critically low ({m:.0f}% mean) — drone is rarely holding "
            "position above the target; PPO needs significantly more training"
        )

    m = a["lock"]["mean"]
    if m >= 80:
        right.append(f"Tracker lock is reliable ({m:.0f}% mean) — Sub-1 stays on target")
    elif m >= 60:
        right.append(f"Tracker lock is adequate ({m:.0f}% mean) — acceptable for this environment")
    else:
        wrong.append(
            f"Tracker lock is poor ({m:.0f}% mean) — Sub-1 is losing the target frequently; "
            "check for occlusion, lighting changes, or crowd distractors"
        )

    m = a["umb"]["mean"]
    if m >= 85:
        right.append(f"Umbrella decisions are accurate ({m:.0f}% mean) — SVM classifying weather well")
    elif m >= 65:
        right.append(f"Umbrella accuracy is acceptable ({m:.0f}% mean) — minor misclassifications")
    else:
        wrong.append(
            f"Umbrella accuracy is low ({m:.0f}% mean) — Sub-3 SVM is misreading weather; "
            "retrain with data from this scenario"
        )

    m = a["bat"]["mean"]
    if m >= 30:
        right.append(f"Battery management is safe ({m:.0f}% remaining on average)")
    elif m >= 10:
        wrong.append(f"Battery is running low ({m:.0f}% remaining) — MDP may need tuning")
    else:
        wrong.append(f"Battery is critical ({m:.0f}% remaining) — nav policy is draining too fast")

    # trend
    ot = a["overall"]["trend"]
    if ot > 1.0:
        right.append(f"Strong positive trend (+{ot:.1f} pp/run) — system is learning")
    elif ot < -1.0:
        wrong.append(f"Declining trend ({ot:.1f} pp/run) — recent runs are getting worse")

    # grade distribution
    grade_d = sum(1 for g in a["grades"] if g == "D")
    if grade_d == 0 and len(a["grades"]) >= 3:
        right.append("No D-grade runs — consistent above-threshold performance")
    elif grade_d / len(a["grades"]) > 0.7:
        wrong.append(
            f"{grade_d}/{len(a['grades'])} runs graded D — system is failing this scenario "
            "consistently; address the subsystems above before running more"
        )

    return right, wrong

# ── main report ───────────────────────────────────────────────────────────────

def print_report(history):
    by_scenario = defaultdict(list)
    for r in history:
        sc = r.get("scenario", "unknown")
        by_scenario[sc].append(r)

    print()
    print(f"{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}  SkyShade — Scenario Performance Report{RESET}")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"  Generated {ts}  ·  {len(history)} total runs across {len(by_scenario)} scenario(s)")
    print(f"{BOLD}{'═' * 68}{RESET}")

    for sc_key, meta in SCENARIOS.items():
        runs = by_scenario.get(sc_key, [])
        print()
        label = meta["label"]
        run_count = len(runs)
        print(f"{BOLD}{CYAN}  ── {label.upper()} {'─' * (54 - len(label))}{RESET}")
        print(f"  {meta['description']}")
        print()

        # Challenges
        print(f"  {BOLD}Key challenges:{RESET}")
        for c in meta["challenges"]:
            print(f"    • {c}")
        print()

        if run_count == 0:
            print(f"  {YELLOW}No runs recorded yet for this scenario.{RESET}")
            print(f"  Select '{label}' from the launcher to start collecting data.")
            continue

        a = _analyse(runs)
        latest = a["latest"]

        # Metrics table
        print(f"  {BOLD}Performance ({run_count} run{'s' if run_count != 1 else ''}){RESET}")
        print(f"  {'Metric':<26} {'Mean':>7}  {'Latest':>7}  {'Trend':>14}  Sparkline")
        print(f"  {'-' * 64}")

        def _row(name, key, good, ok):
            mn  = a[key]["mean"]
            lat = a[key]["vals"][-1]
            tr  = _trend_str(a[key]["trend"])
            sp  = _mini_sparkline(a[key]["vals"])
            c_mn  = _col(mn,  good, ok)
            c_lat = _col(lat, good, ok)
            print(f"  {name:<26} {c_mn}{mn:6.1f}%{RESET}  "
                  f"{c_lat}{lat:6.1f}%{RESET}  {tr:>22}  {sp}")

        _row("Sub-2  Hover accuracy",   "hover",   70, 40)
        _row("Sub-1  Tracker lock",     "lock",    80, 60)
        _row("Sub-3  Umbrella correct", "umb",     85, 65)
        _row("Sub-4  Battery at end",   "bat",     30, 10)
        _row("Overall score",           "overall", 80, 65)

        # Grade history
        g_str = " ".join(
            f"{_grade_col(g)}{BOLD}{g}{RESET}" for g in a["grades"]
        )
        print(f"\n  Grade history:  {g_str}")

        best = a["best"]
        best_ts = datetime.fromtimestamp(best["timestamp"]).strftime("%Y-%m-%d")
        print(f"  Best run:  {_grade_col(best['grade'])}{BOLD}{best['grade']}{RESET}  "
              f"{best['overall_pct']:.0f}%  (on {best_ts})")

        # What's going right / wrong
        right, wrong = _verdict(a)
        print()
        if right:
            print(f"  {BOLD}{GREEN}✓ Going well:{RESET}")
            for item in right:
                print(f"    {GREEN}+{RESET} {item}")
        if wrong:
            print()
            print(f"  {BOLD}{RED}✗ Needs attention:{RESET}")
            for item in wrong:
                print(f"    {RED}–{RESET} {item}")

    # ── cross-scenario summary ────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}  CROSS-SCENARIO SUMMARY{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")
    print()

    rows = []
    for sc_key, meta in SCENARIOS.items():
        runs = by_scenario.get(sc_key, [])
        if not runs:
            rows.append((meta["label"], 0, "—", "—", "—", "—", "—", "—"))
            continue
        a = _analyse(runs)
        rows.append((
            meta["label"],
            len(runs),
            f"{a['hover']['mean']:.0f}%",
            f"{a['lock']['mean']:.0f}%",
            f"{a['umb']['mean']:.0f}%",
            f"{a['bat']['mean']:.0f}%",
            f"{a['overall']['mean']:.0f}%",
            "  ".join(a["grades"][-3:]) or "—",
        ))

    hdr = f"  {'Scenario':<20} {'Runs':>4}  {'Hover':>6}  {'Lock':>6}  {'Umb':>6}  {'Bat':>6}  {'Overall':>8}  Last 3"
    print(hdr)
    print(f"  {'-' * 64}")
    for r in rows:
        label, n, hover, lock, umb, bat, overall, last3 = r
        print(f"  {label:<20} {n:>4}  {hover:>6}  {lock:>6}  {umb:>6}  {bat:>6}  {overall:>8}  {last3}")

    print()
    print(f"  {BOLD}Priority actions:{RESET}")

    # Determine the most pressing issues across all scenarios
    issues = []
    for sc_key, meta in SCENARIOS.items():
        runs = by_scenario.get(sc_key, [])
        if not runs:
            continue
        a = _analyse(runs)
        label = meta["label"]
        if a["hover"]["mean"] < 40:
            issues.append((a["hover"]["mean"], f"{RED}Sub-2 PPO hover policy is failing in {label} "
                           f"({a['hover']['mean']:.0f}% avg) — train more steps or reduce learning rate{RESET}"))
        if a["lock"]["mean"] < 60:
            issues.append((a["lock"]["mean"], f"{RED}Sub-1 tracker loses lock in {label} "
                           f"({a['lock']['mean']:.0f}% avg) — environment is too occluded for current model{RESET}"))
        if a["umb"]["mean"] < 65:
            issues.append((a["umb"]["mean"], f"{YELLOW}Sub-3 SVM accuracy in {label} "
                           f"({a['umb']['mean']:.0f}% avg) — retrain SVM with {label.lower()} weather samples{RESET}"))

    if not issues:
        print(f"    {GREEN}All active scenarios are meeting minimum thresholds.{RESET}")
        print(f"    Consider running Building District to expand coverage.")
    else:
        issues.sort(key=lambda x: x[0])
        for _, msg in issues:
            print(f"    • {msg}")

    if "buildings" not in by_scenario:
        print(f"    • {YELLOW}Building District has never been run — "
              f"try it to stress-test the nav avoidance system{RESET}")

    print()


def json_report(history):
    by_scenario = defaultdict(list)
    for r in history:
        by_scenario[r.get("scenario", "unknown")].append(r)

    out = {}
    for sc_key, meta in SCENARIOS.items():
        runs = by_scenario.get(sc_key, [])
        if not runs:
            out[sc_key] = {"label": meta["label"], "runs": 0}
            continue
        a = _analyse(runs)
        right, wrong = _verdict(a)
        out[sc_key] = {
            "label":   meta["label"],
            "runs":    a["n"],
            "hover":   round(a["hover"]["mean"], 1),
            "lock":    round(a["lock"]["mean"],  1),
            "umbrella": round(a["umb"]["mean"],  1),
            "battery": round(a["bat"]["mean"],   1),
            "overall": round(a["overall"]["mean"], 1),
            "grades":  a["grades"],
            "going_right": right,
            "needs_attention": wrong,
        }
    print(json.dumps(out, indent=2))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SkyShade scenario performance report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--history", default=_HISTORY_PATH,
                        help="Path to run_metrics_history.json")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"No run history found at {args.history}", file=sys.stderr)
        print("Run at least one simulation first.", file=sys.stderr)
        sys.exit(1)

    with open(args.history) as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        print("Run history is empty.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        json_report(history)
    else:
        print_report(history)


if __name__ == "__main__":
    main()
