"""Check log for the reproduction notebooks.

Each notebook records every comparison between a value produced from the committed
result files and the value printed in the manuscript. `summary()` prints the tally
and writes `_checks_<stage>.json`, which `run_notebooks.py` turns into REPORT.md.

Statuses
  ok         produced == reported (to the printed precision, or the stated relation)
  near       within one unit of the last printed digit but beyond half a unit (rounding of
             an intermediate value; reported separately, not counted as reproduced)
  mismatch   they differ
  error      the expression raised
  flag       reported value has no committed artifact (pointer only)
  table      documentation row for a table checked cell by cell
"""
import json
import math

_CHECKS = []
_NAME = None


def start(name):
    global _NAME
    _NAME = name
    _CHECKS.clear()


def _tol(nd, tol):
    if tol is not None:
        return tol
    if nd is not None:
        return 0.5 * 10 ** (-nd) + 1e-9
    return 1e-6


def _close(a, b, nd, tol):
    try:
        return abs(float(a) - float(b)) <= _tol(nd, tol)
    except (TypeError, ValueError):
        return a == b


def _near(a, b, nd):
    try:
        return nd is not None and abs(float(a) - float(b)) <= 10 ** (-nd) + 1e-9
    except (TypeError, ValueError):
        return False


def check(cid, desc, produced, reported, nd=None, tol=None, mode="round", reason=None):
    ok = False
    status = "ok"
    near = False
    try:
        if mode == "round":
            ok = _close(produced, reported, nd, tol)
            near = (not ok) and _near(produced, reported, nd)
        elif mode == "eq":
            ok = produced == reported
        elif mode in ("ge", "le", "gt", "lt"):
            p, r = float(produced), float(reported)
            ok = {"ge": p >= r, "le": p <= r, "gt": p > r, "lt": p < r}[mode]
        elif mode == "range":
            lo, hi = reported
            vals = produced if isinstance(produced, (list, tuple)) else [produced]
            ok = all(lo <= float(v) <= hi for v in vals)
        elif mode in ("pair", "list"):
            ok = len(produced) == len(reported) and all(_close(a, b, nd, tol) for a, b in zip(produced, reported))
            near = (not ok) and len(produced) == len(reported) and all(_close(a, b, nd, tol) or _near(a, b, nd) for a, b in zip(produced, reported))
        elif mode == "eq_or_flag":
            if produced is None:
                status = "flag"
                ok = True
            else:
                ok = tuple(produced) == tuple(reported) if isinstance(reported, (list, tuple)) else produced == reported
        else:
            raise ValueError(f"unknown mode {mode}")
    except Exception as e:  # noqa: BLE001
        status = "error"
        produced = f"ERR: {e}"
        ok = False
    if status == "ok" and not ok:
        status = "near" if near else "mismatch"
    mark = {"ok": "OK ", "near": "~~ ", "mismatch": "XX ", "error": "ER ", "flag": "-- "}[status]
    _CHECKS.append({"id": cid, "desc": desc, "produced": _plain(produced), "reported": _plain(reported), "status": status, "mode": mode,
                    "reason": reason if status == "flag" else None})
    print(f"[{mark}] {cid} {desc}: produced={_fmt(produced)}  reported={_fmt(reported)}" + (f"  ({reason})" if status == "flag" and reason else ""))
    return ok


def flag(cid, desc, reported, reason):
    _CHECKS.append({"id": cid, "desc": desc, "produced": None, "reported": _plain(reported), "status": "flag", "mode": "flag", "reason": reason})
    print(f"[-- ] {cid} {desc}: reported={_fmt(reported)}  NO COMMITTED ARTIFACT: {reason}")


def table(cid, desc, reported):
    _CHECKS.append({"id": cid, "desc": desc, "produced": None, "reported": _plain(reported), "status": "table", "mode": "table", "reason": None})


def _plain(x):
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:  # noqa: BLE001
            return str(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in x) + ")"
    return str(x)


def summary():
    n = {s: sum(1 for c in _CHECKS if c["status"] == s) for s in ("ok", "near", "mismatch", "error", "flag", "table")}
    n_num = n["ok"] + n["near"] + n["mismatch"] + n["error"]
    print(f"\n=== {_NAME}: {n['ok']}/{n_num} numeric checks reproduced exactly; {n['near']} within one unit of the last printed digit; {n['mismatch']} mismatch, {n['error']} error, {n['flag']} pointer-only ===")
    for c in _CHECKS:
        if c["status"] in ("near", "mismatch", "error"):
            print(f"  {c['status'].upper():8s} {c['id']} {c['desc']}: produced={_fmt(c['produced'])} reported={_fmt(c['reported'])}")
    with open(f"_checks_{_NAME}.json", "w") as f:
        json.dump({"stage": _NAME, "counts": n, "checks": _CHECKS}, f, indent=1)
    return n
