#!/usr/bin/env python3
"""Execute every stage notebook in place and write REPORT.md.

    python run_notebooks.py            # all stages
    python run_notebooks.py 01 04      # selected stages

The notebooks read only the committed JSON / CSV / npy files under <stage>/results/
and need numpy and scipy. Each notebook's code cells run in one namespace with the
stage folder as the working directory; stdout is written back into the notebook so
the committed .ipynb files carry their outputs. Any jupyter front end can execute
them as well (`jupyter nbconvert --execute --inplace <stage>/<stage>.ipynb`).
"""
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit() and (p / f"{p.name}.ipynb").exists())


def run_notebook(stage):
    nb_path = ROOT / stage / f"{stage}.ipynb"
    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(ROOT / stage)
    failed = False
    n = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            n += 1
            src = "".join(cell["source"])
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    exec(compile(src, f"{stage}:cell{n}", "exec"), ns)
                text = buf.getvalue()
            except Exception:  # noqa: BLE001
                text = buf.getvalue() + "\n" + traceback.format_exc()
                failed = True
            cell["execution_count"] = n
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}] if text else []
            if failed:
                break
    finally:
        os.chdir(cwd)
    nb["metadata"]["executed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return failed


def write_report(results):
    lines = ["# Reproduction report", "",
             f"Executed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `run_notebooks.py` (Python {sys.version.split()[0]}).",
             "", "Legend: **ok** reproduces the printed value to its precision (half a unit of the last printed digit) or satisfies the stated relation; **near** lies within one unit of the last printed digit, which points to rounding of an intermediate value rather than a different result; **mismatch** differs by more; **error** the check raised; **pointer** the manuscript value has no committed artifact and is listed rather than verified.", ""]
    tot = {"ok": 0, "near": 0, "mismatch": 0, "error": 0, "flag": 0}
    rows = []
    detail = []
    for stage, failed in results:
        f = ROOT / stage / f"_checks_{stage}.json"
        if not f.exists():
            rows.append(f"| {stage} | — | — | — | — | — | notebook did not reach its summary cell |")
            continue
        d = json.loads(f.read_text())
        c = d["counts"]
        for k in tot:
            tot[k] += c[k]
        n_num = c["ok"] + c["near"] + c["mismatch"] + c["error"]
        rows.append(f"| [{stage}]({stage}/{stage}.ipynb) | {c['ok']}/{n_num} | {c['near']} | {c['mismatch']} | {c['error']} | {c['flag']} | {'halted' if failed else 'completed'} |")
        for ch in d["checks"]:
            if ch["status"] in ("near", "mismatch", "error", "flag"):
                rep = "" if (ch["status"] == "flag" and ch["reported"] == ch["reason"]) else ch["reported"]
                detail.append(f"| {stage} | {ch['id']} | {ch['status']} | {ch['desc']} | {rep} | {ch['produced'] if ch['produced'] is not None else ''} | {ch['reason'] or ''} |")
    n_num = tot["ok"] + tot["near"] + tot["mismatch"] + tot["error"]
    lines += [f"**Headline: {tot['ok']}/{n_num} numeric checks reproduced exactly; {tot['near']} within one unit of the last printed digit; {tot['mismatch']} mismatch, {tot['error']} error; {tot['flag']} pointer-only.**", "",
              "| Stage | reproduced | near | mismatch | error | pointer | run |", "|---|---|---|---|---|---|---|", *rows, ""]
    if detail:
        lines += ["## Items that did not reproduce cleanly", "", "| Stage | id | status | description | reported | produced | note |", "|---|---|---|---|---|---|---|", *detail, ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    wanted = sys.argv[1:]
    stages = [s for s in STAGES if not wanted or any(s.startswith(w) for w in wanted)]
    results = []
    for s in stages:
        print("=" * 72 + f"\n{s}\n" + "=" * 72)
        failed = run_notebook(s)
        results.append((s, failed))
        f = ROOT / s / f"_checks_{s}.json"
        if f.exists():
            c = json.loads(f.read_text())["counts"]
            print(f"-> {c}")
        print("STATUS:", "ERROR (halted)" if failed else "completed")
    if not wanted:
        write_report(results)
    else:
        write_report([(s, False) for s in STAGES])
