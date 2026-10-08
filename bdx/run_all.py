"""Run the full experimental grid of the paper and write every raw result to results/.

    python -m bdx.run_all                 # full grid (≈1 h on 2 CPU cores)
    python -m bdx.run_all --quick         # 2 seeds, MC-Dropout only (≈5-10 min) -> results_quick/
    python -m bdx.run_all --config my.yaml --out results_other

Outputs (CSV, one row per unit of analysis):
    runs.csv          one row per (dataset, drift, model, seed, detector)
    signals.csv       one row per (dataset, drift, model, seed, signal): AUROC, Cohen's d
    xai.csv           one row per (dataset, drift, seed): localisation & stability scores
    xai_importances.csv
    elec2.csv         natural-drift lead-lag analysis per (model, seed)
    example_trace.npz a full trace used by figures and the app
    manifest.json     versions, config hash, dataset checksums, wall time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from . import data as D


def _job(args):
    import torch
    torch.set_num_threads(1)
    from .experiment import run_one
    dataset, drift_type, model, seed, cfg = args
    try:
        o = run_one(dataset, drift_type, model, seed, cfg)
        return o["rows"], o["signal_rows"], o["seconds"], None
    except Exception:
        return [], [], 0, f"{args[:4]}: {traceback.format_exc()}"


def _xai_job(args):
    import torch
    torch.set_num_threads(1)
    from . import xai
    from .experiment import run_one
    dataset, drift_type, seed, cfg = args
    try:
        o = run_one(dataset, drift_type, "mcd", seed, cfg, keep_traces=True)
        tr = o["traces"]
        st = tr["stream"]
        al = [a for a in tr["alarms"]["PH[lr]"] if a >= st.t0]
        alarm, src = (al[0], "PH[lr]") if al else (st.t0 + st.width // 2, "fallback:mid-ramp")
        res, imps = xai.explain_run(tr["model"], st, tr["signals"], alarm, cfg, seed=seed)
        res.update(dataset=dataset, drift=drift_type, seed=seed, alarm=alarm, alarm_source=src,
                   gt_features=";".join(st.feature_names[i] for i in st.gt_features))
        imp_rows = []
        for ex, v in imps.items():
            for j, f in enumerate(st.feature_names):
                imp_rows.append(dict(dataset=dataset, drift=drift_type, seed=seed, explainer=ex, feature=f,
                                     is_gt=j in st.gt_features, preA=v["preA"][j], preB=v["preB"][j],
                                     post=v["post"][j], delta=v["delta"][j]))
        return res, imp_rows, None
    except Exception:
        return None, [], f"{args[:3]}: {traceback.format_exc()}"


def _elec_job(args):
    import torch
    torch.set_num_threads(1)
    from . import stats as S
    from .experiment import run_one
    model, seed, cfg = args
    o = run_one("elec2", "natural", model, seed, cfg, keep_traces=True)
    tr = o["traces"]
    blk = cfg.get("elec_block", 200)
    n = len(tr["err"]) // blk
    e = tr["err"][: n * blk].reshape(n, blk).mean(1)
    rows = []
    lags = list(range(-5, 6))
    for s in ["conf", "tu", "au", "eu", "lr"]:
        x = tr["signals"][s][: n * blk].reshape(n, blk).mean(1)
        est, lo, hi = S.cross_correlation_block_bootstrap(x, e, lags, block=5, B=500, seed=seed)
        p_xy, _ = S.granger(x, e, maxlag=3)
        p_yx, _ = S.granger(e, x, maxlag=3)
        for l, c, a, b in zip(lags, est, lo, hi):
            rows.append(dict(model=model, seed=seed, signal=s, lag=l, corr=c, ci_low=a, ci_high=b,
                             granger_p_signal_to_error=p_xy, granger_p_error_to_signal=p_yx))
    counts = {k: len(v) for k, v in tr["alarms"].items()}
    return rows, counts, dict(model=model, seed=seed, acc_ref=tr["acc_ref"], acc_stream=float(1 - tr["err"].mean()))


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(D.ROOT, "config", "default.yaml"))
    ap.add_argument("--out", default=None, help="output folder (default: results/, or results_quick/ with --quick)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: grid xai elec2 example")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    if a.out is None:   # a quick check never overwrites the paper's results
        a.out = os.path.join(D.ROOT, "results_quick" if a.quick else "results")
    if a.quick:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["grid"]["models"] = ["mcd"]
        cfg["xai"]["seeds"] = cfg["xai"]["seeds"][:1]
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    for k in D.SPECS:          # verify data provenance up-front
        D.fetch(k)

    if "grid" not in a.skip:
        g = cfg["grid"]
        units = [(ds, dr) for ds in g["tabular_datasets"] for dr in g["tabular_drifts"]] + [tuple(x) for x in g["extra"]]
        jobs = [(ds, dr, m, s, cfg) for ds, dr in units for m in g["models"] for s in cfg["seeds"]]
        # ablation: the same drifts but abrupt (width = 0) — where a changepoint model should shine
        cfg_abrupt = {**cfg, "width": 0}
        jobs += [(ds, dr, m, s, cfg_abrupt) for ds, dr in (tuple(x) for x in g.get("abrupt", []))
                 for m in g["models"] for s in cfg["seeds"]]
        print(f"[grid] {len(jobs)} runs on {a.jobs} workers")
        rows, srows, errs = [], [], []
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs)):
                r, s, sec, err = f.result()
                rows += r
                srows += s
                if err:
                    errs.append(err)
                    print("ERROR", err)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(jobs)} done ({time.time() - t_all:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "runs.csv"), index=False)
        pd.DataFrame(srows).to_csv(os.path.join(a.out, "signals.csv"), index=False)

    if "xai" not in a.skip:
        xc = cfg["xai"]
        jobs = [(ds, dr, s, cfg) for ds in xc["datasets"] for dr in xc["drifts"] for s in xc["seeds"]]
        print(f"[xai] {len(jobs)} explanation runs")
        res, imps = [], []
        with ProcessPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(_xai_job, j) for j in jobs]):
                r, ir, err = f.result()
                if err:
                    print("ERROR", err)
                    continue
                res.append(r)
                imps += ir
                print(f"  xai {r['dataset']}/{r['drift']}/seed{r['seed']} done ({time.time() - t_all:.0f}s)", flush=True)
        pd.DataFrame(res).to_csv(os.path.join(a.out, "xai.csv"), index=False)
        pd.DataFrame(imps).to_csv(os.path.join(a.out, "xai_importances.csv"), index=False)

    if "elec2" not in a.skip:
        jobs = [(m, s, cfg) for m in cfg["grid"]["models"] for s in cfg["seeds"]]
        print(f"[elec2] {len(jobs)} natural-stream runs")
        rows, counts, meta = [], [], []
        with ProcessPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(_elec_job, j) for j in jobs]):
                r, c, m = f.result()
                rows += r
                counts.append({**m, **c})
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "elec2.csv"), index=False)
        pd.DataFrame(counts).to_csv(os.path.join(a.out, "elec2_alarm_counts.csv"), index=False)

    if "example" not in a.skip:
        from .experiment import run_one
        o = run_one("credit", "covariate", "mcd", 0, cfg, keep_traces=True)
        tr = o["traces"]
        np.savez_compressed(os.path.join(a.out, "example_trace.npz"),
                            **{f"sig_{k}": v for k, v in tr["signals"].items()},
                            err=tr["err"], p=tr["p"], t0=tr["t0"], t_acc=-1 if tr["t_acc"] is None else tr["t_acc"],
                            acc_ref=tr["acc_ref"], acc_thr=tr["acc_thr"],
                            alarms=json.dumps(tr["alarms"]))

    import river, shap, sklearn, torch, lime  # noqa
    manifest = dict(config=cfg, config_hash=config_hash(cfg), python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                                  river=river.__version__, shap=shap.__version__, sklearn=sklearn.__version__),
                    datasets={k: dict(file=v.file, sha256=v.sha256, licence=v.licence) for k, v in D.SPECS.items()},
                    wall_seconds=time.time() - t_all)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2, default=str)
    print(f"[done] {time.time() - t_all:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
