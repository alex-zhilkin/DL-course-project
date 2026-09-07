"""Collect the finished AE matrix and persist a review packet, including failures."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from aggregate_lj_ae_repair import main as aggregate
from diagnose_lj_ae_predictions import main as diagnose

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/results/lj_ae_repair'


def main():
    aggregate()
    status=pd.read_csv(BASE/'status.csv')
    done=int(status.completed.sum())
    if done:
        diagnose()
    report=['# AE reconstruction study review',f'\nCompleted: {done}/{len(status)}.\n']
    failures=status[~status.completed]
    if len(failures):
        report+=['Incomplete runs (inspect job logs before retrying):','']
        report += [f'- {r.variant}, seed {r.seed}' for r in failures.itertuples()]
    path=BASE/'sourcewise_aggregate.csv'
    if path.exists():
        frame=pd.read_csv(path)
        report+=['\n## Frame-100 validation reconstruction','',
            '| Recipe | Source | Seeds | R² mean ± SD | Minimum seed R² | Position MSE | Valid/total minimum |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in frame[frame.frame==100].itertuples():
            report.append(f'| {r.variant} | {r.source} | {r.seeds} | {r.r2_mean:.3f} ± {r.r2_std:.3f} | {r.r2_min:.3f} | {r.position_mse:.3g} | {r.valid_min}/{r.total_min} |')
    report += ['\n## Continuation instructions','',
        'Read experiment_results_index.md, the latest 06b experiment log, every candidate recipe, sourcewise_results.csv, and retained_source_ranking.csv.',
        'Inspect failures and paired-seed changes relative to the existing mixed-T-unseen AE baseline. Review all horizons and strain errors, not endpoint rank alone.',
        'Mixed-T is evaluation-only and must not select the recipe. Historical response-selection variants violate the current dynamics-only requirement and must not be promoted. Expert observables are diagnostic only.',
        'Do not advance to LJ propagator supervision merely because one recipe ranks first. Use state/trajectory reconstruction for selection; expert response metrics are post-training diagnostics only.',
        'Preserve the successful 2D mixed-T-transfer baseline. Keep shared-node mendels_q placement and exact saved recipes for future runs.',
        '\nThis packet was generated automatically. It does not represent a new assistant review or automatically launch a new training study.']
    (BASE/'review_packet.md').write_text('\n'.join(report)+'\n')
    marker=BASE/'automatic_review_collected.json'
    if not marker.exists():
        with (ROOT/'notebooks/latent_space/06b_experiment_log.md').open('a') as f:
            f.write(f'\n## Automatic AE reconstruction collection\n\nCollected {done}/{len(status)} completed runs. Source-wise results, seed SDs, counts,\nfield/strain errors and a continuation packet are under `notebooks/results/lj_ae_repair/`.\nRead `review_packet.md` and the underlying recipes before making a scientific claim\nor advancing to LJ dynamics. Automatic collection is not a declaration of success.\n')
    marker.write_text(json.dumps(dict(collected_at=datetime.now(timezone.utc).isoformat(),completed=done,
        expected=len(status),requires_assistant_review=True),indent=2))
    print(f'Saved {BASE / "review_packet.md"}',flush=True)


if __name__=='__main__':main()
