"""Descriptive training-history audit; no expert metrics used for selection."""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]/'notebooks/results/lj_ae_repair'

def main():
    rows=[]
    for marker in sorted(BASE.glob('*/completed.json')):
        run=marker.parent
        recipe=json.loads((run/'recipe.json').read_text())
        if 'response' in recipe['variant']:continue
        h=pd.read_csv(run/'ae_history.csv')
        criterion='val_objective' if recipe['variant']=='longer' else 'val_max_source_reconstruction'
        selected=h.loc[h[criterion].idxmin()]
        for source in ('reid','depablo_low_temp','lj_noisy'):
            train=float(selected[f'train_source_{source}_reconstruction'])
            val=float(selected[f'val_source_{source}_reconstruction'])
            rows.append(dict(variant=recipe['variant'],seed=recipe['seed'],source=source,
                selected_epoch=int(selected.epoch),epochs=len(h),train_mse=train,val_mse=val,
                val_train_ratio=val/train,criterion=criterion))
    d=pd.DataFrame(rows)
    d.to_csv(BASE/'training_gap_diagnostic.csv',index=False)
    a=d.groupby(['variant','source']).agg(seeds=('seed','nunique'),train_mse=('train_mse','mean'),val_mse=('val_mse','mean'),val_train_ratio=('val_train_ratio','mean')).reset_index()
    a.to_csv(BASE/'training_gap_diagnostic_aggregate.csv',index=False)
    print(a.to_string(index=False))
    print('Training loss is online within an epoch; not a frozen-checkpoint train reevaluation. No proof of irreducible error.')
if __name__=='__main__':main()
