"""Source-wise metrics for validation diagnostics; no figure output."""
import numpy as np
import pandas as pd
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box


def strain(reference, graph):
    sides = directional_side_indices_from_box(reference)
    def extent(g):
        p = g.x[:, :2].detach().cpu().numpy()
        return np.array([p[sides['right'], 0].mean()-p[sides['left'], 0].mean(),
                         p[sides['top'], 1].mean()-p[sides['bottom'], 1].mean()])
    initial = extent(reference)
    return (extent(graph)-initial)/initial


def metrics(sim, frame, pred):
    true_s, pred_s = strain(sim[0], sim[frame]), strain(sim[0], pred)
    diff = pred.x[:, :2].detach().cpu()-sim[frame].x[:, :2].detach().cpu()
    return dict(position_mse=float(diff.square().mean()),
        strain_x_error=float(abs(true_s[0]-pred_s[0])), strain_y_error=float(abs(true_s[1]-pred_s[1])),
        true_p_ratio=float(calc_p_ratio_rollout_sides(sim, frame)),
        pred_p_ratio=float(calc_p_ratio_rollout_sides([sim[0], pred], -1)),
        true_strain_x=true_s[0], true_strain_y=true_s[1])


def summarize(rows, keys):
    out = []
    for key, g in rows.groupby(keys, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(keys, key))
        valid = np.isfinite(g.true_p_ratio) & np.isfinite(g.pred_p_ratio)
        y, p = g.loc[valid, 'true_p_ratio'].to_numpy(), g.loc[valid, 'pred_p_ratio'].to_numpy()
        denominator = np.square(y-y.mean()).sum() if len(y) else 0
        row.update(total=len(g), valid=int(valid.sum()), true_valid=int(np.isfinite(g.true_p_ratio).sum()),
            p_ratio_r2=1-np.square(y-p).sum()/denominator if denominator > 0 else np.nan,
            p_ratio_mae=np.abs(y-p).mean() if len(y) else np.nan,
            target_variance=np.var(y) if len(y) else np.nan)
        for col in ('position_mse', 'strain_x_error', 'strain_y_error', 'latent_mse', 'normalized_latent_mse'):
            if col in g: row[col] = g[col].mean()
        out.append(row)
    return pd.DataFrame(out)
