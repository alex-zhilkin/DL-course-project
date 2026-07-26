"""Test whether global attention can identify p-ratio from the undeformed LJ graph."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from lss.latent.simulation import pearson_r, r2_score
from lss.models.transformer_simulator import TwoStageDownUpTransformer
from lss.utils import resolve_device
from train_lj_static_velocity import OUT, SEED


RESULT = OUT.parent / "12_lj_static_pratio_attention"


class StaticPRatioAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim):
        super().__init__()
        self.core = TwoStageDownUpTransformer(
            in_node_dim=node_dim, in_edge_dim=edge_dim, hidden_size=64, pos_dim=2,
            transformer_layers=3, transformer_heads=4, transformer_dropout=.1, num_mlp=2,
        )
        self.global_in = nn.Sequential(nn.Linear(global_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Dropout(.15), nn.Linear(96, 1))

    def forward(self, data):
        self.core(data)
        graph_context = self.core.last_cv
        global_context = self.global_in(data.u.reshape(graph_context.size(0), -1))
        return self.head(torch.cat([graph_context, global_context], dim=1))


def run_epoch(model, loader, mean, std, device, optimizer=None):
    model.train(optimizer is not None); losses=[]
    for data in loader:
        data=data.to(device); prediction=model(data)
        target=(data.p_ratio-mean)/std
        mse=F.mse_loss(prediction,target)
        pred_centered=prediction-prediction.mean()
        target_centered=target-target.mean()
        correlation=(pred_centered*target_centered).mean() / (
            pred_centered.square().mean().sqrt()*target_centered.square().mean().sqrt()+1e-6
        )
        # The correlation term prevents the easy constant/mean-response shortcut.
        loss=mse+0.5*(1-correlation)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def main():
    torch.manual_seed(SEED);np.random.seed(SEED);device=resolve_device("auto")
    rows=torch.load(OUT/"static_velocity_targets.pt",map_location="cpu",weights_only=False)
    test_idx=list(range(len(rows)-200,len(rows)));pool=np.arange(len(rows)-200);np.random.default_rng(SEED).shuffle(pool)
    val_idx,train_idx=pool[:148].tolist(),pool[148:1148].tolist()
    y=torch.cat([rows[i].p_ratio for i in train_idx]);mean=y.mean().reshape(1,1).to(device);std=y.std(unbiased=False).reshape(1,1).to(device)
    model=StaticPRatioAttention(rows[0].x.size(1),rows[0].edge_attr.size(1),rows[0].u.size(1)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=5e-4)
    train_loader=DataLoader([rows[i] for i in train_idx],batch_size=24,shuffle=True)
    val_loader=DataLoader([rows[i] for i in val_idx],batch_size=48)
    best=None;best_val=float("inf");stale=0;history=[]
    for epoch in range(1,121):
        train=run_epoch(model,train_loader,mean,std,device,optimizer)
        with torch.no_grad():val=run_epoch(model,val_loader,mean,std,device)
        history.append({"epoch":epoch,"train_loss":train,"val_loss":val})
        if val<best_val-1e-4:best_val=val;stale=0;best=deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:stale+=1
        if epoch==1 or epoch%10==0:print(f"p epoch {epoch:03d} train={train:.4f} val={val:.4f} stale={stale}",flush=True)
        if stale>=20:break
    model.load_state_dict(best);model.eval();pred=[]
    with torch.no_grad():
        for data in DataLoader([rows[i] for i in test_idx],batch_size=48):
            pred.extend((model(data.to(device))*std+mean).cpu().reshape(-1).tolist())
    true=np.array([float(rows[i].p_ratio) for i in test_idx]);pred=np.asarray(pred)
    result={"train_networks":len(train_idx),"val_networks":len(val_idx),"test_networks":len(test_idx),
        "p_ratio_r2":r2_score(true,pred),"p_ratio_pearson":pearson_r(true,pred),"best_val_loss":best_val}
    RESULT.mkdir(parents=True,exist_ok=True);pd.DataFrame(history).to_csv(RESULT/"history.csv",index=False)
    pd.DataFrame({"sim_index":test_idx,"true_p_ratio":true,"pred_p_ratio":pred}).to_csv(RESULT/"test_predictions.csv",index=False)
    pd.DataFrame([result]).to_csv(RESULT/"summary.csv",index=False);torch.save({"state_dict":model.state_dict(),"mean":mean.cpu(),"std":std.cpu()},RESULT/"model.pt")
    print(result)


if __name__=="__main__":main()
