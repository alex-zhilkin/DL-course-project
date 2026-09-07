"""Check algebra of reversing standardized directional edge attributes."""
from pathlib import Path
import json
import torch
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'notebooks/results/lj_ae_repair/equal_source_s456/ae.pt'
b=torch.load(p,map_location='cpu',weights_only=False)
rows=[]
for prefix in ('','ref_'):
 mean=b['stats'][prefix+'edge_mean'].flatten()[:2]
 std=b['stats'][prefix+'edge_std'].flatten()[:2]
 raw=torch.tensor([0.7,-0.4])
 encoded=(raw-mean)/std
 expected=(-raw-mean)/std
 actual=-encoded
 corrected=-encoded-2*mean/std
 assert torch.allclose(expected,corrected,rtol=1e-5,atol=1e-5)
 rows.append(dict(features=prefix+'edge',mean=mean.tolist(),std=std.tolist(),
     implemented_minus_expected=(actual-expected).tolist(),
     corrected_max_abs_error=float((corrected-expected).abs().max())))
out=ROOT/'notebooks/results/ae_information_audit/edge_reversal.json'
out.write_text(json.dumps(dict(checkpoint=str(p),rows=rows,
 caveat='Algebraic check with saved training normalizers; not a measurement of downstream accuracy impact.'),indent=2))
print(out)
