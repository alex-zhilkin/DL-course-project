"""Independent full-frame audit of in-memory reference-box normalization."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('LSS_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT/'src'))
import torch
from lss.data import load_dataset, normalize_trajectory_to_reference_box, _install_legacy_auxetic_box_alias

FILES={'reid':'reid_200_frames.pt','depablo_low_temp':'depablo-near-zero-temp.pt',
       'depablo_mixed_temp':'depablo-10k-mix-temp.pt','lj_noisy':'lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',choices=FILES,required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    raw_path=ROOT/'data'/FILES[args.source]
    saved_path=ROOT/'data/normalized_reference_box_minus1_1'/FILES[args.source]
    output=ROOT/'notebooks/results/in_memory_normalization_audit'; output.mkdir(parents=True,exist_ok=True)
    report={'source':args.source,'job_id':os.environ.get('PBS_JOBID'),'failures':{},'max_absolute_error':{}}
    def compare(name,a,b):
        if a is None and b is None: return
        if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):
            good=a==b
        elif a.shape!=b.shape:
            good=False
        else:
            good=torch.allclose(a.to(torch.float64),b.to(torch.float64),atol=2e-6,rtol=2e-5,equal_nan=True)
            if a.numel():
                error=float((a.to(torch.float64)-b.to(torch.float64)).abs().nan_to_num().max())
                report['max_absolute_error'][name]=max(error,report['max_absolute_error'].get(name,0))
        if not good: report['failures'][name]=report['failures'].get(name,0)+1
    started=time.time()
    _install_legacy_auxetic_box_alias()
    raw=load_dataset(raw_path)
    import copy
    saved=copy.deepcopy(raw)
    for trajectory in saved: normalize_trajectory_to_reference_box(trajectory)
    assert len(raw)==len(saved)
    frames=0
    for i,(original,normalized) in enumerate(zip(raw,saved)):
        assert len(original)==len(normalized)
        ref=original[0]; box=ref.box
        lower=torch.tensor([box.x1,box.y1],dtype=ref.x.dtype)
        upper=torch.tensor([box.x2,box.y2],dtype=ref.x.dtype)
        center=(lower+upper)/2; half=(upper-lower)/2
        compare('reference_context_positions',normalized[0].reference_context_positions,ref.x[:,:2])
        compare('reference_context_edges',normalized[0].reference_context_edge_attr,ref.edge_attr)
        compare('reference_box_width',normalized[0].box_tensor[:2],torch.tensor([2.,2.]))
        for r,n in zip(original,normalized):
            frames+=1
            compare('position_affine',n.x[:,:2],(r.x[:,:2]-center)/half)
            compare('other_node_features',n.x[:,2:],r.x[:,2:])
            compare('edge_identity',n.edge_index,r.edge_index)
            compare('edge_scalars',n.edge_attr[:,3:],r.edge_attr[:,3:])
            compare('marker',getattr(n,'coordinate_normalization',None),'position_normalization')
            compare('normalization_center',n.reference_box_center,center)
            compare('normalization_scale',n.reference_box_half_extent,half)
            b=r.box
            expected=torch.tensor([(b.x2-b.x1)/half[0],(b.y2-b.y1)/half[1]])
            compare('evolving_box',n.box_tensor[:2],expected)
            nb=n.box
            compare('box_origin',torch.tensor([nb.x1,nb.y1]),(torch.tensor([b.x1,b.y1])-center)/half)
            a,z=n.edge_index
            v=n.x[z,:2]-n.x[a,:2]; v=v-torch.round(v/n.box_tensor[:2])*n.box_tensor[:2]
            compare('periodic_edge_vector',n.edge_attr[:,:2],v)
            compare('edge_length',n.edge_attr[:,2],v.norm(dim=-1))
            for field in ('pos','vel_state'):
                value=getattr(r,field,None)
                if isinstance(value,torch.Tensor):
                    expected=(value[:,:2]-center)/half if field=='pos' else value[:,:2]/half
                    compare(field,getattr(n,field)[:,:2],expected)
            for field in ('temperature','lj_epsilon'):
                rv,nv=getattr(r,field,None),getattr(n,field,None)
                if isinstance(rv,float) and rv!=rv and isinstance(nv,float) and nv!=nv: continue
                compare(field,nv,rv)
            for field in ('lj_sigma','lj_cutoff'):
                value=getattr(r,field,None)
                if isinstance(value,(int,float)):
                    compare(field,torch.as_tensor(getattr(n,field)),torch.tensor(value/float(half.prod().sqrt())))
        before=normalized[0].x.clone()
        normalize_trajectory_to_reference_box(normalized)
        compare('idempotence',normalized[0].x,before)
        if i%50==0: print('audited',args.source,i,flush=True)
    report.update(trajectories=len(raw),frames=frames)
    del raw,saved; gc.collect()
    report.update(seconds=time.time()-started,passed=not report['failures'])
    (output/f'{args.source}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if not report['passed']: raise RuntimeError(report['failures'])


if __name__=='__main__': main()
