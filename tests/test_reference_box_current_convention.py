"""Independent anisotropic/compressing-box checks for the current convention."""
from copy import deepcopy
import torch
from torch_geometric.data import Data
from graph_utils.box import Box
from lss.data import normalize_trajectory_to_reference_box


def test_fixed_affine_map_preserves_strain_context_and_stiffness():
    edges=torch.tensor([[0],[1]])
    def frame(x,box):
        return Data(x=torch.tensor(x),pos=torch.tensor(x),vel_state=torch.tensor([[2.,4.],[2.,4.]]),
                    edge_index=edges,edge_attr=torch.tensor([[2.,4.,20**.5,7.]]),box=box)
    raw=[frame([[2.,2.],[4.,6.]],Box(0,8,0,16,-.1,.1)),
         frame([[1.6,2.2],[3.2,6.6]],Box(0,6.4,0,17.6,-.1,.1))]
    sim=deepcopy(raw); normalize_trajectory_to_reference_box(sim)
    torch.testing.assert_close(sim[0].x,torch.tensor([[-.5,-.75],[0.,-.25]]))
    torch.testing.assert_close(sim[1].x,torch.tensor([[-.6,-.725],[-.2,-.175]]))
    torch.testing.assert_close(sim[0].box_tensor,torch.tensor([2.,2.]))
    torch.testing.assert_close(sim[1].box_tensor,torch.tensor([1.6,2.2]))
    torch.testing.assert_close(sim[0].reference_context_positions,raw[0].x)
    torch.testing.assert_close(sim[0].reference_context_edge_attr,raw[0].edge_attr)
    torch.testing.assert_close(sim[1].edge_attr,torch.tensor([[.4,.55,(.4**2+.55**2)**.5,7.]]))
    torch.testing.assert_close(sim[1].vel_state,torch.tensor([[.5,.5],[.5,.5]]))
    raw_strain=(raw[1].x[1]-raw[1].x[0])/(raw[0].x[1]-raw[0].x[0])-1
    norm_strain=(sim[1].x[1]-sim[1].x[0])/(sim[0].x[1]-sim[0].x[0])-1
    torch.testing.assert_close(norm_strain,raw_strain)
    before=deepcopy(sim); normalize_trajectory_to_reference_box(sim)
    for a,b in zip(sim,before):
        for key in ('x','pos','vel_state','edge_attr','box_tensor'):
            torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)
