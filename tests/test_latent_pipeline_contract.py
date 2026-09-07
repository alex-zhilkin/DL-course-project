"""Training/inference contracts for the active normalized-displacement AE path."""
from copy import deepcopy
import torch
from torch_geometric.data import Data
from lss.latent.models import NodeDeltaAttentionAutoEncoder
from lss.latent.simulation import batch_delta_graphs, ae_target_tensor
from lss.latent.training import encode_frame_latent, decode_latent_positions


def setup_case():
    torch.manual_seed(37)
    edges=torch.tensor([[0,0,1],[1,2,2]])
    ref=torch.tensor([[-1.,-.5],[.4,-.2],[.8,1.]])
    sim=[]
    for t in range(3):
        x=ref+torch.tensor([[.01,.02],[-.02,.01],[.03,-.01]])*t
        vec=x[edges[1]]-x[edges[0]]
        g=Data(x=x,edge_index=edges,edge_attr=torch.cat([vec,vec.norm(dim=1,keepdim=True),torch.ones(3,1)],1))
        g.reference_context_positions=ref*torch.tensor([4.,7.])+3
        rv=g.reference_context_positions[edges[1]]-g.reference_context_positions[edges[0]]
        g.reference_context_edge_attr=torch.cat([rv,rv.norm(dim=1,keepdim=True),torch.ones(3,1)],1)
        sim.append(g)
    model=NodeDeltaAttentionAutoEncoder(pos_dim=2,edge_dim=4,hidden_size=8,latent_dim=2,latent_tokens=3)
    model.edge_mode='compact_stored';model.eval()
    norms={}
    for name,d in [('node_feature',2),('target',2),('edge',4),('ref_edge',4)]:
        norms[name+'_mean']=torch.linspace(.02,.08,d).reshape(1,-1)
        norms[name+'_std']=torch.linspace(.4,1.1,d).reshape(1,-1)
    return sim,model,norms


def test_batched_training_matches_single_frame_encoding_and_decoding():
    sim,m,n=setup_case()
    b=batch_delta_graphs([sim],[(0,1),(0,2)],pos_dim=2,device='cpu',node_feature_mode='normalized_delta',edge_mode='compact_stored')
    with torch.no_grad():
        recon,z=m((b['node_feature']-n['node_feature_mean'])/n['node_feature_std'],b['ref_pos'],
            (b['edge_attr']-n['edge_mean'])/n['edge_std'],(b['ref_edge_attr']-n['ref_edge_mean'])/n['ref_edge_std'],b['edge_index'],b['batch'])
        for i,t in enumerate((1,2)):
            single=encode_frame_latent(m,sim,t,pos_dim=2,node_feature_mode='normalized_delta',normalizers=n,device='cpu')
            torch.testing.assert_close(single,z[i])
            pred=decode_latent_positions(m,sim,single,t,pos_dim=2,ae_target_mode='normalized_delta',normalizers=n,device='cpu')
            target=recon[b['batch']==i]*n['target_std']+n['target_mean']
            scale=sim[0].x.max(0).values-sim[0].x.min(0).values
            torch.testing.assert_close(pred,sim[0].x+target*scale)


def test_target_normalization_is_invertible_in_model_coordinate_system():
    sim,_,n=setup_case()
    b=batch_delta_graphs([sim],[(0,2)],pos_dim=2,device='cpu',node_feature_mode='normalized_delta',edge_mode='compact_stored')
    y=ae_target_tensor(b,'normalized_delta')
    recovered=((y-n['target_mean'])/n['target_std'])*n['target_std']+n['target_mean']
    torch.testing.assert_close(sim[0].x+recovered*b['position_scale'],sim[2].x)


def test_unseen_future_cannot_change_initial_encoding_or_decoded_prediction():
    sim,m,n=setup_case(); changed=deepcopy(sim)
    for g in changed[1:]:
        g.x.fill_(1234);g.edge_attr.fill_(-5678)
    with torch.no_grad():
        z=encode_frame_latent(m,sim,0,pos_dim=2,node_feature_mode='normalized_delta',normalizers=n,device='cpu')
        z2=encode_frame_latent(m,changed,0,pos_dim=2,node_feature_mode='normalized_delta',normalizers=n,device='cpu')
        torch.testing.assert_close(z,z2,rtol=0,atol=0)
        args=dict(pos_dim=2,ae_target_mode='normalized_delta',normalizers=n,device='cpu')
        torch.testing.assert_close(decode_latent_positions(m,sim,z,2,**args),decode_latent_positions(m,changed,z,2,**args),rtol=0,atol=0)
