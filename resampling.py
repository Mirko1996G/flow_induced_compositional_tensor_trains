import torch
from f_interpolation import lin_int


def log_w(prior, posterior,samples, v_list, k, min_clamp, max_clamp):
    t = (k+1)/len(v_list)
    log_weights =  lin_int(prior, posterior, torch.tensor(t), samples)-calculate_log_weight_flow(prior, posterior,k, samples, v_list, min_clamp, max_clamp)
    lw_mean = log_weights.mean()
    lw_std = log_weights.std()

    log_weights = log_weights.clamp(lw_mean-lw_std, lw_mean+lw_std)
    return log_weights

def simple_resampling(x, prior, posterior, v_list, k, min_clamp, max_clamp): 
    """
    Simple resampling of log_weights and samples pair.

    from https://github.com/google-deepmind/annealed_flow_transport/blob/master/annealed_flow_transport/resampling.py
    with chat gpt to pytorch lol better check 
    """
    log_weights = log_w(prior, posterior, x, v_list, k, min_clamp, max_clamp)
    num_batch = log_weights.size(0)

    # Compute softmax probabilities
    probs = torch.nn.functional.softmax(log_weights, dim=0)
    #print(probs)
    
    # Sample indices with replacement
    indices = torch.multinomial(probs, num_batch, replacement=True)
    #print(indices)
    # Resample using the selected indices
    resamples = torch.index_select(x, 0, indices)
    resamples = resamples #+ torch.randn_like(resamples, device = device)*0.01
    return resamples

def calculate_log_weight_flow(prior, posterior, k,x, v_list, min_clamp, max_clamp):
    No_TT = len(v_list) 
    # with torch.set_grad_enabled(True):
    dlogp = torch.zeros(x.shape[0],1)
    v = v_list[k](x)
    divs = v_list[k].divergence(x).unsqueeze(1)
    divs = divs.clamp(min_clamp,max_clamp)
        
    dlogp = -divs/No_TT#trace_df_dz(v, x).view(x.shape[0], 1)/No_TT
    x = x - v/No_TT
    x = torch.clamp(x, torch.tensor(min_clamp),torch.tensor(max_clamp))
    t = k/No_TT

    log_w = (lin_int(prior, posterior, torch.tensor(t), x) + dlogp.squeeze())
    return log_w