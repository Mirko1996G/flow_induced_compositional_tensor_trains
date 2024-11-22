import torch


# We define the gradient of the time-linear interpolation between the starting density f_0 and the target density f_1
def grad_f(prior, posterior, t,x):
    """
    f(t,x) = (t)*f_1(x) + (1-t)*f_0(x)
           = (t)*[-(x-mu)**2*0.5/var] + (1-t) * [-x**2*0.5]
    grad f_t(x) = -t*(x-mu)/var - (1-t) * x
    """
    x.requires_grad_(True)
    out = lin_int(prior, posterior, t,x)
    return torch.autograd.grad(out.sum(),x,retain_graph=True)[0]

def grad_interpolation(prior, posterior, t,x, f_t, multiply_with_t = True):
    """
    f(t,x) = (t)*f_1(x) + (1-t)*f_0(x)
           = (t)*[-(x-mu)**2*0.5/var] + (1-t) * [-x**2*0.5]
    grad f_t(x) = -t*(x-mu)/var - (1-t) * x
    """

    lin_grad = grad_lin_interpolation(prior, posterior, t,x)
    
    TT_grad = (1-t)*f_t.grad(torch.cat((x.detach(),t),1))[:,:-1]
    if multiply_with_t:
        TT_grad*=t
    return lin_grad + TT_grad

def grad_lin_interpolation(prior, posterior, t,x):
    x.requires_grad_(True)
    out = lin_int(prior, posterior, t,x)
    lin_grad = torch.autograd.grad(out.sum(),x,retain_graph=True)[0]
    return lin_grad

def partial_t_f_lin(prior,posterior, x):
    dimension=x.shape[1]
    # if (dimension == 1 and isinstance(posterior,D.MixtureSameFamily)):
    #     d_ft = (posterior.log_prob(x)-prior.log_prob(x)[:,None])#(1-time.squeeze())*    
    # else:
    d_ft = (posterior.log_prob(x)-prior.log_prob(x))#(1-time.squeeze())*
    return d_ft.squeeze()


def partial_t_interpolation(prior,posterior,t,x,f_t, multiply_with_t=True):
    dimension = x.shape[1]
    tt = t.squeeze()
    xt = torch.cat((x,tt.unsqueeze(1)),1)
    learned = f_t(xt).squeeze()
    learned_partial = f_t.grad(xt)[:,-1]*(1-tt)
    if multiply_with_t:
        learned*=(1-2*tt)
        learned_partial*=tt
    else:
        learned*=-1
    return partial_t_f_lin(prior,posterior, x) + learned +  learned_partial

def lin_int(prior, posterior, t,x):
    tt = t.squeeze()
    # if (dimension==1 and isinstance(posterior, D.MixtureSameFamily)):
    #      return (tt[:,None])*posterior.log_prob(x)+prior.log_prob(x)[:,None]*(1-tt[:,None])     
    # else:
    return (tt)*posterior.log_prob(x)+prior.log_prob(x)*(1-tt)


def Langevin(x, prior, posterior,t, num_steps = 20, step_size = 1e-3):
    for step in range(num_steps):
        x = x.requires_grad_()
        grad_v = torch.autograd.grad(-lin_int(prior, posterior, t,x).sum(), x)[0]
        x = x - grad_v*step_size + torch.sqrt(torch.tensor(2*step_size))*torch.randn_like(x)
        x = x.clone().detach()
    return x