import torch

# Absolute error
def abs_error_func(model, x: torch.Tensor, y: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str, backward: bool):

    y_pred = model(x, ts, rtol, atol, method, backward)

    abs_result = ((y_pred - y) ** 2).mean()
    return abs_result


# Absolute error shallow net
def abs_error_func_shallow_net(model, x: torch.Tensor, y: torch.Tensor):

    y_pred = model(0, x) # 0 is just an arbitrary point if time dependency is deactivated

    abs_result = ((y_pred.squeeze() - y) ** 2).mean()
    return abs_result


# Relative error
def rel_error_func(model, x: torch.Tensor, y: torch.Tensor, ts: torch.Tensor, rtol: float, atol: float, method: str, backward: bool):

    y_pred = model(x, ts, rtol, atol, method, backward)
    
    abs_result = ((y_pred - y) ** 2).mean()
    norm_y = (y ** 2).mean()
    result = abs_result / norm_y
    return result


# Relative error shallow net
def rel_error_func_shallow_net(model, x: torch.Tensor, y: torch.Tensor):

    y_pred = model(0, x) # 0 is just an arbitrary point if time dependency is deactivated
    
    abs_result = ((y_pred.squeeze() - y) ** 2).mean()
    norm_y = (y ** 2).mean()
    result = abs_result / norm_y
    return result