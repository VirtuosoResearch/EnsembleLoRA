import torch
import random
import numpy as np

from collections import OrderedDict
from transformers.pytorch_utils import Conv1D


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_layers(model, use_lora=True, use_adapter=False, use_ensemble=True):
    assert use_lora or use_adapter
    layers = OrderedDict()
    for name, module in model.named_modules():
        if use_lora:
            if use_ensemble:
                if ("adapter_list.0" in name) and (("lora_A__default" in name) or ("lora_B__default" in name)):
                    if "dummy" in name: # skip dummy layers
                        continue
                    layers[name] = module
            else:
                if ("lora_A.default" in name) or ("lora_B.default" in name):
                    layers[name] = module
        elif use_adapter:
            if use_ensemble:
                if ("adapter_list.0" in name) and ("adapter" in name) and ("__" in name) and hasattr(module, "weight"):
                    if "dummy" in name: # skip dummy layers
                        continue
                    layers[name] = module
            else:
                if ("adapter" in name) and hasattr(module, "weight"):
                    layers[name] = module

    for name in layers.keys():
        print(name)
    return layers

# def normalization(vs):
#     """
#     normalization of a list of vectors
#     return: normalized vectors v
#     """
#     norms = [torch.sum(v*v) for v in vs]
#     norms = [(norm**0.5).cpu().item() for norm in norms]
#     vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
#     return vs

def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def orthnormal(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)

"""
compute the trace of hessian using Hutchinson's method
https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
"""
def compute_hessians_trace(model, loss, device = "cpu", maxIter=100, tol=1e-3, use_lora=True, use_adapter=False, use_ensemble=True):
    # Get parameters and gradients of corresponding layer

    layers = get_layers(model, use_lora, use_adapter, use_ensemble)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    layer_traces = []
    trace_vhv = []
    trace = 0.

    # Start Iterations
    for _ in range(maxIter):
        vs = [torch.randint_like(weight, high=2) for weight in weights]
            
        # generate Rademacher random variables
        for v in vs:
            v[v == 0] = -1

        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    return np.mean(np.array(layer_traces), axis=0)

""" Calculate Top Eigenvalue of Hessian """ 
def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-8, top_n=1, use_lora=True, use_adapter=False, use_ensemble=True):
    layers = get_layers(model, use_lora, use_adapter, use_ensemble)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    """ use negative loss to get the minimum eigenvalue here """
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
            tmp_eigenvalues = [ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors

""" Calculate Hessian Norms: (W-W^)^T (H) (W - W^s)"""
def compute_hessians_quantity(model, criterion, data, target, device="cpu", state_dict = None, use_lora=True, use_adapter=False, use_ensemble=True):
    # Get parameters and gradients of corresponding layer
    data, target = data.to(device), target.to(device)
    model = model.to(device)    
    output = model(data)
    loss = criterion(output, target)

    layers = get_layers(model, use_lora, use_adapter, use_ensemble)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
    
    vs = []
    for name, module in layers.items():
        weight = module.weight
        if "pred_head" in name:
            v = weight.detach().clone()
        else:
            v = weight.detach().clone() - state_dict[name+".weight"]
        vs.append(v)

    model.zero_grad()    
    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

    layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
    
    return loss.detach(), np.array(layer_hessian_quantities)
