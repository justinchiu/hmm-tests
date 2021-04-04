
def zero_grads(params):
    for param in params:
        param.grad.zero_()

def clone_and_zero_grads(params):
    grads = (param.grad.detach().clone() for param in params)
    zero_grads(params)
    return grads

def run_inference(text, params, inference_fn):
    evidence = inference_fn(text, *params)
    evidence.sum().backward()
    grads = clone_and_zero_grads(params)
    return evidence.detach(), grads
