# tensordict-optim
Pytorch omptimizers, implemented for tensordicts.

Minimal and easy to modify versions of many pytorch optimizers, made for use with tensordict.
Instead of iterating over every parameter, just use TD! Fully compatible with torch autograd.

Example: 
```
params = tensordict.TensorDict(dict(net.named_parameters())) # create tensor dict
optimizer = tensordict_optim.Adam(params, weight_decay=1e-2)
  ...
  # during trainiong loop
  params.apply_(tensordict_optim.zero_grad) # reset gradients manually 
  l.backward() # pytorch autograd - saves grad in our tensordict!
  optimizer.step(params, lr) # perform optimization step
```

Currently implemented optimizer:
- SGD
- SGD with momentum
- Nesterov momentum
- AdaGrad
- RMSProp
- Adam

All optimizers support weight decay. (In this implementation, weight decay is not added to the moments. For moments with weight decay, add L2 regularization to the loss before applying `backward()`.)
