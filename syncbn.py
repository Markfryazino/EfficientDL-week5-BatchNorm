import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float, run_all_reduce: bool):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        batch_size = input.size(0)
        sum_first_order = input.sum(dim=0)
        sum_second_order = (input ** 2).sum(dim=0)
        talk_tensor = torch.cat([torch.tensor([batch_size], device=input.device), sum_first_order, sum_second_order])

        dist.all_reduce(talk_tensor, op=dist.ReduceOp.SUM)

        num_samples = talk_tensor[0]
        total_sum_first_order = talk_tensor[1: 1 + sum_first_order.size(0)]
        total_sum_second_order = talk_tensor[1 + sum_first_order.size(0) :]

        est_mean = total_sum_first_order / num_samples
        est_var = total_sum_second_order / num_samples - est_mean ** 2
        est_std = torch.sqrt(est_var)

        new_running_mean = (1 - momentum) * running_mean.to(est_mean.device) + momentum * est_mean
        new_running_std = (1 - momentum) * running_std.to(est_std.device) + momentum * est_std

        std_eps = torch.sqrt(new_running_std ** 2 + eps)

        ctx.save_for_backward(
            input,
            new_running_mean,
            std_eps,
            new_running_std,
            torch.tensor(momentum, device=input.device),
            num_samples,
            est_mean,
            est_std,
            torch.tensor(run_all_reduce, device=input.device)
        )

        running_mean[:] = new_running_mean
        running_std[:] = new_running_std

        return (input - new_running_mean) / std_eps


    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        input, new_mean, std_eps, new_std, momentum, num_samples, est_mean, est_std, do_reduce = ctx.saved_tensors

        direct_gradient = grad_output / std_eps
        grad_inv_std_eps = ((input - new_mean) * grad_output).sum(dim=0)
        grad_new_mean = ((- 1 / std_eps) * grad_output).sum(dim=0)
        grad_new_std = - new_std / (std_eps ** 3) * grad_inv_std_eps

        grads = torch.cat([grad_new_mean, grad_new_std])

        if do_reduce:
            dist.all_reduce(grads, op=dist.ReduceOp.SUM)

        grad_new_mean = grads[:grad_new_mean.size(0)]
        grad_new_std = grads[grad_new_mean.size(0):]

        grad_est_mean = momentum * grad_new_mean
        grad_est_std = momentum * grad_new_std

        grad_x_through_mean = grad_est_mean / num_samples
        grad_x_through_std = grad_est_std * (input - est_mean) / (num_samples * est_std)

        grad_x = direct_gradient + grad_x_through_mean + grad_x_through_std

        return grad_x, None, None, None, None, None



class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))
        self.bn = sync_batch_norm.apply

    def forward(self, input: torch.Tensor, run_all_reduce: bool = True) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above

        if self.training:
            x = self.bn(
                input, self.running_mean, self.running_std, self.eps, self.momentum, run_all_reduce
            )
        else:
            x = (input - self.running_mean.to(input.device)) / torch.sqrt(self.running_std.to(input.device) ** 2 + self.eps)

        return x
