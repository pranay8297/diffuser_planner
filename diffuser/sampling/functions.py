import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, **sample_kwargs):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

@torch.no_grad()
def n_step_guided_p_sample_ddim(
    model, x, cond, cts, pts, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, gamma = 0, **sample_kwargs
):
    
    model_log_variance = extract(model.posterior_log_variance_clipped, cts, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    # sample kwargs contain unnecessary kwargs which is irrelevant here

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, cts)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[cts < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    # breakpoint()
    gamma = 0 if 'gamma' not in sample_kwargs else sample_kwargs['gamma']

    noise_hat = model.model(x, cond, cts) # call the temporal UNET model
    noise_new = torch.randn_like(x)

    x_recon = model.predict_start_from_noise(x, t = cts, noise = noise_hat) # x0_hat

    # sqrt(alpha_bar_prev) * x_recon + (1 - alpha_bar_prev - sigma**2) * noise_pred + sigma * noise_new
    xt_prev = (
            extract(model.sqrt_alphas_cumprod, pts, x_recon.shape) * x_recon +
            extract(model.sqrt_one_minus_alphas_cumprod, pts, x_recon.shape) * noise_hat 
        )

    return xt_prev, y
