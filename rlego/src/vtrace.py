import torch


def vtrace_td_error(v_tm1, v_t, r_t, discoun_t, rho_tm1, lambda_=1., clip=1.):
    c_tm1 = torch.clamp(rho_tm1, max=1.) * lambda_
    clipped_rhos_tm1 = torch.clamp(rho_tm1, max=clip)

    td_errors = clipped_rhos_tm1 * (r_t + discoun_t * v_t - v_tm1)

    err = 0.0
    errors = torch.zeros_like(v_t)
    for i in reversed(range(v_t.shape[0])):
        err = td_errors[i] + discoun_t[i] * c_tm1[i] * err
        errors[i] = err
    with torch.no_grad():
        target = errors + v_tm1
    return target - v_tm1


def vtrace_td_error_advantage(v_tm1, v_t, r_t, discoun_t, rho_tm1, lambda_=1.):
    errors = vtrace_td_error(v_tm1, v_t, r_t, discoun_t, rho_tm1)
    with torch.no_grad():
        targets_tm1 = errors + v_tm1
        v_t = torch.cat([lambda_ * targets_tm1[1:] + (1 - lambda_) * v_tm1[1:], v_t[-1:]], dim=0)
        q_t = r_t + discoun_t * v_t
        rho_clipped = torch.clamp(rho_tm1, max=1.)
        adv = rho_clipped * (q_t - v_tm1)
    return adv, errors
