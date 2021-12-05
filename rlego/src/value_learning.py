def td_learning(v_tm1, r_t, discount_t, v_t, stop_grad=True):
    if stop_grad:
        v_t = v_t.detach()
    td = r_t + discount_t * v_t - v_tm1
    return td


def q_learning(q_tm1, r_t, discount_t, q_t, stop_grad=True):
    if stop_grad:
        q_t = q_t.detach()
    td = r_t + discount_t * q_t.max(-1) - q_tm1
    return td
