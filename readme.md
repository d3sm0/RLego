From [RLax](https://github.com/deepmind/rlax):

Many functions consider policies, actions, rewards, values, in consecutive timesteps in order to compute their outputs. In this case the suffix _t and tm1 is often to clarify on which step each input was generated, e.g:
```
    q_tm1: the action value in the source state of a transition.
    a_tm1: the action that was selected in the source state.
    r_t: the resulting rewards collected in the destination state.
    discount_t: the discount associated with a transition.
    q_t: the action values in the destination state.
```
