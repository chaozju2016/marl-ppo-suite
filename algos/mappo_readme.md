# MAPPO Implementation Notes

## State Representation Types

The implementation supports different state representation types:

- **FP (Feature Pruned)**: Agent-specific global states where each agent has its own state representation.
- **EP (Environment Provided)**: A single state per environment shared across all agents.
- **AS (Agent-Specific)**: A combination of global state and agent-specific observations, where each agent has a unique state representation that includes both global information and its own observations.

In practice, users often combine global state and observations in Agent-Specific (AS) mode, which requires running critic training per agent rather than per environment. This is because each agent has a unique state representation, even though they may share some global information.

## Optimizing Critic Evaluation for Environment-Central (EP) State Type

When using the Environment-Provided (EP) state type, all agents in the same environment share the same state. This creates an opportunity for optimization: instead of evaluating the critic once per agent, we can evaluate it once per environment and broadcast the results to all agents.

## Implementation Options

### Option 1: Optimized `evaluate_actions` Method

This implementation handles both sequence-first and non-sequence-first layouts:

```python
def evaluate_actions(self,
                     state,             # (S, B, F)  or  (B, F)
                     obs,               # unchanged
                     actions,
                     available_actions,
                     masks,             # (S, B, 1) or (B, 1)
                     actor_h0=None,
                     critic_h0=None):

    # ---------- actor branch (unchanged) -------------------------------
    act_logp, dist_ent, _ = self.actor.evaluate_actions(
        obs, actions, actor_h0, masks, available_actions)

    # ---------- fast path for non-EP state types -----------------------
    if self.state_type != "EP":
        values, _ = self.critic(state, critic_h0, masks)
        return values, act_logp, dist_ent

    # ------------------------------------------------------------------
    #                EP  (centralised state)  path
    # ------------------------------------------------------------------
    A = self.n_agents

    if state.dim() == 3:                    # (S, B, F)  sequence-first
        Slen, Btot, F = state.shape
        assert Btot % A == 0, "batch not divisible by n_agents"
        Nenv = Btot // A                    # n_envs * rollout_steps

        # (S, n_envs, n_agents, F) -> pick agent-0
        st_env = state.view(Slen, Nenv, A, F)[:, :, 0, :]       # (S, Nenv, F)

        if self.use_rnn:
            # critic_h0: (num_layers, B, H)
            if critic_h0 is None or masks is None:
                raise ValueError("critic_h0 / masks missing with RNN enabled")

            num_layers, _, H = critic_h0.shape
            h_env = critic_h0.view(num_layers, Nenv, A, H)[:, :, 0, :]   # (L, Nenv, H)
            m_env = masks.view(Slen, Nenv, A, 1)[:, :, 0, :]             # (S, Nenv, 1)
        else:
            h_env, m_env = None, None

        # ----- critic forward  ----------------------------------------
        values_env, h_env_out = self.critic(st_env, h_env, m_env)        # (S, Nenv, 1)

        # ----- broadcast back to per-agent ----------------------------
        values = values_env.repeat_interleave(A, dim=1)                  # (S, Nenv·A, 1)

        return values, act_logp, dist_ent

    # ------------------------------------------------------------------
    # fallback: state is (B, F) (no sequence-first layout)
    # (Rare – happens if you disabled sequence batching.)
    # ------------------------------------------------------------------
    Btot, F = state.shape
    assert Btot % A == 0, "batch not divisible by n_agents"
    Nenv = Btot // A

    st_env = state.view(Nenv, A, F)[:, 0, :]                  # (Nenv, F)
    if self.use_rnn:
        if critic_h0 is None or masks is None:
            raise ValueError("critic_h0 / masks missing with RNN enabled")
        L, _, H = critic_h0.shape
        h_env = critic_h0.view(L, Nenv, A, H)[:, :, 0, :]      # (L, Nenv, H)
        m_env = masks.view(Nenv, A, 1)[:, 0, :]                # (Nenv, 1)
    else:
        h_env, m_env = None, None

    values_env, h_env_out = self.critic(st_env, h_env, m_env)  # (Nenv,1)
    values = values_env.repeat_interleave(A, dim=0).unsqueeze(0)  # (Btot,1) → add fake S=1

    return values, act_logp, dist_ent
```

### Option 2: Optimized `get_values` Method

This implementation focuses on the `get_values` method for rollout collection:

```python
def get_values(self, state, rnn_states=None, masks=None):
    with torch.no_grad():
        # Handle EP state type
        if self.state_type == "EP":
            B, n_agents = state.shape[0], self.n_agents
            assert B % n_agents == 0, "batch not divisible by n_agents"
            N = B // n_agents # n_rollout_threads(n_envs)

            state_env = torch.tensor(state, dtype=torch.float32,
                                    device=self.device).view(N, n_agents, -1)[:, 0] # (n_envs, s_dim)

            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states/masks missing with RNN enabled")
                h_env = torch.tensor(rnn_states, dtype=torch.float32,
                                        device=self.device).view(N,
                                                                n_agents,
                                                                *rnn_states.shape[1:])[:, 0]
                m_env = torch.tensor(masks, dtype=torch.float32,
                                        device=self.device).view(N, n_agents, 1)[:, 0]
            else:
                h_env, m_env = None, None

            values, rnn_states_out = self.critic(state_env, h_env, m_env) # (n_envs,1)

            # broadcast back so downstream code (GAE, PPO minibatching) stays unchanged
            values = values.repeat_interleave(n_agents, dim=0).cpu().numpy()
            if rnn_states_out is not None:
                rnn_states_out = rnn_states_out.repeat_interleave(n_agents, dim=0).cpu().numpy()

            return values, rnn_states_out

        # ----- compute-each-agent branch ---------------------------------------
        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # Handle RNN states and masks based on whether RNN is enabled
        if self.use_rnn:
            if rnn_states is None or masks is None:
                raise ValueError("rnn_states and masks must be provided when RNN is enabled")
            rnn_states = torch.tensor(rnn_states, dtype=torch.float32).to(self.device)
            masks = torch.tensor(masks, dtype=torch.float32).to(self.device)
        else:
            rnn_states = None
            masks = None

        # Get values and states
        values, rnn_states_out = self.critic(
            state,
            rnn_states,
            masks
        )

        # Convert values to numpy
        values_np = values.cpu().numpy()

        # Handle RNN states output
        rnn_states_out_np = rnn_states_out.cpu().numpy() if rnn_states_out is not None else None

        return values_np, rnn_states_out_np
```

## Implementation Decision

We decided not to implement these optimizations in the current version of the code. Instead, we've implemented a more general solution using accessor methods in the buffer that handle both state types transparently.

The current implementation:

1. Uses `get_state()` and `get_critic_rnn()` accessor methods in the buffer
2. These methods handle replication for environment-central states when needed
3. The rest of the code remains unchanged, maintaining a consistent interface

This approach is more maintainable and less error-prone, though it doesn't eliminate the redundant critic evaluations for EP state types.

## Future Work

In the future, we may implement these optimizations to improve performance, especially for environments with many agents per environment. The code examples provided here serve as a reference for how this could be implemented.

It's important to note that while the EP optimization can significantly improve performance, it's only applicable when all agents share exactly the same state. For the more common AS (Agent-Specific) case where global state is combined with agent-specific observations, we still need to run the critic once per agent, as each agent has a unique state representation.
