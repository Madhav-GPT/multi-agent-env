# SPECTRA: Coordinating Specialists Under Partial Observability

SPECTRA is an OpenEnv benchmark for multi-agent incident response. Three specialist agents each see only one slice of the environment:

- metrics
- logs
- security alerts

The commander never sees raw state. It receives only typed specialist reports and must act under uncertainty.

## Why it matters

Real incidents are not solved from a single dashboard. Operations, application, and security teams each hold incomplete evidence. SPECTRA turns that coordination problem into a deterministic RL environment with explicit reward signals for:

- resolution
- root-cause targeting
- specialist coordination
- step efficiency
- trust calibration

## What is novel

The partitioning is enforced in code with Pydantic schemas, not prompt instructions. A specialist cannot access the wrong channel because those fields do not exist in its typed observation.

## Training setup

- Rule-based specialists during training
- Qwen-family specialists in demo mode
- Qwen2.5-3B commander trained with GRPO

## Takeaway

SPECTRA shows that partial observability plus structured specialist communication yields a cleaner and more realistic multi-agent benchmark than prompt-only roleplay.

