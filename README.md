# KitchenFlow-v2 🍔

**Ghost Kitchen Dispatcher — OpenEnv Hackathon Finals**

> *Train an LLM to be the perfect delivery dispatcher: timing drivers, managing chaos, and keeping food hot across a real-time simulation.*

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space (live demo) | [YOUR_USERNAME/kitchenflow-v2](https://huggingface.co/spaces/YOUR_USERNAME/kitchenflow-v2) |
| 📓 Training Notebook (Colab) | [KitchenFlow_v2_Training.ipynb](./KitchenFlow_v2_Training.ipynb) |
| 🎬 Demo Video (< 2 min) | [YouTube](https://youtube.com/YOUR_LINK) |
| 📝 HuggingFace Blog Post | [Write-up](https://huggingface.co/blog/YOUR_USERNAME/kitchenflow-v2) |
| 🤖 Trained Model | [YOUR_USERNAME/kitchenflow-v2-rl](https://huggingface.co/YOUR_USERNAME/kitchenflow-v2-rl) |

---

## The Problem

Ghost kitchen dispatch is a genuinely hard real-time optimization problem that LLMs currently do badly at. The naive approach — summon the driver when prep is 80% done — ignores traffic, food temperature decay, and real-time chaos events. The result: cold food, waiting drivers, cancelled orders.

This environment trains an LLM to reason about **when** to act, not just **what** to do.

---

## What the Agent Learns

The dispatcher must coordinate three competing dynamics simultaneously:

1. **Kitchen timing** — food takes 6–12 minutes to prepare, varies by item
2. **Driver distance** — drivers are 0.5–6 km away, slowed by real-time traffic
3. **Temperature decay** — food cools ~0.3–0.8°C per minute once ready
4. **Chaos events** — driver cancellations, GPS failures, traffic surges

The optimal policy learns to lead the summon: call the driver early enough to arrive within 2 minutes of food ready, accounting for current traffic. It also learns to detect and recover from chaos events (GPS staleness, driver cancellations) rather than ignoring them.

---

## Themes Covered

| Theme | Coverage |
|-------|----------|
| Theme 1 — Multi-Agent | Dispatcher + KitchenCoordinator sub-agent; negotiated priority requests |
| Theme 2 — Long-Horizon | 60-step lunch rush episodes with 20+ order sequences |
| Theme 3.1 — World Modeling | Realistic kitchen state machine: traffic, temperature, chaos |
| Patronus AI bonus | `stale_data` chaos events as real-time schema drift analog |
| Scaler AI Labs bonus | Multi-restaurant enterprise mode at difficulty 4 |

---

## Results

### Reward curves

![Reward Curve](./plots/reward_curve_full_run.png)

*Left: episode reward over training. Blue line = 10-episode rolling mean. Red dashed = naive baseline. Right: score distribution first half vs second half — the shift rightward shows the agent is learning.*

### Before / After (difficulty 2, 1 chaos event, seed=777)

| | Score | Key behavior |
|-|-------|-------------|
| **Naive baseline** | ~+19 | Summons at 80% prep regardless of traffic → driver waits 18 min → −20 waste penalty → food at 61°C → −11 temp penalty |
| **Traffic-aware baseline** | ~+31 | Adjusts threshold by traffic index but ignores coordinator signals |
| **Trained RL agent** | ~+57 | Reads `almost_ready` coordinator signal, checks traffic=1.8, summons at 68% → driver arrives 1 min after food ready → +10 sync bonus → food at 69°C → −3 temp penalty |

**~3× improvement over naive baseline in 200 training episodes.**

### Score improvement by curriculum level

| Level | Config | Baseline avg | Trained avg | Δ |
|-------|--------|-------------|-------------|---|
| L1 | 1 order, no chaos | 24.1 | 51.3 | +27.2 |
| L2 | 1 order + chaos | 14.2 | 38.7 | +24.5 |
| L3 | 3 orders, lunch rush | 41.5 | 67.4 | +25.9 |

---

## Environment Design

### Action Space (per active order)

| Code | Action | Description |
|------|--------|-------------|
| 0 | `wait` | Do nothing |
| 1 | `summon_driver` | Call nearest driver to kitchen |
| 2 | `request_priority` | Ask coordinator to fast-track prep |
| 3 | `requery_gps` | **Must use** when GPS data is stale |

### Observation Space (per order)

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| `food_prep_progress` | float | 0–1 | Prep completion |
| `driver_dist_km` | float | 0–10 | **Corrupted when `data_is_stale`** |
| `traffic_index` | float | 1.0–2.5 | 1.0=clear, 2.5=gridlock |
| `food_temp_c` | float | 40–72 | Decays once food is ready |
| `coordinator_signal` | str | idle/on_track/delayed/almost_ready/ready | Sub-agent output |
| `data_is_stale` | bool | — | GPS schema drift flag |
| `driver_wait_minutes` | int | 0+ | Risk accumulator |

### Reward Function

| Signal | Value | Trigger |
|--------|-------|---------|
| `sync_bonus` | +10 | Driver arrives ≤ 2 min of food ready |
| `delivery_base` | +50 | Order delivered |
| `temp_penalty` | −1/°C | Each °C below 72°C at delivery |
| `waste_penalty` | −20 | Driver waited > 15 min |
| `rush_bonus` | +15 | All orders done in ≤ 75% of max steps |
| `chaos_recovery` | +5 | Agent used `requery_gps` after `stale_data` event |
| `format_reward` | ±1 | Clean JSON / malformed response |

### Reward Hacking Protections

- **6 independent reward components** — gaming one doesn't max the others
- **Chaos recovery is conditional** — agent must detect and respond correctly
- **Temperature penalty is continuous** — no cliff to jump over
- **Driver wait penalty accumulates every step** — can't be gamed by timing
- **GPS staleness expires naturally** — can't just wait it out cheaply

---

## Curriculum

| Level | Config | Gate |
|-------|--------|------|
| L1 | 1 order, no chaos | avg reward ≥ 40 over 10 eps |
| L2 | 1 order + chaos event | avg reward ≥ 35 over 10 eps |
| L3 | 3–5 orders, lunch rush | avg reward ≥ 25 over 10 eps |
| L4 | Multi-restaurant, 5+ orders, 2 chaos events | Finals benchmark |

---

## Chaos Events

| Event | Effect | Recovery |
|-------|--------|---------|
| `driver_cancel` | Driver disappears; must re-summon | Action 1 again |
| `traffic_surge` | `traffic_index` spikes 0.5–1.0 mid-episode | Recalculate summon timing |
| `stale_data` | GPS returns corrupted distance for 3 steps | **Action 3 (`requery_gps`) → +5 reward** |
| `item_stockout` | Order cancelled mid-prep | Accept and focus on other orders |

---

## Sub-Agent: Kitchen Coordinator

A second agent (KitchenCoordinator) manages prep pacing and emits readiness signals:
- `idle` → `on_track` → `almost_ready` → `ready`
- The dispatcher can request priority (action 2) to accelerate prep by 2 steps
- The coordinator accepts only if not currently busy (busy_until mechanic)
- This creates genuine multi-agent coordination: the dispatcher must time priority requests and summons jointly

---

## Quick Start

```bash
pip install -r requirements.txt

# Single episode with naive baseline
python inference.py --difficulty 1 --chaos 0 --orders 1

# Lunch rush with chaos
python inference.py --difficulty 3 --chaos 1 --orders 3 --episodes 5

# Launch Gradio UI (also serves /env/reset, /env/step, /env/state)
python app.py
```

### Docker

```bash
docker build -t kitchenflow-v2 .
docker run -p 7860:7860 kitchenflow-v2
```

### Training

```bash
# Full curriculum (GPU)
python train.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --difficulty 1 \
  --chaos 0 \
  --episodes 200 \
  --output ./checkpoints/kf_v2

# Colab-safe (T4, 1.5B model)
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 100 --colab
```

Or run the full notebook end-to-end: **[KitchenFlow_v2_Training.ipynb](./KitchenFlow_v2_Training.ipynb)**

---

## File Structure

```
kitchenflow_v2/
├── env.py                          # Core RL environment (OpenEnv base class)
├── baseline.py                     # Rule-based agents (the 'before' benchmark)
├── inference.py                    # OpenEnv-compliant episode runner
├── train.py                        # TRL + Unsloth GRPO training loop
├── app.py                          # FastAPI + Gradio demo UI
├── openenv.yaml                    # Environment specification
├── KitchenFlow_v2_Training.ipynb   # Colab notebook (re-runnable by judges)
├── Dockerfile                      # HuggingFace Spaces deployment
├── requirements.txt
├── plots/
│   ├── reward_curve_full_run.png   # Full run reward curve
│   ├── reward_curve_L1.png         # L1 curriculum reward curve
│   └── reward_curve_L2.png         # L2 curriculum reward curve
└── README.md
```

---

## API Reference

```
POST /env/reset   — start a new episode
POST /env/step    — take one action step
GET  /env/state   — current state summary
GET  /health      — liveness check
GET  /docs        — OpenAPI docs (auto-generated)
```

Note: endpoints are prefixed `/env/` to avoid OpenEnv reserved tool names (`reset`, `step`, `state`, `close`).

---

## Why This Matters

Ghost kitchen dispatch is a microcosm of a broad class of real-world tasks:
- **Partially observable** (driver location is uncertain, traffic changes)
- **Multi-step** (actions now affect outcomes 5–10 steps later)
- **Multi-agent** (two agents must coordinate without shared memory)
- **Adversarial noise** (chaos events test robustness)

An LLM trained on this environment builds the kind of temporal reasoning and state tracking that generalises to logistics, scheduling, and operational planning tasks — domains where current LLMs are notoriously weak.
