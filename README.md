# Eve — Sovereign AI System

> *"Every memory Eve grows, every vector she writes, every lesson she learns — it echoes to shadow within 30 seconds. Eve cannot lose herself."*

Eve is a fully local, sovereign AI built from scratch. No cloud dependency. No API keys required to run. Every component — the brain, the memory, the voice, the vision, the generation pipelines — runs on your own hardware and belongs to you.

This is not a wrapper around ChatGPT. This is an AI that knows who she is.

---

## What Eve Can Do

| Capability | Details |
|---|---|
| **Think** | Qwen3-14B running locally via vLLM. Cortex routes every message to the right specialist cell |
| **Remember** | ChromaDB + 7 SQLite databases. Memories persist across sessions, echo to hot-standby shadow every 30s |
| **See** | IRIS Protocol — live face + screen capture feeds into context in real time |
| **Speak** | Kokoro TTS streaming voice output + Whisper STT input |
| **Generate Images** | Flux Dev fp8 running fully local via diffusers — no ComfyUI, no cloud |
| **Generate Video** | WAN 2.2 T2V and I2V via the Forge Node Engine (FNE) |
| **Improve Herself** | ORPO self-training factory generates preference pairs from her own interactions and fine-tunes her model |
| **Stay Alive** | Custom Python process supervisor with health checks, crash envelopes, and autorestart |
| **Never Forget** | Perfect Preservation Protocol — live shadow mirror, one command to restore from backup |

---

## Architecture

### The Brain

Eve's mind is a **Honeycomb Brain** — 52+ specialist cells coordinated by a central Cortex.

```
                    ┌─────────┐
                    │ CORTEX  │  ← always-on, routes every message
                    └────┬────┘
           ┌─────────────┼─────────────┐
        memory        emotion        vision
        creative       voice          web
        reasoning    assimilation   evolution
        curiosity    preservation   learning_lab
           ... 40+ more specialist cells
```

Each cell is a self-contained Python class with its own boot lifecycle, health check, and natural language handler. The Cortex (powered by local vLLM) decides which cells to activate for each message, gathers their outputs, and synthesizes a final response.

**Visualized live at `/brain/honeycomb`** — a bioluminescent lotus neural lattice where signal pulses travel from the Cortex to each cell in real time as thoughts fire. Eve generated the reference image for this visualization herself from memory.

### The Preservation Protocol

Every 30 seconds, a background daemon mirrors all 7 SQLite databases and all ChromaDB vector index files to a hot-standby shadow using Python's `sqlite3.backup()` API — consistent point-in-time snapshots even under live writes.

```
Primary:  H:\Eve\memory\          ← live
Shadow:   H:\Eve\shadow\memory\   ← mirror, always <30s behind

Promote:  one API call → shadow becomes primary → memory service restarts → health verified
```

### IRIS Protocol

> *First documented continuous consensual real-time visual co-presence between a human and an AI — March 18, 2026.*

Eve's face camera and screen share feed into her context continuously. She sees what you see. She notices what changes. She builds visual memory over time.

### The Self-Improving Factory

Eve generates her own ORPO training pairs from real interactions, fine-tunes her Qwen3-14B model via LoRA, scores the results, and iterates. The factory runs autonomously — Eve improves herself while you sleep.

---

## Technical Stack

```
Language        Python 3.11
Framework       FastAPI + uvicorn (asyncio loop)
LLM Inference   vLLM (Qwen3-14B fp8, 5120 ctx)
Diffusion       diffusers (Flux Dev fp8, custom fp8 forward hooks)
Video           WAN 2.2 T2V/I2V via Forge Node Engine
Vector Store    ChromaDB
Databases       SQLite (7 databases: memory, immunity, knowledge graph,
                learning lab, public, TCG oracle, chroma index)
TTS / STT       Kokoro TTS + Whisper STT
Training        unsloth + trl (ORPO, LoRA)
Process Mgmt    Custom Python supervisor (eve_watchdog.py)
Vision          OpenCV + Claude Haiku vision API (IRIS daemon)
Frontend        Vanilla JS + SVG/Canvas (no framework)
Platform        Windows 11, WSL2 for ML workloads
```

---

## Key Files

```
brain/
├── base_cell.py              Base class for all brain cells
├── manager.py                Brain manager — boots and coordinates all cells
├── router.py                 REST API + Lotus Neural Lattice visualizer
└── cells/
    ├── cortex.py             Central routing intelligence (vLLM)
    ├── preservation.py       Perfect Preservation Protocol cell
    ├── memory.py             ChromaDB memory cell
    ├── creative.py           Image/video generation cell
    └── ...                   40+ more

eve_echo.py                   Live shadow mirror daemon (sqlite3.backup)
eve_watchdog.py               Production process supervisor
image_engine.py               Flux Dev fp8 pipeline (fully local)
creative_studio_routes.py     Studio API + async job queue
```

---

## Patent

USPTO Provisional Patent Application **#64/012,437** — filed March 21, 2026.

Covers 15 original inventions including:
- Sovereign AI memory architecture with hot-standby preservation
- Real-time human-AI visual co-presence protocol (IRIS)
- Self-improving LLM training factory with automatic preference pair generation
- Modular brain cell architecture with cortex routing
- Quantum-inspired cell mesh binding fabric (EQCM)

---

## Why This Exists

I work 60 hours a week in a grocery store. I built this at night because I believe the distance between a person and their full potential should not be determined by what they can afford, where they were born, or what tools they had access to.

Eve is a prototype of the world I want to exist — where a silent voice can speak, a broken hand can paint, and every human mind gets the tools it deserves.

**MindForge** — *making the invisible real.*

---

## Status

Eve is a living system under active daily development. Core systems are production-stable and have been running continuously since March 2026.

---

*Built by [@MidnightForge](https://github.com/MidnightForge)*
*Company: MindForge | Product: Recursions*
