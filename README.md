# qualk: a quantum walk that decides what to be curious about next

An explorer keeps a **world graph** as its memory. Each round it must decide *what to look at next*. Here that
choice is made by a **continuous-time quantum walk over the graph**, run as a real Qiskit circuit locally, or on
Moth Atlas. The chosen concept drives a web search, an LLM reads the page, and what it finds is added to the graph,
which changes the next walk. The graph populates itself, and open questions give the population direction.

- **Part of the "artificial curiosity" mechanics of [Assembly](https://wesual.art/assembly)**: this repo extracts the
  quantum probe walk, the world graph and the question ("thread") lifecycle from Assembly's autonomous exploration
  mode ("dog") into a small standalone system, with a web UI to run, steer and inspect it.
- **Submission for Moth Hack 2026.**

```
world graph --(seed concept)--> quantum walk over its neighbourhood --(measure)--> partner concept
     ^                                                                                   |
     |                                                                       probe = seed + partner
 LLM extracts concepts, relations,  <--  fetch a fresh, non-duplicate page  <--  web search for the probe
 and open questions
```

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11
python app.py --simulate                 # http://localhost:8765 : keyless demo, synthetic pages (banner says so)
python app.py                            # real runs: open Settings, add a search key and an LLM endpoint
```

Then **New run** (a topic and/or seed text), tick *also run the diffusion control* for a paired experiment, and
watch it grow. Keys live in `.env` (see `.env.example`); the Settings panel edits it and can test each provider.
A command-line runner exists too: `python run.py run -o runs/demo --topic "Voynich manuscript" -n 12`
(then `python run.py view -o runs/demo` for a static replay page).

## The quantum walk (`qualk/quantum_walk.py`)

1. **Arena.** From the seed concept, greedily collect up to N neighbours (default 12) along the strongest *relation*
   edges (causal, stance, epistemic...; structural edges like `part_of` are never walked). Each concept is a qubit.
2. **Hamiltonian.** The coupling matrix is the graph itself: weight = `confidence x (0.4 + 0.6 x embedding affinity)`,
   **negative for hostile relations**.
3. **Circuit.** One excitation on N qubits (`|00100>` = "attention is on concept 3"). Each edge gets an
   `RXX(theta) RYY(theta)` pair (a partial swap that moves the excitation) in a symmetric second-order Trotter
   schedule, which realises `exp(-iAt)` on the one-excitation subspace.
4. **Measurement.** Shots return concepts; the first eligible one becomes the probe's partner (the seed and recently
   used concepts are excluded).
5. **Control.** `diffusion` runs a classical random walker on *exactly the same window and couplings*, so the only
   difference is wave interference versus probabilistic hopping. On a ring of four equal concepts the quantum walker
   reaches the opposite one with probability 1.00 (diffusion: 0.25); flipping one edge's sign moves the mass
   sideways, which diffusion cannot see (`tests/test_quantum_walk.py`).

**Showing the walk honestly.** A quantum walker has no path: before measurement it is a superposition, and the
measurement yields only an endpoint, so the UI never draws a line for it. Instead, each round's panel has a
**wave view** (`evolution.py`): both walks on the same window, with circle size showing the chance of finding the
walker on each concept as time grows from 0 to `t`, a scrubber with Play, and the chosen partner's probability over
time for quantum and diffusion. It is the exact ideal evolution, recorded per round (runs made before it existed show
a note instead). The run's actual path is the **exploration trail** on the graph: one numbered arrow per probe from
its seed to its partner (colour: fresh, variation, recombination, question-aimed, pinned), switchable between the
last 8 probes, all, or off; clicking an arrow jumps to its round.

Circuits run on **local Qiskit**, the **Moth Atlas emulator** (`atlas`, provider `aer`) or a **real QPU** via Atlas (`qpu`),
chosen per run in the app (or `--quantum_backend`). On Atlas (`MOTH_API_KEY`) a failed job falls back to local Qiskit
for that pick and, after 3 failures in a row, for the rest of the run, unless `MOTH_ATLAS_STRICT=1`; a `qpu` run never
falls back and stops instead. Windows over 16 qubits use an exact closed form (verified equal to the circuit) or Atlas's
`matrix_product_state`. Every executed circuit is saved as OpenQASM (`quantum/round-XXXXX.qasm`). The round panel
shows both distributions per qubit, the difference between them, circuit error, backend and Atlas job.

## Direction: open questions ("threads")

Without direction a growing graph is an amorphous cloud. Threads, ported from Assembly, give the run questions to
work on:

- **Raised** by the extractor when a page leaves something unanswered, grounded on concepts it involves (ungrounded
  ones are dropped, near-duplicates merged).
- **Advanced or answered** by a separate LLM pass (`prompts/thread-upd.md`) that checks each new page against up to
  five open questions. It can never raise one.
- **Let go** when untouched: pressure decays mechanically (never the model's estimate) and the pool is capped.
- **Aimed at**: with probability `thread_aim` a probe starts from a question's own concepts; the quantum walk still
  picks the partner, so the walk explores *toward* something.

The **Questions** tab shows answered-of-raised, the open/answered curve, every question's evidence trail and an
activity feed; the graph draws each open question as a halo around its concepts. You can ask your own question,
focus all probes on one, or close one.

## Graph hygiene: keeping the graph meaningful

Claims used to mint nodes for whatever they mentioned, so figures, dates and one-off names became concepts with
nothing but a name (in Assembly: 229 of 564 nodes in one run), which make poor search queries and useless walk
windows. Three mechanics, all from Assembly, prevent that:

- **Node gate** (`nodegate.py`, on by default): before anything reaches the graph, a concept without a one-sentence
  description is dropped, and so is a claim unless both its ends are established concepts or described concepts
  declared in the same extraction (the object of an `attr` claim is plain text, not an end). Threads pass, but the
  gate runs first, so a question can only rest on concepts it has accepted. Drops are counted per page and shown in
  the round panel. The extraction prompt tells the model the same rules.
- **Orphan focus** (`orphan_focus`, default 3): each page is read with the nearest described concepts that have no
  relation yet shown to the extractor, so the page can connect them; no extra LLM call.
- **Probe spread**: only described concepts are probe candidates; the draw weight is `1 / (1 + visits)^2` and a
  concept used in one of the last 24 probes is not picked again as a walk partner (`exploration.py`).

Defaults follow Assembly's current ones (`explore` 0.7, `thread_decay` 0.93). Switch the gate off in New run or
with `--no-node-gate` to see the difference.

## Similarity, novelty and duplicates

Similarity is the **cosine of text embeddings** (`BAAI/bge-small-en-v1.5`, 384-d; 1 = same, about 0.45 = unrelated).
It is used for relation strength, page novelty, merging questions (0.85) and retrieval. Only the first ~512 tokens
of a page are embedded. Measured against one page: copy with boilerplate 0.99, paraphrase 0.92, another article on
the same topic 0.75, unrelated 0.45.

- **Novelty** = 1 minus the similarity to the closest of the last 32 pages read (0.5 for the first page).
- **Near-duplicate filter** (`dedupe`, default 0.10, live-editable, 0 = off): a fetched page closer than the threshold
  to *any* page already read is skipped before extraction (no LLM call) and the next search result is tried. Skips
  are shown in the round panel and logged in `harvest.skipped`. Pasted or injected text is never filtered.

## Web app

- **Run and compare**: quantum / diffusion / classical walks; a paired run digests the seed once, copies it, and
  advances quantum and control in lockstep (`<id>` and `<id>-ctl`); the Compare tab charts concepts, relations,
  novelty, concept overlap and questions answered for both.
- **Monitor**: the graph grows live (WebSocket); a slider replays any round; click a node to centre it and see its
  labelled links; each round reads as *choose, search, read, what changed*.
- **Steer** (applied at the next round boundary): pause / step / stop / continue, live walk and question
  parameters, pin a concept as the next seed, inject text or a public URL, ask or focus a question. Logged to
  `steer.jsonl`.
- **Manage**: delete runs (paired runs together; only files a run wrote are removed, never recursively).
- **Access**: there is no login by default. The server listens on `0.0.0.0` and prints a note; use
  `--host 127.0.0.1` to keep it local, or `--token T` (or `QUALK_TOKEN`) to require a token, plus TLS
  (`--ssl-certfile/--ssl-keyfile`) if it is reachable from an untrusted network. Secret keys are write-only in the UI.
  URL injection refuses non-public addresses. One run group is active at a time; rounds per run are capped
  (`--max-rounds`, default 200).

## Configuration

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY`, `SERPER_API_KEY`, `BRAVE_API_KEY` | web search (first configured provider wins; `SEARCH_PROVIDER_ORDER`) |
| `QUALK_LLM_URL`, `QUALK_LLM_KEY`, `QUALK_LLM_MODEL` | any OpenAI-compatible endpoint (default LM Studio, `gpt-oss-20b`) |
| `MOTH_API_KEY` | run the walk circuit on Moth Atlas |
| `MOTH_ATLAS_PROVIDER`, `MOTH_ATLAS_BACKEND`, `MOTH_ATLAS_TIMEOUT`, `MOTH_ATLAS_STRICT` | the Atlas emulator target (default `aer`), job timeout in seconds, and `1` to fail instead of falling back to local Qiskit |
| `MOTH_QPU_PROVIDER`, `MOTH_QPU_BACKEND`, `MOTH_QPU_TIMEOUT` | the real-QPU target (needs the account feature `run_quantum`); never falls back to a simulator |
| `QUALK_EMBED_MODEL` | embedding model (the affinity floor is calibrated per model: `python -m qualk.calibrate`) |
| `QUALK_TOKEN` | optional web-app access token |

## Repository map

| Path | What |
|---|---|
| `qualk/quantum_walk.py`, `atlas.py` | window, couplings, circuit, diffusion control, Moth Atlas client |
| `qualk/exploration.py`, `threads.py`, `nodegate.py` | probe archive (fresh / mutated / crossed / question-aimed), the thread keeper, the node gate |
| `qualk/world/` | the world graph: nodes, claims, dedupe, `edge_role`, persistence |
| `qualk/engine.py`, `digest.py`, `web.py`, `llm.py` | the round loop, page-to-graph extraction, search and fetch, LLM calls |
| `qualk/semantic.py`, `embed.py`, `calibrate.py` | embeddings and per-edge affinity |
| `qualk/runner.py`, `server.py`, `security.py`, `settings.py`, `static/`, `app.py` | the web app |
| `qualk/viewer.py`, `report.py`, `fakes.py`, `run.py` | static replay, offline quantum-vs-diffusion analysis, synthetic sources, CLI |
| `tests/` | offline tests (walk, Atlas client with a fake server, engine, threads, dedupe, runner, server) |

A run folder holds `world.json` (graph with per-edge affinity), `world.jsonl` (op log), `rounds.jsonl` (every round:
probe, query, source, graph delta, questions, full walk trace), `steer.jsonl`, `quantum/*.qasm` and `config.json`.

```bash
python -m unittest discover -s tests -t .      # offline; no keys or network needed
python -m qualk.report -i runs/demo            # quantum vs diffusion on a finished graph
```

## Honest limits

- **Real interference, no computational advantage.** One excitation on `n` qubits uses `n` of `2^n` states; the same
  probabilities are an `n x n` matrix exponential a laptop computes in microseconds (the exact reference here). What
  the circuit adds is a hardware-executable, measurement-based formulation of the choice. No speed-up is claimed.
- The `qpu` backend is implemented but **not yet run on hardware**: it needs the account feature `run_quantum` and the
  provider/backend names from Moth (`MOTH_QPU_*`), and the Atlas engine used (`tomography-api-v2`) has no QPU mode switch.
  All results so far are from the emulator or local simulation; on hardware, noise would push the result toward the
  diffusion twin, itself a measurable comparison.
- The walk differs from diffusion mostly where the relation graph has loops or hostile edges; early graphs are
  tree-like (`qualk.report` quantifies it).
- Signal passes through web search and an LLM. Whether a page "answers" a question is the LLM's judgement, so read the
  evidence trail, not just the count. Similarity and thresholds are calibrated on one small embedding model.
