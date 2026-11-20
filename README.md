# MI (MAD Intelligence) 0.4.0 - your own Android coder, trained from scratch

MI is a small decoder-only Transformer that you train yourself, on your own device, from random
numbers. You type a request in plain English and it answers with Android (Kotlin/Java) source code.
No pretrained weights, no inference API, no cloud model is ever used. The only network use is
downloading **permissively licensed open-source code** as training data and (optionally) backing
your own files up to **your own** Hugging Face repository.

```
./start.sh
```

That one command fetches data, trains, fine-tunes, commits progress to Hugging Face as it goes, and
heals itself when it can. Details are below; every step can also be run by hand (section 6).

Contents: 1 Quick start - 2 What happens - 3 Options - 4 Self-healing - 5 Hugging Face backup -
6 Manual execution step by step - 7 Using the model - 8 Data sources and licensing - 9 Presets and
what to expect - 10 Moving between devices - 11 Troubleshooting - 12 File layout - 13 Tests - 14 Upgrading

---

## 1. Quick start

### Termux (Android phone)

```bash
pkg update
pkg install python python-numpy python-torch clang      # do NOT pip-install torch in Termux
cd ~/MI
export HF_TOKEN=hf_xxxxxxxx        # optional but recommended: a WRITE token, see section 5
./start.sh
```

### Linux / macOS / Windows (WSL)

```bash
python3 -m pip install torch numpy huggingface_hub
cd MI && export HF_TOKEN=hf_xxxxxxxx && ./start.sh
```

If `HF_TOKEN` is not set and you are at a terminal, `start.sh` asks for it once (press Enter to
train without cloud backup) and can remember it in `~/.config/mi/hf_token` (owner-only file).
Optional: `export GITHUB_TOKEN=...` raises GitHub's anonymous API limit (60 requests/hour).

Stop any time with **Ctrl+C**: a checkpoint is saved and pushed. Run `./start.sh` again and it
continues exactly where it stopped. Check progress with `./start.sh --status` and
`reports/latest_report.md`.

Good first runs:

| Goal | Command |
|---|---|
| Everything, default | `./start.sh` |
| Cap the total time (it fits the plan to the budget) | `./start.sh --hours 8` |
| Local only, no Hugging Face | `./start.sh --no-hf` |
| Quick smoke test on a phone (small data, tiny model) | `./start.sh --mb 5 --preset tiny --hours 1` |
| Start over (old files are moved to `backup_*/`) | `./start.sh --fresh` |

## 2. What happens (stages)

`start.sh` installs missing dependencies, loads your tokens, keeps the phone awake
(`termux-wake-lock`), then runs `mi_run.py`, which restarts itself from the last checkpoint if it
ever dies unexpectedly. `mi_run.py` runs these stages; finished stages are skipped on re-run:

1. **Preflight** - checks Python, PyTorch, free disk and RAM; picks a model preset from the RAM
   (`tiny` < 2.5 GB, `small`, `medium` >= 6 GB) unless you pass `--preset`.
2. **Hugging Face setup / restore** - finds your account (`hf auth whoami`), creates the **private**
   repo `<you>/MI-android-coder` if missing, and, if this device has no model files, downloads the
   latest checkpoint/state from that repo so work continues instead of restarting.
3. **Data** (`fetch_github_data.py`, `fetch_hf_data.py`, `fetch_hf_cli.py`) - about `--mb` MB of
   permissively licensed code (default 40 MB: half from GitHub, half from Hugging Face). Network
   trouble is retried; the run waits up to 90 minutes for a missing network.
4. **Prepare** (`prepare_data.py`) - removes duplicates, binary files, build output and anything
   that looks like a secret; writes `data/processed/manifest.json` (source + license of every item),
   the corpus, and request->code pairs: documented functions in the downloaded code
   (`/** doc */` above a `fun`) plus real commit messages (commit message -> code change).
5. **Tokenizer + model init** - trains the byte-level BPE tokenizer (vocab 1024 for `tiny`, 4096
   otherwise) on the code *and* the request wording, then creates `config.json`.
6. **Pretraining** (`mad.py train`) - in chunks of `--chunk-minutes` (default 15). After every chunk
   the checkpoint is validated, a known-good copy is kept, and progress is committed to Hugging Face.
   Training stops early when held-out loss stops improving (the model would only be memorizing).
7. **Instruction fine-tuning** (`mad.py finetune`) - teaches it to answer English requests. Same
   chunking, backup, early stopping and healing.
8. **Finish** - held-out evaluation, three sample answers, `reports/latest_report.md`,
   `data/THIRD_PARTY_NOTICES.md`, final push to Hugging Face.

Training length: `--steps` (exact), or by default `--epochs 6` passes over the corpus. With
`--hours N` the first chunk measures speed and the total step count is reduced to fit about 85% of
N hours (the rest is for fine-tuning and uploads).

## 3. Options (`./start.sh --help` shows all)

| Option | Default | Meaning |
|---|---|---|
| `--no-hf` | off | no Hugging Face backup at all |
| `--hf-repo OWNER/NAME` | `<you>/MI-android-coder` | which repo to back up to |
| `--public` | private | make the repo public (default is private: it is *your* model) |
| `--mb N` | 40 | MB of code to download in total |
| `--skip-data` | off | use only what is already in `data/raw/` |
| `--refresh-data` | off | download more even if data was fetched before (new repos each time) |
| `--no-hf-data` | off | GitHub only |
| `--no-hf-cli` / `--no-hf-java` | off | skip the `hf`-CLI commit dataset / its 58 MB Java part |
| `--preset auto/tiny/small/medium/large` | auto | model size (section 9) |
| `--vocab-size N` | 1024 tiny / 4096 | tokenizer size |
| `--steps N` / `--epochs E` | from corpus / 6 | pretraining length |
| `--sft-steps N` / `--no-sft` | from pair count | fine-tune length / skip it |
| `--hours H` | 0 (unlimited) | time budget |
| `--chunk-minutes M` | 15 | minutes between saves and Hugging Face commits |
| `--lr`, `--batch-size`, `--grad-accum`, `--threads` | 5e-4, 4, 4, half the cores | optimizer settings |
| `--patience N` | 12 | evaluations without improvement before early stop |
| `--fresh` | off | start over; old model files move to `backup_<time>/` (raw data is kept) |
| `--status` | | print progress and recent self-heals |

Environment: `HF_TOKEN`, `GITHUB_TOKEN`, `MI_HF_REPO`, `MI_HF_PUBLIC=1`, `MI_MAX_RESTARTS` (25).

## 4. Self-healing

Each training chunk is a separate process. When one fails, `mi_run.py` reads its output, classifies
the failure, changes what needs changing, and resumes from the last good checkpoint. Everything it
does is written to `reports/heals.md` and `logs/mi_run.log` (and pushed to Hugging Face).

| Failure | What it does automatically |
|---|---|
| Loss becomes NaN/infinite | halves the learning rate, resumes from the last good checkpoint (up to 6 times) |
| Out of memory / process killed (137) | halves the batch size and doubles gradient accumulation (same effective batch), then halves threads |
| Corrupt or unreadable checkpoint | quarantines it, falls back to the previous good copy, then the best checkpoint, then the Hugging Face copy |
| Disk full | deletes regenerable caches (`tokens.pt`, `random_init.pt`, staging folders, old logs) and retries |
| Network down / rate limit | retries with growing waits; training proceeds with the data already present |
| Hugging Face upload fails | logged, retried at the next chunk; training is never blocked or stopped by it |
| Corpus too small for the model | fetches 20 MB more, rebuilds the corpus and tokenizer |
| Missing input file | rebuilds data prep and tokenizer |
| PyTorch/numpy/huggingface_hub missing | installs it |
| Pipeline process itself crashes | `start.sh` restarts it (up to 25 times), it resumes from the checkpoint |
| Phone kills Termux / power loss | re-run `./start.sh`; loses at most 25 training steps (checkpoint interval) |
| Unrecognised error | retries 3 times with a pause, then stops, backs up the logs and tells you why |

Limits, stated plainly: this repairs the failure classes above; it cannot rewrite its own source
code. When it stops with exit code 2 it prints what needs a human (for example: you changed the
preset after training started, so the old checkpoint no longer fits: use `--fresh`).

## 5. Hugging Face backup (commit frequently, never lose work)

1. Create an account at https://huggingface.co, then a token at
   https://huggingface.co/settings/tokens with **Write** access.
2. `export HF_TOKEN=hf_...` (or paste it when `start.sh` asks).
3. Run `./start.sh`. It creates a **private model repo** `<you>/MI-android-coder` and commits to it:
   after every chunk (every ~15 minutes) and at the end. Uploads run in the background while
   training continues.

Pushed: `checkpoints/*.pt` (latest incl. optimizer, best), `training.jsonl`, `config.json`,
`tokenizer/mad_bpe.json`, `state/pipeline.json`, `data/manifest.json`, `data/THIRD_PARTY_NOTICES.md`,
`data/licenses/`, a redacted tail of `logs/mi_run.log`, `reports/`, and a generated model card.
Not pushed: the raw downloaded code (it is re-fetched from its original source) and your tokens
(logs are scrubbed of anything that looks like a token).

Storage note: each checkpoint push is a new file version. A `small` model is roughly 50 MB per push.
To keep the repo small, use a longer `--chunk-minutes`, or squash history occasionally with
`huggingface_hub`'s `super_squash_history`. Other commands: `python3 hf_sync.py status|push|pull`.

**Where is my backup?** When the repo is created or opened, `start.sh` prints a boxed banner with its
links, and saves them to `HF_BACKUP_LINK.txt` (also shown by `./start.sh --status` and
`python3 hf_sync.py status`):

```
 Cloud backup: PRIVATE model repo for account 'YOU'
   Repo:     https://huggingface.co/YOU/MI-android-coder
   Files:    https://huggingface.co/YOU/MI-android-coder/tree/main
   Commits:  https://huggingface.co/YOU/MI-android-coder/commits/main
```

It uses the account and token from `hf auth login` (or `HF_TOKEN`); you do not need to export anything
if `hf auth whoami` works. A private repo shows "404" to anyone not signed in as you.

**"Accept the terms" message.** That comes from a *gated dataset* (`bigcode/the-stack-smol`), not from
your backup repo. The `hf` CLI cannot accept terms; open the dataset link the message prints while signed
in to the same account as your token, click accept, and re-run. It is optional: skipping it loses nothing
for Android. The fetchers use your `hf auth login` token automatically (0.4.3+).

**Committing by hand, even while training runs.** Commits already happen automatically after every
chunk. To make one yourself at any moment, open a second Termux session (swipe from the left edge, "New
session") and run:

```bash
cd ~/MI && python3 hf_sync.py push -m "my note"
```

This is safe during training: it only reads the files (checkpoints are replaced atomically, so you always
get a complete one) and uses its own staging folder. It uploads the latest saved checkpoint, so you get
what the trainer last saved (every 25 steps). To restore later: `python3 hf_sync.py pull --force`.

**"Warning: non-finite grad_norm (inf)".** The trainer skips that one update and carries on, so it is not a
crash. A few per hundred steps is normal for a tiny model. If more than 5% of a chunk's steps are skipped,
the pipeline lowers the learning rate by 30% by itself (never below 1e-4) and logs it in `reports/heals.md`.

**What exactly is it training on?** After every data preparation the log prints a report; run it any time with
`python3 mi_tools.py data-report`:

```
== What the model trains on ==
By file type:   .kt 6.1 MB 52% | .java 3.0 MB | .xml 1.2 MB | .kts/.gradle ... | .toml ...
By source:      github | hf:JetBrains/KStack | hf:bigcode/commitpackft | local files
Left out:       not android-related, duplicates, possible secrets, license ... (details: data/processed/dropped.txt)
Training corpus: 10.5 MB -> 10,566,374 tokens (95% train / 5% validation)
```
The token count equals the number the trainer prints at the start of each chunk (`10,566,374 tokens`).
Records from the fetchers keep any file type. Plain files you put in `data/raw/` are accepted for `.kt .java .xml
.gradle .kts .json .py .md .txt .properties .pro .aidl .toml .yml .yaml .sh .cfg`; any other type is dropped and
the report names it (for example `.xyz x2`), so nothing disappears silently. Add types in `EXTENSIONS` in `prepare_data.py`.

**Searching for and adding data while it trains.** Between chunks the pipeline can refresh the data by itself:
* every `--refresh-hours` (default 6; `0` turns it off) it fetches `--refresh-mb` (default 10) of new code (new GitHub repos
  each time, more Hugging Face rows), rebuilds the corpus and keeps training on the larger corpus;
* **drop licensed files into `data/raw/`** from a second session and the next chunk boundary notices and includes them
  (no fetch, just a re-prepare, a few seconds);
* `touch data/REFRESH_NOW` asks for an immediate fetch + rebuild at the next chunk boundary.
The tokenizer stays fixed (the model was built for it), held-out items stay held out (the split is a stable hash of each
item), and the token cache re-encodes automatically. Training pauses for the fetch (about a minute or two). A failed
refresh never stops training; it keeps the previous corpus. The planned step count does not grow by itself:
raise it with `./start.sh --steps N` if the corpus grew a lot.

## 6. Manual execution, step by step

Everything `start.sh` does can be done by hand from the project folder. Use `python3` where Termux
says `python`.

```bash
# 0. dependencies
pkg install python python-numpy python-torch clang          # Termux
pip install -U huggingface_hub                              # gives the `hf` command
export HF_TOKEN=hf_...                                      # write token

# 1. fetch data (each is independent; run any or all)
python3 fetch_github_data.py --mb 20                        # permissively licensed GitHub repos
python3 fetch_hf_data.py --mb 14                            # JetBrains/KStack Kotlin (+ the-stack-smol with token)
python3 fetch_hf_cli.py                                     # commit instruction data via `hf download`
python3 fetch_hf_cli.py --discover "android kotlin"         # find more datasets (shows licenses)

# 2. prepare: dedupe, secret scan, manifest, corpus
python3 prepare_data.py
cat data/processed/train.txt data/processed/val.txt > data/processed/corpus.txt

# 3. instruction pairs (request -> code)
python3 mi_tools.py pairs-from-raw                          # from documented code
cat data/processed/sft_train.jsonl data/processed/sft_val.jsonl > data/instructions/hf_commits.jsonl
python3 mad.py prepare-instructions --input data/instructions

# 4. model + tokenizer (tokenizer reads the code AND the instruction wording)
python3 mad.py doctor                                       # RAM + speed test
python3 mad.py init --preset small --vocab-size 4096 --force
python3 mad.py train-tokenizer --vocab-size 4096

# 5. pretrain (steps = TOTAL target; resume with --resume)
python3 mad.py train --steps 3000 --max-minutes 30
python3 mad.py train --steps 3000 --resume checkpoints/mad_latest.pt
python3 mad.py eval
python3 diagnose.py                                         # optional: can this model learn at all?

# 6. fine-tune on English requests
python3 mad.py finetune --steps 400
python3 mad.py finetune --steps 400 --resume checkpoints/mad_sft_latest.pt

# 7. use it
python3 mad.py ask "In Kotlin: Show a toast with a message"

# 8. back up / restore by hand
python3 hf_sync.py push -m "manual checkpoint"
python3 hf_sync.py pull                                     # restore files missing locally
python3 mi_tools.py notices                                 # rebuild data/THIRD_PARTY_NOTICES.md
```

Put your own licensed pairs in `data/instructions/mine.jsonl` (one JSON per line:
`{"instruction": "In Kotlin: ...", "response": "..."}`, optional `"context"`). They are merged into
every fine-tune. `python3 mad.py status` shows which steps are done.

## 7. Using the model

```bash
python3 mad.py ask "In Kotlin: Adapter that shows a list of notes in a RecyclerView"
python3 mad.py ask                                  # interactive
python3 mad.py ask --context-file Foo.kt "In Kotlin: add a method that clears the list"
python3 mad.py generate "class MainActivity : AppCompatActivity() {"     # plain completion
```

`ask` warns when the checkpoint has not been instruction-tuned and when the output's brackets do
not pair up. Treat output as a draft: compile it and read it.

## 8. Data sources and licensing

Only these sources are used, and only with these rules:

* **GitHub** (`fetch_github_data.py`): official Android samples and widely used libraries plus
  search results (`language` x `license` x `topic:android`, most-starred first). A repo is used only
  if GitHub reports a license on the allowlist (Apache-2.0, MIT, BSD-2/3, ISC, Unlicense, CC0, 0BSD,
  Zlib, BSL-1.0, ...). If GitHub reports none, the repo's own LICENSE file must match exactly one
  permissive license and contain no copyleft text, otherwise the repo is skipped. Forks are skipped.
* **Hugging Face datasets**: `JetBrains/KStack` (per-file license, permissive rows only),
  `bigcode/commitpackft` (per-row license, permissive only), optionally `bigcode/the-stack-smol`
  (gated; accept its terms and set `HF_TOKEN`).
* **Never used**: your own apps or files, pretrained models, or model-generated datasets.

Every item keeps its source and license in `data/processed/manifest.json`; each GitHub repo's license
text is saved in `data/licenses/`; `data/THIRD_PARTY_NOTICES.md` lists all sources. Licenses such as
Apache-2.0/MIT/BSD ask you to keep copyright notices when redistributing the code itself. A
trained model is not a copy of that code, but whether weights are a "derivative work" is not settled
law everywhere, and this README is not legal advice. The notices file and the private-by-default repo
are there so you can show provenance; ask a lawyer before commercial redistribution of the weights.

**Ownership:** the architecture, tokenizer, training loop and weights are yours. Weights live in
your files and your private repo. Add a license file of your choosing to the repo if you share them.

**Android and Jetpack coverage.** Code is kept when it uses `androidx.*`, `android.*`, `@Composable` or
`com.android.*`, which covers Jetpack Compose, Room, ViewModel/Lifecycle, Navigation, DataStore, WorkManager,
Hilt and Material. The GitHub seed list includes Google's Compose, Now in Android, Sunflower and
Architecture Components samples. File types learned from GitHub repos: `.kt`, `.java`, Gradle build
files (`build.gradle`, `*.gradle.kts`, `settings.gradle*`), `gradle/libs.versions.toml` (version catalogs),
`proguard-*.pro`, `.aidl`, `AndroidManifest.xml`, and resource XML: `layout*/`, `menu/`, `navigation/`,
`xml/` (backup, network-security, shortcuts), `values/` and the dark-theme/API variants of
`themes.xml`/`styles.xml`/`colors.xml`/`dimens.xml`. Skipped on purpose: drawable vector XML, translated
`strings.xml`, images, build output, generated code. Request-and-code pairs are only extracted from
Kotlin/Java comments, so XML and Gradle files teach the model syntax but not English requests.
`commitpackft` and `KStack` contribute Kotlin/Java only. Put your own licensed files of any other type in
`data/raw/` and they are included (`prepare_data.py` accepts `.kt .java .xml .gradle .kts .toml .pro .aidl .json .properties`).

Adding sources: GitHub seeds are the `SEED_REPOS` list in `fetch_github_data.py`; Hugging Face
datasets go in `SOURCES` (`fetch_hf_data.py`, parquet-backed) or `FILE_SOURCES` (`fetch_hf_cli.py`,
JSONL files). Check each dataset's license first: `python3 fetch_hf_cli.py --card OWNER/NAME`.

## 9. Presets and what to expect

| Preset | Parameters (vocab 4096) | Context | Use |
|---|---:|---:|---|
| tiny | ~0.5M (vocab 1024) | 128 | smoke tests, weak phones |
| small | ~3.4M | 256 | default for phones with 2.5 GB+ free RAM |
| medium | ~7.5M | 256 | 6 GB+ free RAM |
| large | ~19M | 512 | a PC; select with `--preset large` |

Speed is the real limit on a phone: the trainer runs on CPU, and I could not measure phone speed
(run `python3 mad.py doctor` for yours). Millions of tokens will take hours to days on a phone. Use `--hours`, or train
on a PC with the same `./start.sh` and continue on the phone through the Hugging Face repo (section 10).

What this model is: a pattern and boilerplate generator for Android code, not a general programmer.
In an earlier measured test (README of 0.3.1) a 2.9M model on synthetic Kotlin reproduced learned patterns
almost perfectly but substituted names it had seen for new names in the request, and longer
fine-tuning made that worse. More varied real code, more pairs and a larger preset help most; always
judge it with `ask` on requests it has not seen, not by the training loss. The held-out loss in
`reports/latest_report.md` is the honest number.

## 10. Moving between devices

Train on whatever is fastest, then continue anywhere: on the new device install MI, set `HF_TOKEN`,
and run `./start.sh --hf-repo YOU/MI-android-coder`. With no local model files it downloads the
checkpoint, config, tokenizer and state from the repo and continues. Data is re-fetched from its
original sources. Checkpoints are plain PyTorch files.

## 11. Troubleshooting

* **"PyTorch is not installed"** - Termux: `pkg install python-torch`. Do not pip-install torch there.
* **Exit code 2 and a message** - it needs a decision from you; the message says what. Fix it and
  re-run `./start.sh` (it resumes).
* **"Hugging Face: not logged in"** - set `HF_TOKEN` (write access) or run `hf auth login`. Training
  continues locally meanwhile; the next push after login uploads everything.
* **GitHub rate limit** - `export GITHUB_TOKEN=...` (any classic token with no scopes works).
* **Phone gets hot / app killed** - `--threads 2`, keep it charged, keep `termux-wake-lock` active,
  disable battery optimization for Termux. Re-run `./start.sh`; it resumes.
* **Want to see what it did** - `./start.sh --status`, `reports/heals.md`, `logs/mi_run.log`,
  `checkpoints/training.jsonl` (loss, learning rate, grad_norm, tokens/s).
* **Changed preset/vocab after training started** - `./start.sh --fresh`.
* **Train longer later** - `./start.sh --steps 20000` (the total is raised and training continues).

## 12. File layout

```
start.sh              one-command entry point (deps, tokens, wake lock, auto-restart)
mi_run.py             the pipeline + self-healing orchestrator
hf_sync.py            Hugging Face repo create / commit / restore (uses the `hf` CLI)
mi_tools.py           helpers: count tokens, check checkpoints, extract pairs, license notices
fetch_github_data.py  permissively licensed GitHub repos -> data/raw/gh_*.jsonl
fetch_hf_data.py      Hugging Face parquet datasets via the Datasets Server REST API
fetch_hf_cli.py       Hugging Face JSONL datasets via `hf download`; also --discover/--card
prepare_data.py       dedupe, secret scan, manifest, corpus, instruction records
mad.py + mad/         the model, tokenizer, trainer, instruction tuning, CLI
diagnose.py           baselines and an overfit-one-batch learning check
tests/                unit tests + end-to-end tests (mock GitHub, stub hf, failure injection)
data/raw/             downloaded code (jsonl) - or put your own licensed files here
data/licenses/        license text of every GitHub repo used
data/instructions/    request->code pairs (yours + auto-generated)
data/processed/       corpus, manifest, sft.jsonl (regenerated)
checkpoints/          mad_latest.pt, mad_best.pt, mad_sft_*.pt, training.jsonl
state/pipeline.json   progress, learning-rate/batch decisions, self-heal history
reports/              latest_report.md, heals.md
logs/mi_run.log       everything the pipeline printed (tokens redacted)
```

## 13. Tests

```bash
pip install pytest
python3 -m pytest tests -q                                    # everything (about 10 minutes)
python3 -m pytest tests -q --deselect tests/test_pipeline_e2e.py   # fast unit tests only (~15 s)
```

The end-to-end tests run the real pipeline offline against a mock GitHub server and a stub `hf`
command, inject NaN/out-of-memory/kill/corrupt-checkpoint failures, simulate Hugging Face being
down or logged out, and check that a wiped device resumes from the repo copy. They do not
contact the real GitHub or Hugging Face; the first real run is the first real network test.

## 14. Upgrading

Unzip over your existing `~/MI` (`unzip -o`). The zip contains no data, checkpoints or config, so
nothing of yours is overwritten. Models trained with 0.3.x still load if you keep their
`config.json` and `tokenizer/` (the checkpoint forward pass did not change in 0.4.0).
`run_all.sh` still works: it now just calls `start.sh`.
