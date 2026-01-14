# Introduction
Code for paper: [Multi-hop Reasoning via Early Knowledge Alignment](https://arxiv.org/abs/2512.20144). We show here the implementation based on [Graph-R1](https://github.com/LHRLAB/Graph-R1).

## Repository Layout

- `graphr1/`: retrieval + hypergraph core.
- `agent/`: agent/tool wrappers.
- `verl/`: RL training stack (GRPO / PPO / REINFORCE++).
- `script_build.py`: build Knowledge HyperGraph and FAISS indexes.
- `script_api.py`: retrieval API server (port 9001).
- `get_knowledge.py`: retrieve top-k early knowledge per question.
- `script_process.py`: build parquet data with early-knowledge prompts.
- `run_grpo.sh`, `run_ppo.sh`, `run_rpp.sh`: training entrypoints.
- `evaluation/`: evaluation helpers.

## Environment Setup

```bash
conda create -n eka python==3.11.11
conda activate eka
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

If you run the retrieval server, make sure `faiss`, `FlagEmbedding`, `fastapi`, `uvicorn`, and `datasets` are included in `requirements.txt` for your environment.

## Data Preparation

We conduct experiments on six datasets: 2WikiMultiHopQA, HotpotQA, Musique, NQ, PopQA, and TriviaQA. Graph-R1 has the download link from [TeraBox](https://1024terabox.com/s/1wvopC4wO60nLzcc96HnSOg) and you can place them under `datasets/`. We prepare another link [https://pan.baidu.com/s/1dlVYbApjagclIlN7FY4mNw?pwd=6yar] for researchers not available to TeraBox.

Expected raw layout:

```
datasets/<DATASET>/raw/qa_train.json
datasets/<DATASET>/raw/qa_dev.json
datasets/<DATASET>/raw/qa_test.json
datasets/<DATASET>/corpus.jsonl
```

### 1. Build Knowledge HyperGraph

`script_build.py` reads `datasets/<DATASET>/corpus.jsonl` directly and writes to `expr/<DATASET>`. No parquet is needed for this step. Provide `openai_api_key.txt` in the repo root.

```bash
python script_build.py --data_source 2WikiMultiHopQA
```

Pre-built Knowledge HyperGraph download: Graph-R1 has the download link from [TeraBox](https://1024terabox.com/s/1SZaGxO1VWD2YGWnYSIJFuA), You can place the dataset folders under `expr/`. We prepare another link [https://pan.baidu.com/s/1T-7NvLnI8tUlkZ63I0B5tA?pwd=y38j] for researchers not available to TeraBox.
### 2. Start Retrieval Server

```bash
python script_api.py --data_source 2WikiMultiHopQA
```

### 3. Retrieve Early Knowledge

```bash
python get_knowledge.py --data_source 2WikiMultiHopQA --top_n 5 --api_url http://localhost:9001
```

### 4. Convert to Parquet

```bash
python script_process.py --data_source 2WikiMultiHopQA
```

Reads:
`ik_datasets/<DATASET>/datasets_raw_with_top_5_knowledge`

Writes:
`ik_datasets/<DATASET>/datasets_processed_with_top_5_knowledge`

## Training

```bash
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m Qwen2.5-3B-Instruct -d 2WikiMultiHopQA -u <unique_id>
```

Training scripts expect:
`datasets/<DATASET>/processed_with_top_5_knowledge`. If needed, move the parquet files or create a symlink.

## Evaluation

See `evaluation/README.md`.

## Acknowledgements

Some retrieval and training components build on prior open-source Graph-R1. Thanks to the community for releasing these building blocks.
