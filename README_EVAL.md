# RoboTwin Evaluation

Evaluation runs through two components:

- `eval.sh` (this project root) — the entry point. It shards the task list,
  launches several worker processes in parallel, and aggregates their results.
- `eval_vla_bridge.py` — the evaluation loop. **It must be placed in the
  RoboTwin root directory**, NOT in this project:

  ```
  /home/yf/Desktop/Code/VLA/RoboTwin/RoboTwin/eval_vla_bridge.py
  ```

## Why the bridge lives in the RoboTwin directory

RoboTwin internally uses many paths relative to its root (e.g. `./assets/...`,
`./task_config/...`). The bridge therefore runs with the RoboTwin root as its
working directory, so all of those paths resolve without modifying any
existing RoboTwin code. The bridge is a NEW file — no RoboTwin code is changed.

If you move or re-clone RoboTwin, copy `eval_vla_bridge.py` into the new
RoboTwin root and update `ROBOTWIN_ROOT` in `eval.sh` accordingly.

## Usage

```bash
conda activate RoboTwin     # or: PYTHON=/path/to/python bash eval.sh
bash eval.sh
```

To watch per-task success rates live while eval.sh is running, open a second
terminal:

```bash
python monitor_eval.py      # refresh every 2s (pass a number to change)
```

It shows `suc/done/target` and the current rate for every running task, plus
finished tasks and an overall line.

Edit the top of `eval.sh` to configure the run:

- `TASKS` — uncomment the tasks to evaluate
- `NUM_WORKERS` — number of parallel worker processes (default 4); tasks are
  distributed round-robin, each worker evaluates its shard sequentially in a
  single process
- `CKPT_SETTING` / `CHECKPOINT_EP` — checkpoint to evaluate
- `TEST_NUM` — episodes per task
- `TASK_CONFIG` / `INSTRUCTION_TYPE` — `demo_clean`/`demo_randomized`,
  `unseen`/`seen`

## Outputs

- `eval_result/eval_summary.txt` — one line appended as soon as each task
  finishes (timestamp, checkpoint, task, success count, rate); survives
  interrupted runs
- `eval_result/logs/worker_<i>.log` — per-worker logs
- `eval_result/bridge_results/worker_<i>.json` — machine-readable results
- `eval_result/<task>/test_policy/<task_config>/<ckpt>/<timestamp>/result.txt`
  — per-task result (and videos if `eval_video_log` is enabled)

The final per-task and average success rates are printed at the end of the
`eval.sh` run.
