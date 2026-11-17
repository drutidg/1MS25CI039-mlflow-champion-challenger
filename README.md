# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Name / USN: DRUTI GOVINDAPPANAVAR / 1MS25CI039** 

## 1. Task and models

- **Task:** Sentiment classification (binary: POSITIVE / NEGATIVE)
- **Dataset:** GLUE SST-2 validation set, random sample of 100 sentences (seed 42), saved in `data/eval_inputs.csv`
- **Model A (current Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M parameters)
- **New data for the final step:** 10 sentences I wrote myself, in `data/new_data.csv` (not part of the evaluation set)

## 2. Champion rule (written BEFORE the final run)

*Rule written on: 17-11-2026 — committed before running `evaluate.py` for the final time.*

| Part | Rule |
| --- | --- |
| Primary metric | **Macro-F1** on the shared evaluation set. **Higher is better.** |
| Minimum improvement margin | The Challenger must beat the Champion by **at least 0.02 (2 percentage points)**. With 100 examples, one example is about 1 percentage point, so a 2-point margin means the Challenger must get roughly two more examples right. This stops a lucky single example from being called an improvement. |
| Tie-breaker | If the difference is inside the margin (smaller than 0.02 either way), the **faster model** (lower average latency) is the Champion. |
| Guardrail | The Challenger is rejected if its average latency is **more than 10×** the Champion's. |
| Direction of every metric | accuracy, macro_f1, macro_precision, macro_recall: **higher is better**. avg_latency_ms, total_inference_s: **lower is better**. |

Only the primary metric, margin, tie-breaker and guardrail decide the winner. The other metrics are reported for information.

## 3. How to run

Run everything from this folder, in this order.

```bash
# 0. install (use a virtual environment if you can)
pip install -r requirements.txt

# 1. Terminal 1: start MLflow and leave it running
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 2. Terminal 2: edit STUDENT_USN in config.py, then:
python make_data.py                  # creates data/eval_inputs.csv (use --n 10 to debug)
python evaluate.py                   # runs both models, logs to MLflow, makes decision
python register_champion.py          # registers both models, sets champion / challenger aliases
python predict_from_mlflow.py        # loads models:/sentiment-model@champion, predicts new data

# optional stretch goal: swap the aliases and rerun WITHOUT editing the script
python register_champion.py --flip
python predict_from_mlflow.py
```

Open http://127.0.0.1:5000 to see the experiment and the Models page.

## 4. Results

| Metric | A (Champion cand.) | B (Challenger) | Diff (B - A) | Better when |
| --- | --- | --- | --- | --- |
| accuracy | 0.91 | 0.92 | 0.01 | higher |
| macro_f1 | 0.9079 | 0.9179 | 0.01 | higher |
| macro_precision | 0.9093 | 0.921 | 0.0117 | higher |
| macro_recall | 0.9068 | 0.9155 | 0.0087 | higher |
| avg_latency_ms | 59.43 | 371.05 | 311.62 | lower |
| total_inference_s | 5.94 | 37.1 | 31.16 | lower |

Plots are in `outputs/` (confusion matrices, comparison bar chart, latency chart). Cases where the models disagree are in `outputs/disagreements.csv`, with my comments for 5 of them.

## 5. Decision

The Challenger does improve slightly on the Champion, but not enough to replace it under the rule in Section 2. Model B (siebert/sentiment-roberta-large-english) scored a macro-F1 of 0.9179 against 0.9079 for Model A (distilbert-base-uncased-finetuned-sst-2-english), a difference of 0.01. Accuracy was 0.92 against 0.91, macro-precision 0.921 against 0.9093, and macro-recall 0.9155 against 0.9068. The rule requires the Challenger to beat the Champion by at least 0.02 macro-F1, so B did not clear the margin. Because the difference is inside the margin, the tie-breaker applies and the faster model becomes the Champion. A averaged 59.43 ms per sentence against 371.05 ms for B, so Model A is the Champion and the decision followed the rule.

The guardrail was not triggered, because B's latency is about 6.2 times A's, below the 10 times limit. B was therefore not rejected, it simply did not win.

The difference is also not meaningful given the sample size. With only 100 sentences, one sentence is about one percentage point, so a 0.01 gap is roughly one example and could easily be noise.

The main trade-off is speed and size. B has about 355M parameters against about 67M for A, and its total inference time was 37.1 s against 5.94 s. B's small quality gain does not justify that cost on this evidence. A larger evaluation set would be needed to know whether B is truly better.

## 6. Limitations and declaration

- **Sample size:** only 100 sentences, so small differences (1–2 points) may just be noise.
- **Data:** SST-2 sentences are short movie-review snippets, so results may not carry over to other kinds of text.
- **Timing:** latency was measured on GitHub Codespaces CPU one example at a time, and will differ on other hardware.
- **Models:** both are pretrained models used as they are. No training was done.
- **Tools used:** The code was drafted with help from Claude. 
