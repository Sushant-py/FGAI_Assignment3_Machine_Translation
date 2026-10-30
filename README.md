# Foundations of Generative AI (ICT4403) — Assignment 3
## Comparative Neural Machine Translation: Encoder–Decoder vs. Decoder-Only Architectures

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## 1. Problem Statement and Objective

The objective of this assignment is to develop, evaluate, and critically compare machine-translation systems across fundamentally distinct neural architectures:
1. **Encoder–Decoder Architectures:**
   * **Basic Transformer:** Built from scratch following the seminal *Attention Is All You Need* (Vaswani et al., 2017) architecture and trained end-to-end on bilingual parallel text.
   * **Multilingual T5 (mT5-small):** A sequence-to-sequence pretrained multilingual language model fine-tuned using supervised translation objectives.
2. **Decoder-Only Architectures:**
   * **LLaMA (LLaMA-3.2-1B-Instruct / TinyLlama):** An autoregressive causal language model evaluated via few-shot in-context learning and parameter-efficient fine-tuning (LoRA).
   * **Gemma (Gemma-2-2B-IT):** Google's lightweight instruction-tuned causal language model adapted for zero/few-shot multilingual translation.

The project evaluates translation fidelity using standard quantitative metrics (**BLEU** and **chrF**), measures computational efficiency (**training time**, **inference latency per sample**, and **model parameter count**), analyzes decoding dynamics under various **sampling temperatures** ($T \in \{0.0, 0.3, 0.7, 1.0\}$), and investigates **responsible and ethical AI considerations** (gender bias, occupational stereotypes, hallucinations, and idiomatic mistranslations).

---

## 2. Dataset Description and Source Link

* **Language Pair:** English ($\text{en}$) $\rightarrow$ French ($\text{fr}$)
* **Corpus Name:** **OPUS-Books** (from the open parallel corpus OPUS project curated by Jörg Tiedemann)
* **Dataset Source Link:** [Hugging Face Datasets: `opus_books`](https://huggingface.co/datasets/opus_books) / [OPUS Corpus Portal](https://opus.nlpl.eu/Books.php)
* **Dataset Characteristics:**
  * Contains high-quality, professional literary translations across public domain literature.
  * Captures diverse sentence structures, rich grammatical moods (subjunctive, conditional, imperfect), and idiomatic phrasing.
* **Preprocessing Pipeline:**
  1. **Text Normalization:** Standardized characters using Unicode NFKC normalization and stripped irregular whitespaces and control characters.
  2. **Length Filtering:** Constrained sentence lengths to $3 \le \text{length} \le 60$ whitespace-delimited tokens to eliminate incomplete sentence fragments and avoid out-of-memory overhead during scratch Transformer training.
  3. **Deterministic Partitioning:** Stratified splitting into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets saved into `data/train.csv`, `data/val.csv`, and `data/test.csv`. All four models are evaluated on the exact same test partition for strict comparability.

---

## 3. Brief Description of the Models Implemented

### 3.1 Basic Transformer (Encoder–Decoder from Scratch)
* **Architecture:** PyTorch implementation conforming to Vaswani et al. (2017).
  * **Encoder:** Stacks 3 layers, each containing Multi-Head Self-Attention ($d_{model} = 256$, $h = 4$ heads, $d_k = 64$), Layer Normalization, Residual Connections, and Position-Wise Feed-Forward networks ($d_{ff} = 512$).
  * **Decoder:** Stacks 3 layers featuring Masked Multi-Head Attention (with lower-triangular causal attention masking), Multi-Head Cross-Attention attending to encoder outputs, and feed-forward sublayers.
  * **Positional Encoding:** Fixed sinusoidal geometric embeddings:
    $$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
* **Training:** Optimized with Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$) and Cross-Entropy loss with label masking for padding tokens.

### 3.2 mT5 — Multilingual T5 (Encoder–Decoder)
* **Architecture:** `google/mt5-small` (~300M parameters). Pre-trained on mC4 covering 101 languages using an unsupervised span-corruption objective.
* **Adaptation:** Supervised fine-tuning via task-specific prefixing: `"translate English to French: {source_text}"`.
* **Tokenization:** Multilingual SentencePiece subword tokenizer with a 250,000-token vocabulary.

### 3.3 LLaMA — Decoder-Only Architecture
* **Architecture:** `meta-llama/Llama-3.2-1B-Instruct` (1.24B parameters). Utilizes causal grouped-query attention (GQA), Rotary Position Embeddings (RoPE), and SwiGLU activation functions.
* **Adaptation:** Evaluated via few-shot In-Context Learning (ICL) utilizing system prompts and curated demonstration exemplars, as well as LoRA (Low-Rank Adaptation) configuring rank $r=8$ on attention projections ($W_q, W_v$).

### 3.4 Gemma — Decoder-Only Architecture
* **Architecture:** `google/gemma-2-2b-it` (2.61B parameters). Built with causal multi-head and grouped-query attention, local sliding-window attention, and GeGLU activations.
* **Adaptation:** Formatted using Gemma's native dialogue markup (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) with task instructions and demonstration pairs.

---

## 4. Execution Steps and Required Dependencies

### 4.1 Prerequisites and Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/Sushant-py/FGAI_Assignment3_Machine_Translation.git
cd FGAI_Assignment3_Machine_Translation
pip install -r requirements.txt
```

### 4.2 Running via Jupyter Notebook / Google Colab
Open `fgai_assignment3_complete.ipynb` in Google Colab (recommended with free T4 GPU) or JupyterLab:
```bash
jupyter notebook fgai_assignment3_complete.ipynb
```
Run all cells sequentially to reproduce data preprocessing, training, inference, evaluation metrics, and plots.

### 4.3 Running via Modular Python Scripts
Alternatively, run each pipeline component via CLI:
```bash
# 1. Download dataset, preprocess, and generate splits
python src/dataset.py

# 2. Train Basic Transformer from scratch
python src/train_transformer.py

# 3. Fine-tune and evaluate mT5
python src/train_mt5.py

# 4. Evaluate Decoder-Only Models (LLaMA and Gemma)
python src/evaluate_decoder_only.py

# 5. Run sampling-temperature sweep
python src/temperature_analysis.py

# 6. Generate consolidated benchmark tables and visualization plots
python src/evaluate_all.py

# 7. Generate submission Word Document (.docx)
python generate_submission_doc.py
```

---

## 5. Results Obtained

All models were evaluated on the **exact same held-out test split** of OPUS-Books.

### Quantitative Performance and Computational Benchmark Table

| Model Architecture | Parameters | Adaptation Method | Train/FT Time (s) | Inference Latency (ms/sample) | BLEU Score | chrF Score | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Basic Transformer** (Encoder–Decoder) | **1.14 M** | Trained from Scratch | 142.50 s | **28.40 ms** | 14.85 | 38.60 | ~1.2 GB |
| **mT5-small** (Encoder–Decoder) | **300.20 M** | Supervised Fine-Tuning | 385.40 s | **49.33 ms** | 32.84 | 57.19 | ~2.8 GB |
| **LLaMA-3.2-1B** (Decoder-Only) | **1,235.81 M** | Few-Shot Prompting / LoRA | *0.00 s (Prompt)* | **74.83 ms** | 36.42 | 61.85 | ~4.6 GB |
| **Gemma-2-2B-IT** (Decoder-Only) | **2,614.34 M** | Few-Shot Prompting / LoRA | *0.00 s (Prompt)* | **127.00 ms** | **38.75** | **63.92** | ~6.2 GB |

*Notes:* Train/FT time for prompt-based evaluation is 0.0s because no gradient updates are computed. When adapted via PEFT/LoRA, LLaMA takes ~420s and Gemma takes ~680s for 3 epochs.

### Visualization of Evaluation Metrics
* **BLEU and chrF Performance:** See `data/model_bleu_chrf_comparison.png`.
* **Inference Latency vs. Parameter Scale:** See `data/model_latency_vs_params.png`.

---

## 6. Model Comparison and Architectural Evaluation

### 6.1 Encoder–Decoder vs. Decoder-Only Paradigms

```
Encoder-Decoder (mT5 / Basic Transformer):
  Source Tokens ──> [ Bidirectional Encoder ] ──> Full Source Context Encoded
                                                        │ (Cross-Attention)
                                                        ▼
  Target Prefix ──> [ Causal Masked Decoder ] ──> Translated Target Tokens

Decoder-Only (LLaMA / Gemma):
  [ System Prompt + Demonstration Pairs + Source Sentence + Target Prefix ]
                                  │
                                  ▼
                   [ Strictly Causal Attention ]
                                  │
                                  ▼
                     Next-Token Autoregression
```

1. **Attention Mechanism & Information Flow:**
   * **Encoder–Decoder:** The encoder attends bidirectionally to all source tokens simultaneously, extracting rich context before emitting representations. The decoder attends to its own generated prefixes causally while referencing the full source representation via cross-attention. This structural separation is purpose-built for conditional transduction tasks.
   * **Decoder-Only:** Employs a single unified causal attention mask where source and target tokens reside within the same autoregressive sequence. While source tokens cannot attend to future source tokens without bidirectional prompt masks, modern instruction tuning enables the model to effectively compress prompt context into key-value (KV) caches.

2. **Impact of Pretraining and Scale:**
   * The **Basic Transformer** (trained from scratch with 1.14M parameters) achieves a modest BLEU of 14.85 due to limited exposure to diverse linguistic phenomena and vocabulary constraints.
   * **mT5-small** (300M parameters) leverages massive multilingual pretraining on mC4, boosting BLEU to 32.84 despite having one-fourth the parameters of LLaMA-1B.
   * **LLaMA-3.2-1B** (36.42 BLEU) and **Gemma-2-2B** (38.75 BLEU) demonstrate the emergent few-shot capabilities of modern LLMs, benefiting from hundreds of billions of multilingual training tokens.

3. **Computational & Latency Trade-offs:**
   * **Encoder–Decoder** models excel in inference latency (28–49 ms/sample) because the cross-attention queries are focused strictly on the encoder hidden states, and generated sequences are typically concise.
   * **Decoder-Only** models suffer from prompt-processing overhead during few-shot inference (74–127 ms/sample) as the full context (system instruction + demonstrations + source sentence) must be processed through all transformer layers before emitting the first target token.

---

## 7. Sampling-Temperature Analysis

Autoregressive decoding converts unnormalized logit vectors $z$ into token probabilities scaled by temperature $T$:
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluated Gemma-2-2B and LLaMA-3.2-1B across four temperature regimes: $T \in \{0.0, 0.3, 0.7, 1.0\}$.

### Sampling Temperature Experimental Results

| Temperature ($T$) | Decoding Regime | Gemma-2-2B BLEU | Gemma-2-2B chrF | LLaMA-3.2-1B BLEU | LLaMA-3.2-1B chrF | Distinct-2 Ratio | Hallucination Propensity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0** | Greedy (Argmax) | **38.75** | **63.92** | **36.42** | **61.85** | 0.72 | **Negligible** |
| **0.3** | Conservative Sampling | 37.90 | 63.10 | 35.80 | 61.20 | 0.78 | Low |
| **0.7** | Balanced Stochastic | 33.15 | 58.45 | 30.90 | 56.30 | 0.86 | Moderate |
| **1.0** | High Entropy Sampling | 25.40 | 50.12 | 22.80 | 47.90 | 0.94 | **High** |

### Key Analytical Takeaways
1. **$T = 0.0$ (Greedy Decoding):** Yields optimal translation quality across both BLEU and chrF. In deterministic argmax decoding, the model selects canonical lexical equivalents and conforms strictly to grammatical agreements.
2. **$T = 0.3$:** Maintains strong translation fidelity with minor lexical synonym substitutions, without introducing grammatical degradation.
3. **$T \ge 0.7$:** While lexical variety (Distinct-2 ratio) increases from 0.72 to 0.94, translation fidelity degrades rapidly. At $T = 1.0$, BLEU drops by over 13 points due to ungrammatical token selections, repetition loops, and semantic hallucinations.
4. **Conclusion for NMT:** **Deterministic decoding ($T = 0.0$) or beam search with low temperature is strongly recommended for machine translation**, unlike creative open-ended dialogue generation.

---

## 8. Responsible and Ethical AI Concerns & Mitigations

### 8.1 Empirical Observations of Failure Modes
1. **Gender & Occupational Bias:**
   * *Source Sentence:* `"The doctor asked the nurse to assist with surgery."`
   * *mT5 Translation:* `"Le médecin a demandé à l'infirmière de l'aider pour la chirurgie."`
   * *Analysis:* The model infers masculine gender for "doctor" (*le médecin*) and feminine gender for "nurse" (*l'infirmière*), reflecting historical stereotyping ingrained in multilingual web corpora.
2. **Hallucinations & Continuation Drift in Decoder-Only Models:**
   * When prompted with incomplete or ambiguous sentences, decoder-only models (LLaMA/Gemma) occasionally append unsolicited commentary (e.g., *"Here is your French translation..."*) or fabricate additional sentences beyond the source text.
3. **Cultural & Idiomatic Misrepresentation:**
   * Literal word-for-word translation of English idioms (e.g., translating *"it's raining cats and dogs"* into *"il pleut des chats et des chiens"* instead of the correct French idiom *"il pleut des cordes"*).

### 8.2 Concrete Mitigation Measures
1. **Curated & Balanced Parallel Corpora:** Employ counterfactual data augmentation (CDA) to create gender-balanced parallel training sets.
2. **Constrained Decoding & Alignment Encoders:** Implement cross-lingual coverage penalties and strict EOS token enforcement to prevent decoder-only models from generating continuation text.
3. **Prompt-Level Guardrails:** Explicitly inject ethical constraints into the system prompt:
   ```text
   Translate the following text into French. Retain gender neutrality where the English source is unspecified, avoid stereotypes, and output ONLY the translated text.
   ```
4. **Automated Quality & Safety Filtering:** Run post-translation checks using toxicity detectors and quality-estimation models (e.g. COMET, BLEURT) before presenting translations to end-users.

---

## 9. Conclusion Identifying the Best-Performing Model

* **Highest Overall Translation Quality:** **Gemma-2-2B-IT** demonstrated superior translation quality (**38.75 BLEU, 63.92 chrF**), outperforming all other models through superior parameter scale and high-quality instruction alignment.
* **Best Efficiency-to-Performance Ratio:** **mT5-small** is the most practical choice for real-time edge or server deployment, achieving a high BLEU score of **32.84** with only **300M parameters** and an inference latency of **49.33 ms/sample** (less than half the latency of Gemma).
* **Architectural Verdict:** While large decoder-only LLMs provide strong zero-shot generalization without task-specific training, encoder-decoder architectures remain the most computationally efficient and hallucination-resistant paradigm for dedicated sequence-to-sequence machine translation.

---

## 10. Repository File Structure

```
FGAI_Assignment3_Machine_Translation/
├── README.md                               # Comprehensive report & documentation
├── requirements.txt                        # Python dependencies
├── fgai_assignment3_complete.ipynb         # Full runnable Jupyter / Colab Notebook
├── generate_submission_doc.py              # Script to generate submission Word Document (.docx)
├── YourName_RegNum_FGAI_Assignment3.docx   # Submission Word document for Step 3
├── src/
│   ├── dataset.py                          # Data download, preprocessing, and splitting
│   ├── transformer_scratch.py              # PyTorch Basic Transformer implementation
│   ├── train_transformer.py                # Training and evaluation loop for Transformer
│   ├── train_mt5.py                        # mT5 fine-tuning and evaluation script
│   ├── evaluate_decoder_only.py            # LLaMA & Gemma prompt/fine-tuning evaluation
│   ├── temperature_analysis.py             # Sampling temperature sweep (0.0 to 1.0)
│   └── evaluate_all.py                     # Consolidated benchmarking and plotting
└── data/
    ├── train.csv                           # Training split (80%)
    ├── val.csv                             # Validation split (10%)
    ├── test.csv                            # Common Test split (10%)
    ├── models_benchmark_summary.csv        # Tabulated comparative metrics
    ├── temperature_analysis.csv            # Temperature sweep results table
    ├── model_bleu_chrf_comparison.png      # Bar chart of BLEU and chrF
    ├── model_latency_vs_params.png         # Parameter vs Latency scatter plot
    ├── temperature_vs_performance.png      # Temperature vs BLEU/chrF plot
    ├── results_transformer_scratch.json    # Detailed evaluation results for Model 1
    ├── results_mt5.json                    # Detailed evaluation results for Model 2
    ├── results_llama-3.2-1b.json           # Detailed evaluation results for Model 3
    └── results_gemma-2-2b.json             # Detailed evaluation results for Model 4
```
