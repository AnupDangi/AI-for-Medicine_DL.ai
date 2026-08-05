# Course 3 · Week 2 — Medical Question Answering

> Most clinical information lives in **unstructured text** (radiology reports, notes). This week you
> automate turning that text into **structured disease labels**, handle **negation**, and use **BERT**
> for extractive **question answering** over clinical notes.

---

## Learning objectives

1. Clean and pattern-match clinical text with **regular expressions**.
2. Convert reports into the **BioC** format and run **NegBio** for negation / uncertainty.
3. Extract disease labels with **text matching**, then improve them with **negation-aware** rules.
4. Evaluate automatic labelers with **precision / recall / F1**.
5. Prepare BERT inputs (tokenize, pad, mask) and recover answer spans for extractive QA.

---

## 0. Foundations from the labs

- **Lab 1 — `C3_W2_Lab_1_cleaning_text.ipynb`**
  Regular expressions (`re`): anchors (`^` / `$`), alternation (`|`), character classes, and
  substitutions — the building blocks for rule-based label extraction.

- **Lab 2 — `C3_W2_Lab_2_bioc_and_negbio.ipynb`**
  The **BioC** document format and **NegBio** pipeline: sentence splitting, dependency parsing, and
  detecting negative / uncertain findings in radiology reports (foundation for CheXpert-style
  labelers).

- **Lab 3 — `C3_W2_Lab_3_prep_input_for_text_classification.ipynb`**
  How transformer models expect text: tokenize → integer IDs → **padding** to a fixed length →
  **attention / input masks**. Practice for the BERT QA section of the assignment.

---

## 1. Automatic label extraction

Unstructured reports must become multi-label vectors (e.g. pneumonia, edema, cardiomegaly) before
you can train imaging models at scale.

### Text matching
Search each report for disease **keywords / phrases**. Simple and fast, but fails when the mention is
negated ("*no* pleural effusion").

### Negation-aware rules
Strip or ignore mentions preceded by negation cues (`no`, `without`, `denies`, …). Improves F1 but
still misses complex grammar.

### Dependency parsing (NegBio / CheXpert)
Parse sentence structure so negation and uncertainty attach to the correct finding. Used by the
**CheXpert** labeler (Irvin et al., 2019) that this week builds toward.

### Evaluation
Compare extracted labels to gold annotations with **F1** (harmonic mean of precision and recall) —
standard for imbalanced multi-label report labeling.

> Assignment exercises: `get_labels`, `get_labels_negative_aware`

---

## 2. Question answering with BERT

**BERT** (Bidirectional Encoder Representations from Transformers) is a pretrained language model.
For **extractive QA**:

1. Concatenate `[CLS] question [SEP] passage [SEP]` and tokenize.
2. BERT scores every token as a candidate **start** and **end** of the answer span.
3. Pick the best `(start, end)` pair and detokenize into the answer string.

Applied to medicine: ask *"What is the ejection fraction?"* and pull the span from a clinical note
(e.g. MTSamples echocardiogram examples).

> Assignment exercises: `prepare_bert_input`, `get_span_from_scores`, `construct_answer`

---

## Notebooks this week

| Notebook | Role |
|----------|------|
| `C3_W2_Lab_1_cleaning_text.ipynb` | Regex cleaning / matching |
| `C3_W2_Lab_2_bioc_and_negbio.ipynb` | BioC + NegBio negation |
| `C3_W2_Lab_3_prep_input_for_text_classification.ipynb` | Tokenizer, padding, masks |
| `C3W2_Ungraded_Assignment.ipynb` | Label extraction + BERT QA |

---

## Key terms

| Term | Meaning |
|------|---------|
| BioC | Standard XML/JSON format for sharing biomedical text annotations |
| NegBio | Negation & uncertainty detector for radiology reports |
| CheXpert labeler | Rule-based NLP that labels 14 chest X-ray observations |
| Negation detection | Distinguishing "no pneumonia" from "pneumonia" |
| Extractive QA | Answer is a contiguous span copied from the passage |
| BERT | Bidirectional Transformer LM used here for QA |
| Input mask | Binary mask marking real tokens vs padding |
| F1 | Harmonic mean of precision and recall |

---

## Citations (Week 2)

- [Labeling methods and dataset](https://arxiv.org/abs/1901.07031) — CheXpert (Irvin et al., 2019)
- [Huggingface transformers library](https://github.com/huggingface/transformers)
- [BERT paper](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [Question answering data set (used for example)](https://rajpurkar.github.io/SQuAD-explorer/) — SQuAD

### Additional references

- [AI for Medical Treatment — Course home](https://www.coursera.org/learn/ai-for-medical-treatment)
- [NegBio](https://github.com/ncbi-nlp/NegBio)
- [BioC](http://bioc.sourceforge.net/)
- [Google Research BERT](https://github.com/google-research/bert)
