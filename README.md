The assignment has been executed on the supplied `toxic_comments_dataset.csv`. The submission file is [submission/Toxic_Comment_Classification_Assignment.docx](submission/Toxic_Comment_Classification_Assignment.docx). Submit this Word document containing screenshots of code and actual outputs; keep the Python files locally as required by the assignment.

The run completed all five tasks and 15 training epochs. Test accuracy was **40% (2 of 5 comments)**, test loss was **1.60736083984375**, and macro F1 was **0.2667**. The checkpoint from **epoch 7** had the lowest validation loss. These results demonstrate the workflow, but they do not establish reliable toxicity classification: the 30,000-row CSV contains only **17 unique comments**.

**Setup and exact run commands**

The existing `outputs/` directory contains the completed real-data run. To reproduce it from the project folder:

```bash
cd '/home/m_talhaleo2002/Desktop/Assignment-AI'
bash setup_and_run.sh --setup-only
source .venv/bin/activate
mkdir -p outputs
python -u main.py --data toxic_comments_dataset.csv --max-length 32 --epochs 15 > outputs/run.log 2>&1
```

The final command records the actual console output in `outputs/run.log`. It overwrites the current run artifacts. Use another project-local `--output-dir` and a matching log path to preserve separate experiments.

The automation can also perform setup and run all five tasks directly:

```bash
bash setup_and_run.sh --data toxic_comments_dataset.csv --max-length 32 --epochs 15
```

To specify the actual CSV columns explicitly:

```bash
python main.py --data toxic_comments_dataset.csv --text-column Comment_Text --label-column Toxicity_Label --max-length 32 --epochs 15 --batch-size 32
```

To supply new comments, repeat `--predict` for each comment. This command runs training and evaluation, then classifies the supplied comments:

```bash
python main.py --data toxic_comments_dataset.csv --max-length 32 --epochs 15 --output-dir outputs_custom --predict 'Thank you for explaining your point.' --predict 'Your reply is rude and insulting.'
```

Without `--predict`, five demonstration comments are classified automatically. They have no supplied ground-truth labels and do not contribute to test accuracy.

The intended environment is Linux x86_64 with Python 3.10, 3.11, or 3.12 and CPU execution. The setup script creates `.venv` locally, rejects environments that include system packages, installs through the environment's own Python, and runs `pip check`. It does not invoke `sudo` or a system package manager. This machine's Python lacks `ensurepip`; the script tries that method first and then downloads the official `get-pip.py` into the project if necessary. Dependency installation needs internet access.

The implementation is in [main.py](main.py), automation is in [setup_and_run.sh](setup_and_run.sh), and direct dependency pins are in [requirements.txt](requirements.txt):

```text
tensorflow-cpu==2.19.1
keras==3.10.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
```

These pins fix direct dependencies; they are not a complete transitive lock file. `main.py` contains no comments, hash characters, or explanatory docstrings. Technical explanations appear outside the code in this README.

**Dataset findings and the evaluation limit**

The supplied CSV has **30,000 rows and 15 columns**, no missing values, and no duplicate complete records. Its unique `Comment_ID` values distinguish records that otherwise reuse the same text. There are **29,983 duplicate comments beyond the first occurrence**, leaving **17 unique comments** after cleaning and deduplication. No cleaned text has conflicting labels.

| Class | Original rows | Unique cleaned comments | Training | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 18,579 | 4 | 2 | 1 | 1 |
| Identity_Attack | 1,468 | 3 | 1 | 1 | 1 |
| Severe_Toxic | 1,801 | 3 | 1 | 1 | 1 |
| Threat | 1,459 | 3 | 1 | 1 | 1 |
| Toxic | 6,693 | 4 | 2 | 1 | 1 |
| Total | 30,000 | 17 | 7 | 5 | 5 |

`Comment_Text` is the input and `Toxicity_Label` is the mutually exclusive multiclass target. The separate `Toxic` column indicates every non-clean category; using it as the target would incorrectly turn this assignment into binary classification. The script prioritizes the dedicated class-label column. Other indicator columns, numeric scores, IDs, and moderation metadata are not model inputs.

Repeated cleaned comments are removed before splitting. Otherwise, identical comments could appear in both training and test sets and make memorization look like generalization. Evaluation therefore describes the deduplicated population, whose class proportions differ from the original repeated rows.

Only five distinct comments remain for testing, one per class, so each correct prediction changes accuracy by 20 percentage points. The model correctly classified the Clean and Toxic test examples, with zero recall on the other three classes. Its new-comment maximum softmax probabilities were approximately 20.2%–20.5%, close to an even five-class distribution. Some demonstration predictions are plainly incorrect. More varied labeled comments are needed before drawing useful conclusions about performance on unseen language; no performance target was imposed or manufactured.

**Task 1: Loading and exploration**

Pandas loads the CSV and prints the first 10 records, last 8 records, shape, column names, data types, and descriptive statistics for numerical columns. This exposes the actual schema and data range before preprocessing. Missing or empty input produces a clear error.

**Task 2: Cleaning and target analysis**

The script displays missing values per column, counts duplicate complete records separately from duplicate comments, and prints the original target distribution. It removes repeated complete records and rows missing the input or target. Missing values in unrelated metadata do not require discarding usable comments.

Cleaning decodes HTML entities, normalizes Unicode using NFKC, converts text to lowercase, removes tags and URLs, retains letters and whitespace, and collapses whitespace. The letter filter removes numbers, punctuation, emojis, and special symbols. Empty cleaned text and blank targets are excluded. Identical cleaned text with conflicting labels is reported and excluded; repeated cleaned comments with the same label retain one example. Cleaned class counts and percentages are saved.

These transformations follow the exercise and reduce superficial vocabulary variation. They can also discard useful toxicity signals, such as punctuation or emojis, so this cleaning policy should not be treated as optimal for every text-classification problem.

**Task 3: Tokenization, split, and padding**

The usual split reserves 20% of unique comments for testing, then 20% of the remainder for validation, using stratification. Those proportions cannot place five classes into each partition for this dataset. A deterministic fallback splits each class separately, reserving at least one comment for validation and one for testing. Every class needs at least three distinct comments; otherwise the script stops with an explanation. For this CSV the fallback produces **7 training, 5 validation, and 5 test comments**, with no identical cleaned text shared between partitions.

The Keras Tokenizer and label encoder are fitted on training data only. Validation, test, and new comments reuse the training vocabulary; unseen words become the out-of-vocabulary token. The saved tokenizer was independently reconstructed from the seven training texts, and its vocabulary and word counts matched exactly. Its vocabulary has 46 indexed entries including the OOV token, plus the zero-padding index, so the embedding input dimension is 47.

The recorded run uses `--max-length 32`, with padding and truncation at the end. This is a conservative fixed bound for the short comments and was not tuned against test accuracy. The script's generic default remains 200, so the explicit argument is necessary to reproduce this run. The maximum vocabulary setting is 20,000. Padding creates uniform input shapes, while `mask_zero=True` lets the recurrent layer ignore zero-padding positions.

**Task 4: Architecture and training**

The model uses a 64-dimensional Embedding, a 64-unit LSTM, Dropout at rate 0.5, and a Dense output with five softmax units. The embedding learns word representations; the LSTM processes word order; dropout reduces dependence on individual activations; and softmax produces a distribution across mutually exclusive classes. Dropout can reduce overfitting risk but cannot compensate for seven distinct training comments.

Adam uses learning rate 0.001, with sparse categorical cross-entropy for integer class labels. Training uses batch size 32 and completes all 15 epochs; with seven training rows this gives one training batch per epoch. The script accepts 10–20 epochs and uses an explicit held-out validation partition. There is no early stopping.

Balanced class weights use training labels only. Training loss is class weighted, while validation/test loss and reported accuracy are unweighted; their absolute loss differences alone do not measure overfitting. The checkpoint with the lowest validation loss is selected, then loaded for test evaluation. In the recorded run this is epoch 7. The test results were not used to choose a different split, model, or epoch.

The seed is 42, TensorFlow deterministic operations are enabled, and CPU threading is limited. Reproduction can still depend on consistent software and hardware. These settings provide an understandable baseline, not evidence that the model or hyperparameters are optimal.

**Task 5: Evaluation, plots, and new comments**

The selected checkpoint is evaluated once on the held-out test partition. The recorded test loss is **1.60736083984375**, accuracy **40%**, macro F1 **0.2667**, and weighted F1 **0.2667**. The confusion matrix uses actual classes as rows and predicted classes as columns. It is printed as a table and saved as a heatmap.

The classification report includes precision, recall, F1, and support for every class. Macro F1 treats classes equally, while weighted F1 accounts for support. Undefined precision or recall is reported as zero. These metrics show errors that an overall accuracy figure can hide.

Training/validation accuracy and loss curves are saved as an image. Matplotlib uses a noninteractive backend, so plotting works without an open desktop window; the saved graphs appear in the submission screenshots.

Five new comments are cleaned, tokenized, padded, and scored using the selected model and saved training tokenizer. The output includes predicted class, maximum model probability, fraction of unseen words, and all class probabilities. Softmax scores are model outputs, not calibrated confidence. Unlabeled demonstration predictions are separate from the held-out accuracy calculation.

**Word submission and regeneration**

The assignment requests only a Word document with screenshots. Submit:

```text
submission/Toxic_Comment_Classification_Assignment.docx
```

[build_submission.py](build_submission.py) builds the document from project code and the recorded real outputs. The document can be regenerated after the run using:

```bash
.venv/bin/python -m pip --isolated --no-cache-dir install -r requirements-report.txt
.venv/bin/python build_submission.py
```

The separate [requirements-report.txt](requirements-report.txt) pins the document-generation dependencies:

```text
python-docx==1.2.0
playwright==1.55.0
```

The builder uses the existing `/usr/bin/google-chrome` to capture screenshot images. It does not install a system browser, retrain the model, or replace test predictions. The screenshot assets and Word document remain in the project. Preserve `outputs/run.log` when regenerating, since it contains the actual console output.

**Generated artifacts**

| Files | Purpose |
| --- | --- |
| `outputs/run.log` | Captured output of the recorded real-data run |
| `missing_values.csv`, `duplicate_records.csv`, `conflicting_labels.csv` | Data quality reports in `outputs/` |
| `original_class_distribution.csv`, `class_distribution.csv`, `cleaned_dataset.csv` | Original/retained class counts and deduplicated input |
| `train_rows.csv`, `validation_rows.csv`, `test_rows.csv` | Original source-row identifiers and partition labels |
| `tokenizer.json`, `classes.json` | Training vocabulary and ordered target classes |
| `best_model.keras` | Saved validation-selected checkpoint |
| `training_history.csv`, `training_curves.png` | All 15 epochs and accuracy/loss curves |
| `confusion_matrix.csv`, `confusion_matrix.png` | Test matrix and heatmap |
| `classification_report.txt`, `classification_report.csv` | Per-class and aggregate evaluation |
| `test_predictions.csv`, `metrics.json` | Test probabilities, predicted labels, and summary metrics |
| `new_comment_predictions.csv` | Five separate demonstration predictions |
| `run_config.json` | Settings, split sizes, dataset/source hashes, and framework version |

Except for the fully qualified paths in the table, run artifacts are under `outputs/`. The test prediction order matches `test_rows.csv`. The stored dataset and source hashes tie the recorded results to the CSV and `main.py` used for the run.

**Verification performed**

Independent checks confirmed that all 17 retained comments occur in exactly one partition, every class occurs in every partition, source-row labels match the input CSV, and cleaned texts never cross partitions. The tokenizer exactly matches a tokenizer fitted only on the seven saved training rows.

The saved model reproduced every recorded test and new-comment probability without retraining. Probability sums, predicted argmax labels, confusion matrix counts, classification report, test accuracy, and test loss were consistent within floating-point precision. All 15 epochs are present, epoch 7 has minimum validation loss, and dataset/source hashes match. `main.py` contains zero hash characters and zero Python comment tokens. Earlier synthetic checks are separate from these actual-data results.

**Environment isolation**

Dependencies remain in `.venv`, runtime caches and bootstrap downloads remain under `.cache`, and outputs/submission files remain in this project. pip's persistent download cache is disabled in the documented installation commands. The existing system Python and browser remain prerequisites and are not modified. No global packages are installed.

Leave the activated environment with:

```bash
deactivate
```

Deleting the project directory removes its local environment and generated artifacts. Preserve the dataset and submission document first if you need them later.
