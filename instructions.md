### Project Setup & Deep Learning Assignment Execution Guidelines

### Core Objective

Develop and evaluate a Deep Learning model for Toxic Comment Classification using the dataset toxic_comments_dataset.csv. All tasks must be completed within an isolated virtual environment to prevent global system modifications, ensuring that deleting the project folder completely removes all dependencies. 

### Part 1: Operational Workflow & Environment Separation

### 1. Isolation Strategy (Virtual Environment)

* **Action:** Before running or generating any machine learning scripts, create a local Python virtual environment named .venv inside the project root folder. Activation must happen locally.
* **Why this approach?** Standardizing dependencies globally creates version conflicts between different projects. Using an isolated .venv ensures that Keras, TensorFlow, and Pandas versions are locked strictly to this project workspace.
* **Cleanup Guarantee:** Once the folder is deleted, your main system remains 100% clean without bloated deep learning packages.

### 2. Execution Automation Script

* **Action:** Generate a single automation script (e.g., setup_and_run.sh or run_project.py) that checks for the environment, installs packages, and triggers the code seamlessly.
* **Why this approach?** Manual execution increases human error (like accidentally installing packages outside the environment). A unified workflow ensures a smooth, single-click execution.

### Part 2: Code Generation Constraints & Humanization

### 1. Strict No-Comments Policy

* **Rule:** Do NOT include any comments (# comment) inside the Python source files (.py).
* **Why this approach?** Real human developers preparing clean assignments write self-documenting code with meaningful variable names rather than cluttered text. This ensures the files match a natural humanized submission.

### 2. Implementation Explanations

* **Rule:** Do not write explanations inside the code. Instead, generate a companion text block or separate markdown logs detailing *why* an action was taken.

### Part 3: Step-by-Step Assignment Requirements & Technical Justifications

### Task 1: Dataset Loading and Initial Exploration

1. Load toxic_comments_dataset.csv using Pandas.
2. Display the first 10 and last 8 records; print the dataset shape.
3. Display column names, data types, and statistical summaries of numerical fields.

* **Technical Justification:** Checking dimensions and types first prevents downstream runtime crashes (e.g., trying to tokenize non-string elements or passing improper shapes to Keras).

### Task 2: Text Cleaning, Preprocessing and Target Analysis

1. Identify and display duplicate records and missing values per column.
2. Clean text: convert to lowercase; remove HTML tags, punctuation, numbers, emojis, special characters, and redundant whitespaces.

* **Technical Justification:** Deep learning models cannot interpret raw text anomalies or emojis effectively without massive datasets. Cleaning standardizes the text inputs, meaning the vocabulary size shrinks, training runs faster, and the model focuses only on semantic words.

### Task 3: NLP Tokenization and Sequence Padding

1. Fit a Keras Tokenizer **strictly on training text** only.
2. Map text data to numerical integer sequences for both train and test splits.
3. Establish a standard max sequence length; apply sequence padding/truncation.

* **Technical Justification:** Tokenizing using the test set causes "data leakage," artificially boosting test performance while failing on real-world inputs. Neural networks require static tensor shapes, making uniform padding mandatory for batch training.

### Task 4: Deep Learning Model Architecture & Training

1. Construct a multi-class sequence model using TensorFlow/Keras.
2. Architecture must include: Embedding layer, Core Model Layer (Dense, LSTM, or CNN), Dropout layer (for regularization), and a final Dense layer with a Softmax activation.
3. Train the model using an appropriate batch size, incorporating a validation split over 10–20 epochs.

* **Technical Justification:** Softmax is mathematically required here to output valid probability distributions for multi-class targets. Dropout randomly deactivates neurons during training, preventing the model from memorizing text (overfitting) and forcing it to generalize.

### Task 5: Performance Evaluation and Visualization

1. Evaluate test dataset to calculate final loss and accuracy.
2. Generate and display a structured Confusion Matrix.
3. Print a Classification Report containing Precision, Recall, and F1-score for each class.

* **Technical Justification:** Accuracy alone is highly misleading in text classification due to class imbalances (e.g., fewer highly toxic comments than neutral ones). Precision and Recall isolate how well the model handles rare, toxic categories.

### Part 4: Instructions for the AI Agent (Codex)

When processing these requirements: 

1. First, output the commands required to set up and isolate the environment.
2. Second, present the clean, uncommented Python scripts fulfilling Tasks 1 to 5.
3. Finally, output the verification explanations outside the code block so the user knows exactly why the selected deep learning layers and parameters represent the optimal approach.