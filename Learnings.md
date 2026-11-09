# 🧠 Developer Field Guide & Project Learnings

Welcome to the engineering knowledge base for the **Bank Transaction Categorizer**.

This document tracks every major architectural decision, algorithm choice, data structure, and technical trade-off made throughout the lifecycle of this codebase. It is written in a direct, human-to-human style so any engineer joining the project can understand not just *how* things work, but *why* they were engineered this way.

---

## 📑 Table of Contents
1. [Chapter 1: Project Scaffolding, Environment & Packaging](#chapter-1-project-scaffolding-environment--packaging)
2. [Chapter 2: Data Acquisition & The Hybrid Synthetic Engine](#chapter-2-data-acquisition--the-hybrid-synthetic-engine)
3. [Chapter 3: Text Preprocessing & Bank Narrative Sanitization](#chapter-3-text-preprocessing--bank-narrative-sanitization)
4. [Chapter 4: The Baseline Classifier (TF-IDF + Linear Models)](#chapter-4-the-baseline-classifier-tf-idf--linear-models)
5. [Chapter 5: The Hybrid Rule Engine & Decision Arbiter](#chapter-5-the-hybrid-rule-engine--decision-arbiter)
6. [Chapter 6: Transparent Explainability & Feature Attribution](#chapter-6-transparent-explainability--feature-attribution)
7. [Chapter 7: Serving Layer (FastAPI), Visual Dashboard (Streamlit) & PDF Export](#chapter-7-serving-layer-fastapi-visual-dashboard-streamlit--pdf-export)
8. [Chapter 8: Security Architecture, Secret Hygiene & API Key Authorization](#chapter-8-security-architecture-secret-hygiene--api-key-authorization)

---

## Chapter 1: Project Scaffolding, Environment & Packaging

### 1. Concept & Tech Stack
- **Tools**: Python `venv` (virtual environment isolation), standard hierarchical modular structure (`src/`, `api/`, `dashboard/`, `data/`, `models/`, `tests/`), locked `requirements.txt`.
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `plotly`, `reportlab`, `faker`.

### 2. Why Did We Perform This?
When building an ML application that encompasses data generation, offline batch training, low-latency REST endpoints, and an interactive UI, mixing concerns into loose root-level scripts quickly creates dependency chaos and deployment friction.
- Decoupling `src/` (core domain logic & ML models) from `api/` (network interfaces) and `dashboard/` (presentation) ensures that the machine learning models can be tested, imported, or deployed in headless worker queues without pulling in Streamlit or web UI overhead.

### 3. What Needs Did We Fulfill?
- **Reproducibility**: Clear package boundary where dependencies don't pollute system-level Python.
- **Maintainability**: Clear separation of concerns so a frontend/dashboard developer and an ML engineer can work on `dashboard/` and `src/` independently.
- **Portability**: Standard packaging allows future packaging as a pip-installable wheel or Docker container.

### 4. What Could Be the Alternatives?
- **Alternative A**: Conda / Mamba environment.
- **Alternative B**: Poetry or Pipenv for deterministic lockfile resolution.
- **Alternative C**: Single monolithic repository script (e.g., everything inside `app.py`).

### 5. Why Didn't We Use the Alternatives?
- *Why not Conda?* Standard Python `venv` + `pip` has zero external binary prerequisites, is native to standard macOS/Linux platforms, boots faster, and minimizes container image size for production.
- *Why not Poetry?* For educational clarity and rapid prototyping across varied operating systems, a transparent `requirements.txt` is universally understood without requiring developers to install specialized package managers.
- *Why not a Monolith?* Monolithic scripts make unit testing impossible, create circular import bugs, and prevent the ML model from being served as an independent microservice.

---

## Chapter 2: Data Acquisition & The Hybrid Synthetic Engine

### 1. Concept & Tech Stack
- **Concept**: High-Fidelity Domain-Specific Synthetic Data Generation.
- **Tech Stack**: Python, `Faker`, regex, structured merchant dictionaries with realistic Indian banking protocols (`UPI`, `NEFT`, `IMPS`, `POS`, `ACH`).

### 2. Why Did We Perform This?
Real-world bank statements are heavily protected by privacy and financial data laws (GDPR, DPDP Act). Furthermore:
- Most open-source transaction datasets on Kaggle are US-centric (e.g., `WALMART`, `TRADER JOES`, `TARGET`) with check and standard credit card prefixes.
- In India, over **14 billion monthly transactions** occur over **UPI**, presenting unique narrative structures like:
  - `UPI/CR/432901238910/SWIGGY/ICICI/pay`
  - `POS 4421XXXX9102 DMART BANNERGHATTA BLR`
  - `NEFT-CR-N12345678-OWNER RENT-HDFC0001`
  - `ACH D- NETFLIX ENTERTAINMENT SVCS`

If a model is trained solely on clean merchant names (`"Starbucks"`), it immediately collapses when presented with real, messy bank statements laden with authorization IDs, masked card numbers, and IFSC branch codes.

### 3. What Needs Did We Fulfill?
- **Zero Privacy Violation**: No personal financial records or real PII are exposed.
- **Realistic Noise Modeling**: Injects real-world artifacts (masked numbers `XXXX`, ref codes, date stamps, merchant location tags).
- **Class Balance Control**: Guarantees adequate sample representation across all 13 target financial categories (`Food & Dining`, `Rent`, `Utilities`, `EMI`, `Transfers`, etc.) instead of suffering severe real-world imbalance (where 80% of transactions are micro-food/tea payments).

### 4. What Could Be the Alternatives?
- **Alternative A**: Scraping personal bank statements or asking users to submit statements.
- **Alternative B**: Relying strictly on public Kaggle datasets (e.g., Kaggle Personal Finance Transaction Dataset).
- **Alternative C**: Calling LLMs (e.g. OpenAI / Gemini API) in a loop to generate transaction strings.

### 5. Why Didn't We Use the Alternatives?
- *Why not real statements?* Severe privacy risk and data scarcity. Collecting enough varied transactions across diverse banks (HDFC, SBI, ICICI, Axis) would take months.
- *Why not only Kaggle?* Kaggle datasets lack Indian banking protocols (UPI handles, NEFT references, Indian merchants like Zepto, Blinkit, PhonePe, Cred). A model trained only on Kaggle fails on Indian statements.
- *Why not LLM API generation?* Generating 10,000+ labeled rows via LLM API calls introduces network latency, rate limits, API costs, and non-deterministic formatting. A Python-driven template engine generates 10,000 high-fidelity rows in under 2 seconds completely offline.

---

## Chapter 3: Text Preprocessing & Bank Narrative Sanitization

### 1. Concept & Tech Stack
- **Concept**: Rule-based lexical normalization, PII sanitization, and scikit-learn custom transformer (`TransactionTextCleaner`).
- **Tech Stack**: Python `re` module, string manipulation, `sklearn.base.BaseEstimator`, `TransformerMixin`.

### 2. Why Did We Perform This?
Raw bank statement narratives are filled with noise that has zero semantic relationship to the expense category:
- Masked card digits (`4421XXXX1234`)
- Long reference tracking numbers (`432423409123`)
- Banking protocol jargon (`UPI`, `CR`, `DR`, `POS`, `NEFT`, `IMPS`, `ACH`, `MAND`)
- Bank names and IFSC codes (`HDFC0001234`, `okhdfcbank`, `ICICI`)

If fed raw to a bag-of-words or TF-IDF vectorizer, unique 12-digit transaction numbers inflate the vocabulary size, dilute the importance of the actual merchant keyword (`swiggy`, `dmart`, `netflix`), and lead to severe overfitting.

### 3. What Needs Did We Fulfill?
- **Feature Sparsity Reduction**: Converts noisy strings like `UPI/CR/432423409123/SWIGGY/ICICI/payment` directly to clean, salient tokens (`swiggy`).
- **Privacy Sanitization**: Strips card masks and account indicators before data is stored or logged.
- **Pipeline Composability**: Implementing `BaseEstimator` and `TransformerMixin` allows the cleaner to be embedded directly into scikit-learn `Pipeline` objects, ensuring test-set and production data receive identical transformations without data leakage.

### 4. What Could Be the Alternatives?
- **Alternative A**: Generic natural language preprocessing using SpaCy or NLTK (lowercasing, English stopwords, lemmatization).
- **Alternative B**: Regular expressions executed inline in ad-hoc pandas lambda functions.
- **Alternative C**: Character-level CNN or subword tokenization (BPE / WordPiece) without manual cleaning.

### 5. Why Didn't We Use the Alternatives?
- *Why not generic SpaCy/NLTK?* Generic English stopwords (`is`, `the`, `at`) don't recognize Indian banking terms (`UPI`, `NEFT`, `IMPS`, `CR`, `DR`, `okaxis`). Furthermore, running heavy linguistic parsers on 50,000 short strings introduces needless CPU latency.
- *Why not ad-hoc lambda functions?* Ad-hoc lambdas cannot be pickled inside a serialized model file. When deployed to a REST API, you'd have to rewrite the same regexes, inviting subtle training-serving skew.
- *Why not subword tokenizers without cleaning?* Subword tokenizers will split numbers like `432423409123` into chunks (`432`, `423`, `409`) which create useless vocabulary clutter.

---

## Chapter 4: The Baseline Classifier (TF-IDF + Linear Models)

### 1. Concept & Tech Stack
- **Algorithm**: Term Frequency-Inverse Document Frequency (TF-IDF) with sublinear term frequency scaling + Multiclass Logistic Regression (`solver='lbfgs'`, `class_weight='balanced'`).
- **Tech Stack**: `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.linear_model.LogisticRegression`, `joblib`.

### 2. Why Did We Perform This?
In machine learning engineering, jumping directly into heavy neural architectures without a benchmark is an anti-pattern.
- Transaction categorization on cleaned text is predominantly driven by salient merchant n-grams (e.g. `swiggy`, `dmart`, `apollo pharmacy`).
- A TF-IDF vectorizer with unigrams, bigrams, and trigrams (`ngram_range=(1, 3)`) paired with a regularized linear model is computationally lightweight, trains in under 3 seconds, achieves **99.31% Accuracy** on our held-out test split, and evaluates in less than **2 milliseconds per inference**.

### 3. What Needs Did We Fulfill?
- **Calibrated Probabilities**: Logistic regression outputs true posterior probabilities via softmax ($P(\text{category} \mid \mathbf{x})$), enabling confidence gating.
- **Ultra-Low Latency**: Inference takes $<3\text{ms}$ on standard CPU, allowing real-time batch processing of thousands of statement rows in the Streamlit UI.
- **Imbalance Resilience**: Using `class_weight='balanced'` ensures smaller classes like `Income` and `EMI & Loans` receive equal optimization weight compared to high-volume classes like `Food & Dining`.

### 4. What Could Be the Alternatives?
- **Alternative A**: Fine-tuned DistilBERT / RoBERTa transformer model.
- **Alternative B**: Gradient Boosted Trees (XGBoost / LightGBM) on TF-IDF features.
- **Alternative C**: Naive Bayes (MultinomialNB).

### 5. Why Didn't We Use the Alternatives?
- *Why not BERT as the primary baseline?* While transformers excel at long conversational nuances, bank transaction descriptions are short 2-to-6 word strings. Running BERT requires PyTorch, 500MB+ dependencies, high memory footprint, and runs 20x to 50x slower on CPU without yielding meaningful accuracy gains over clean n-grams.
- *Why not XGBoost on TF-IDF?* High-dimensional sparse TF-IDF matrices (10,000+ features) are mathematically where linear models (`LogisticRegression` / `LinearSVC`) excel in both training speed and memory efficiency. Tree-based splits on sparse data are computationally wasteful.
- *Why not Naive Bayes?* Naive Bayes produces poorly calibrated, extreme probability scores (near 0 or 1), making confidence thresholds unreliable for safety fallbacks.

---

## Chapter 5: The Hybrid Rule Engine & Decision Arbiter

### 1. Concept & Tech Stack
- **Concept**: Multi-Tier Decision Arbitration (Deterministic Keyword Regexes $\rightarrow$ Statistical ML $\rightarrow$ Safety Fallback).
- **Tech Stack**: Python precompiled regex patterns (`re.compile`), custom `HybridTransactionClassifier`.

### 2. Why Did We Perform This?
No single classification paradigm is optimal for all cases:
- Pure ML models can occasionally make baffling errors on obvious terms if there is subtle noise.
- Deterministic rules cannot generalize to unseen merchants or unusual phrasing.
By cascading **Tier 1 (Rule Engine)** into **Tier 2 (ML Pipeline)** and **Tier 3 (Confidence Fallback)**:
1. Known marquee merchants (e.g. Swiggy, Netflix, Uber, DMart, Salary) resolve in microseconds with 99%+ guaranteed precision.
2. Less frequent or previously unseen descriptions pass to the ML model.
3. If the model's confidence is weak ($<0.35$), the system does not hallucinate a random category; it flags the row as `Other / Uncategorized` and sets `needs_review = True`.

### 3. What Needs Did We Fulfill?
- **Zero Hallucination on High-Stakes Categories**: Salary credits and loan EMI debits are intercepted with 100% precision.
- **Human-in-the-Loop Safeguard**: Users are alerted to uncertain transactions rather than quietly misled.
- **Auditability**: Every prediction explicitly records its `engine_source` (`rule_engine`, `ml_pipeline`, or `low_confidence_fallback`).

### 4. What Could Be the Alternatives?
- **Alternative A**: 100% Rule-based system (huge dictionary of if-else statements).
- **Alternative B**: 100% Machine learning with no rules.
- **Alternative C**: LLM prompting with few-shot examples for every query.

### 5. Why Didn't We Use the Alternatives?
- *Why not 100% Rules?* It is impossible to maintain rules for every local merchant, kirana shop, or peer name in India. The rules quickly become an unmaintainable nightmare.
- *Why not 100% ML?* ML models can occasionally misclassify a rare phrasing of "Salary" or "EMI", which frustrates users when an obvious keyword is present.
- *Why not LLMs?* High cost ($0.001/call), high latency (500–1500ms), and rate limits make LLM calls impractical for processing a 200-row statement instantly.

---

## Chapter 6: Transparent Explainability & Feature Attribution

### 1. Concept & Tech Stack
- **Concept**: Exact Linear Feature Attribution ($Contribution_i = x_i \cdot W_{c, i}$).
- **Tech Stack**: Scikit-learn TF-IDF matrix weights & model coefficient tensors.

### 2. Why Did We Perform This?
Financial applications operate in a zero-trust domain. If a user sees `POS 4421XXXX9102 DMART BANNERGHATTA BLR` categorized as `Groceries`, they need to know *why*. Showing that the token `dmart` carried an attribution weight of `+3.42` immediately validates the system's reasoning.

### 3. What Needs Did We Fulfill?
- **User Trust & Debuggability**: Developers and end-users can visually inspect what words triggered a categorization.
- **Instantaneous Computation**: Because the model is linear, exact attribution is computed in micro-seconds via vector dot-products rather than relying on slow permutation sampling.

### 4. What Could Be the Alternatives?
- **Alternative A**: KernelSHAP / Sampling-based SHAP library.
- **Alternative B**: LIME (Local Interpretable Model-agnostic Explanations).
- **Alternative C**: No explainability (black-box predictions).

### 5. Why Didn't We Use the Alternatives?
- *Why not KernelSHAP or LIME?* For linear text models, SHAP and LIME merely approximate the exact mathematical dot product through costly Monte Carlo permutations (often taking 500ms–2000ms per string). Computing the exact dot product directly is mathematically exact, 10,000x faster, and requires zero external dependency bloat.

---

## Chapter 7: Serving Layer (FastAPI), Visual Dashboard (Streamlit) & PDF Export

### 1. Concept & Tech Stack
- **Serving Engine**: `FastAPI` + `Uvicorn` + `Pydantic v2`.
- **Presentation UI**: `Streamlit` + `Plotly Express`.
- **Reporting**: `ReportLab` (Direct PDF drawing & styling).

### 2. Why Did We Perform This?
A machine learning model isolated in a Jupyter notebook provides zero utility to real users:
- **FastAPI** provides an enterprise-ready, async REST API with auto-generated OpenAPI/Swagger docs for external fintech apps and microservices.
- **Streamlit** provides an intuitive web interface for non-technical users to drag-and-drop their statement CSV and instantly view interactive spending charts.
- **ReportLab** delivers an exportable, print-ready PDF expense report suitable for tax filing, accountants, or personal budgeting.

### 3. What Needs Did We Fulfill?
- **Dual-Audience Delivery**: Serves both programmatic developers (via `/api/v1/predict`) and everyday consumers (via the web dashboard).
- **Offline / Local Privacy**: The entire stack runs on `localhost` without transmitting financial records to third-party cloud servers.
- **Actionable Financial Intelligence**: Automatically aggregates inflow vs. outflow, savings rate, and category distribution.

### 4. What Could Be the Alternatives?
- **Alternative A**: Flask or Django for API + React for frontend.
- **Alternative B**: HTML-to-PDF rendering using Headless Chrome / `WeasyPrint` / `pdfkit`.

### 5. Why Didn't We Use the Alternatives?
- *Why not Flask + React?* Building and maintaining a full React build toolchain (Webpack/Vite, node_modules) adds massive developer overhead without adding ML value. Streamlit allows pure-Python rapid iteration with reactive UI state.
- *Why not WeasyPrint/Headless Chrome?* Tools like `WeasyPrint` or `pdfkit` require complex native system dependencies (`cairo`, `pango`, `wkhtmltopdf`, or a full Chromium browser). `ReportLab` is pure Python, installs cleanly via pip, and compiles professional PDFs in milliseconds with zero system dependencies.

---

## Chapter 8: Security Architecture, Secret Hygiene & API Key Authorization

### 1. Concept & Tech Stack
- **Authentication Scheme**: FastAPI `Security` dependency with `APIKeyHeader(name="X-API-Key", auto_error=False)`.
- **Secret Hygiene**: Zero hardcoded credentials in source code, `.env.example` template with `.env` explicitly blocked in `.gitignore`.
- **CORS Protection**: Configurable allowed origins (`ALLOWED_ORIGINS`) via environment variables rather than hardcoded wildcards.
- **Data Leakage Prevention**: Gitignore policy ensuring `.venv`, `.pytest_cache`, user CSV uploads, and generated PDF reports are never committed to version control.

### 2. Why Did We Perform This?
When deploying financial software or open-sourcing code to GitHub, accidental secret leaks or unauthenticated endpoints pose severe risks:
1. **Public API Abuse**: Unprotected ML inference endpoints (`/predict`) can be abused by automated bots or third parties, causing CPU resource exhaustion.
2. **Credential Leaks**: Committing `.env` files with secret keys, tokens, or personal bank statements can lead to immediate security compromises.
3. **Data Protection Compliance**: Under regulations like GDPR and India's DPDP Act, personal financial statements must never be committed to shared or public repositories.

### 3. What Needs Did We Fulfill?
- **Zero Secret Exposure**: All source files scanned for hardcoded credentials. Developers configure credentials through environment variables.
- **Dual-Mode Ergonomics**: When running locally without configuration (`API_KEY=""`), the API runs in permissive development mode for rapid prototyping. When deployed in production with `API_KEY="secret"`, all protected routes immediately reject unauthenticated requests with `401 Unauthorized`.
- **Safe Version Control**: A hardened `.gitignore` ensures that temporary virtual environments, cache artifacts, and user statement data never enter Git history.

### 4. What Could Be the Alternatives?
- **Alternative A**: OAuth2 with JWT (JSON Web Tokens) and an external identity provider (Auth0, Okta, Keycloak).
- **Alternative B**: HTTP Basic Authentication (`username:password`).
- **Alternative C**: IP Whitelisting / VPC Peering.

### 5. Why Didn't We Use the Alternatives?
- *Why not OAuth2/JWT?* OAuth2 introduces stateful authorization servers, token refresh lifecycles, and cryptographic key rotation. For a high-performance categorization microservice called by backend services, an `X-API-Key` header provides immediate, stateless, low-latency authorization with zero external identity dependencies.
- *Why not Basic Auth?* Sending base64-encoded username/password pairs on every API request is obsolete, difficult to rotate safely across automated clients, and forces services to maintain user credential hashes.
- *Why not IP Whitelisting?* Mobile users, developers on dynamic residential IPs, and distributed client apps cannot use static IP whitelisting.

---
*Maintained with pride by the Bank Transaction Categorizer Engineering Team.*
