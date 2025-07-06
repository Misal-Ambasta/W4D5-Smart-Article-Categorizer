# Smart Article Categorizer – System Flow

This document describes what happens in the background for each major action in the Smart Article Categorizer. It details which API, class, and function is called at each step, both as a step-by-step flow and as a Mermaid diagram.

---

## 1. Training the Models

**User Action:** Start training (via script or Streamlit UI)

**Flow:**
```
User -> main() or Streamlit UI -> ArticleCategorizer.__init__() -> ArticleCategorizer.create_sample_dataset() -> ArticleCategorizer.train_all_models()
    -> ArticleCategorizer.train_word2vec_embeddings()
    -> ArticleCategorizer._initialize_embeddings()
    -> ArticleCategorizer.get_embeddings()
    -> ArticleCategorizer.train_classifier()
    -> ArticleCategorizer.evaluate_model()
    -> Save models (ArticleCategorizer.save_models())
```

**Mermaid:**
```mermaid
flowchart TD
    A[User] -->|Start training| B(main() or Streamlit UI)
    B --> C[ArticleCategorizer.__init__()]
    C --> D[create_sample_dataset()]
    D --> E[train_all_models()]
    E --> F[train_word2vec_embeddings()]
    E --> G[_initialize_embeddings()]
    E --> H[get_embeddings()]
    E --> I[train_classifier()]
    E --> J[evaluate_model()]
    E --> K[save_models()]
```

---

## 2. Making a Prediction

**User Action:** Enter article text for prediction

**Flow:**
```
User -> predict_all_models() -> For each model:
    -> preprocess_text()
    -> get_embeddings()
    -> model.predict()
    -> Decode label
```

**Mermaid:**
```mermaid
flowchart TD
    A[User] -->|Enter article| B[predict_all_models()]
    B --> C[preprocess_text()]
    B --> D[get_embeddings()]
    B --> E[model.predict()]
    E --> F[Decode label]
```

---

## 3. Saving Models

**User Action:** Save trained models

**Flow:**
```
User -> ArticleCategorizer.save_models() -> joblib.dump() for each model/encoder
```

**Mermaid:**
```mermaid
flowchart TD
    A[User] -->|Save| B[save_models()]
    B --> C[joblib.dump()]
```

---

## 4. Loading Models

**User Action:** Load previously saved models

**Flow:**
```
User -> ArticleCategorizer.load_models() -> joblib.load() for each model/encoder
```

**Mermaid:**
```mermaid
flowchart TD
    A[User] -->|Load| B[load_models()]
    B --> C[joblib.load()]
```

---

## 5. Web UI (Streamlit)

**User Action:** Use Streamlit interface

**Flow:**
```
User (browser) -> Streamlit app (create_streamlit_app) -> ArticleCategorizer methods (as above)
```

**Mermaid:**
```mermaid
flowchart TD
    A[User (browser)] -->|Interact| B[Streamlit app]
    B --> C[ArticleCategorizer methods]
```

---

## Notes
- All embedding APIs are called via their respective LangChain wrappers (OpenAI, HuggingFace, etc.)
- The flow is similar whether run from script or Streamlit UI.
- Each endpoint is a method in `ArticleCategorizer` or the Streamlit UI function.

---

Feel free to expand this document as new features or flows are added!
