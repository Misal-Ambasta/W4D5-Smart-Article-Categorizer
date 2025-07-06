# Q: 1 - Smart Article Categorizer

Build a system that automatically classifies articles into 6 categories (Tech, Finance, Healthcare, Sports, Politics, Entertainment) using different embedding approaches.

## Technical Requirements

### 1. Implement 4 Embedding Models

- **Word2Vec/GloVe**: Average word vectors for document representation
- **BERT**: Use [CLS] token embeddings
- **Sentence-BERT**: Direct sentence embeddings (all-MiniLM-L6-v2)
- **OpenAI**: text-embedding-ada-002 API

### 2. Classification Pipeline

- Train Logistic Regression classifier on each embedding type
- Compare accuracy, precision, recall, and F1-score
- Analyze which embedding works best for news classification

### 3. Web UI

- Text input for article classification
- Real-time predictions from all 4 models
- Confidence scores and model comparison
- Visualization of embedding clusters

## Deliverables

- **Code**: Submit a Github repo link with:
  - Working code
- **UI**: Working app with live classification
- **Analysis**: Performance comparison report with recommendations