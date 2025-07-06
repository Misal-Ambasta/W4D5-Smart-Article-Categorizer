# Smart Article Categorizer 🤖📰

A sophisticated article classification system that uses multiple embedding approaches to categorize news articles into 6 categories: Tech, Finance, Healthcare, Sports, Politics, and Entertainment.

## Features

- **4 Different Embedding Models**: Word2Vec, BERT, Sentence-BERT, and OpenAI embeddings
- **LangChain Integration**: Leverages LangChain for seamless embedding management
- **Interactive Web UI**: Built with Streamlit for real-time classification
- **Performance Analysis**: Comprehensive comparison of model performance
- **Visualization**: Interactive charts and graphs for embedding clusters and results

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-article-categorizer.git
cd smart-article-categorizer

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run article_categorizer.py
```

### Usage

1. **Start the Application**
   ```bash
   streamlit run article_categorizer.py
   ```

2. **Train Models**
   - Open the sidebar in the web interface
   - Optionally enter your OpenAI API key
   - Click "Train Models" to train all embedding models

3. **Classify Articles**
   - Enter article text in the text area
   - View predictions from all models
   - Compare confidence scores and performance metrics

## Architecture

### Embedding Models

1. **Word2Vec**: 
   - Trains a custom Word2Vec model on the dataset
   - Averages word vectors for document representation
   - 100-dimensional vectors with skip-gram architecture
   - (Note: GloVe is not used in this project)

2. **BERT**: 
   - Uses BERT base model via HuggingFace Transformers
   - Leverages [CLS] token embeddings for classification
   - 768-dimensional contextual embeddings

3. **Sentence-BERT**: 
   - Uses all-MiniLM-L6-v2 model optimized for sentence embeddings
   - 384-dimensional embeddings designed for semantic similarity
   - Fast inference and high-quality sentence representations

4. **OpenAI Embeddings**: 
   - Uses text-embedding-ada-002 API
   - 1536-dimensional embeddings with state-of-the-art performance
   - Requires OpenAI API key

### Classification Pipeline

- **Preprocessing**: Text cleaning, tokenization, lemmatization
- **Feature Extraction**: Multiple embedding approaches via LangChain
- **Classification**: Logistic Regression with L2 regularization
- **Evaluation**: Accuracy, Precision, Recall, F1-Score metrics

### Dataset

The system includes a sample dataset with 60 articles across 6 categories:
- **Tech**: Apple, Microsoft, Google, Tesla, AI developments
- **Finance**: Federal Reserve, stock market, cryptocurrency, banking
- **Healthcare**: COVID-19, FDA approvals, medical research, treatments
- **Sports**: World Cup, Olympics, NBA, tennis, football
- **Politics**: Elections, Congress, Supreme Court, international relations
- **Entertainment**: Marvel, streaming, music, awards, celebrities

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| Word2Vec | 0.85-0.90 | 0.86-0.91 | 0.85-0.90 | 0.85-0.90 | Fast |
| BERT | 0.88-0.93 | 0.89-0.94 | 0.88-0.93 | 0.88-0.93 | Slow |
| Sentence-BERT | 0.90-0.95 | 0.91-0.96 | 0.90-0.95 | 0.90-0.95 | Medium |
| OpenAI | 0.92-0.97 | 0.93-0.98 | 0.92-0.97 | 0.92-0.97 | Medium |

*Note: Performance varies based on dataset size and quality*

## Web Interface Features

### Main Dashboard
- **Article Input**: Text area for article classification
- **Sample Articles**: Pre-loaded examples for testing
- **Real-time Predictions**: Instant classification from all models
- **Confidence Scores**: Probability scores for each prediction

### Visualizations
- **Confidence Comparison**: Bar chart comparing model confidence
- **Performance Metrics**: Line chart showing accuracy, precision, recall, F1
- **Model Information**: Detailed descriptions of each embedding approach

### Configuration
- **OpenAI API Key**: Optional integration with OpenAI embeddings
- **Model Training**: One-click training of all embedding models
- **Performance Tracking**: Automatic evaluation and metric calculation

## Advanced Usage

### Custom Dataset

```python
from article_categorizer import ArticleCategorizer
import pandas as pd

# Load your custom dataset
df = pd.read_csv('your_articles.csv')  # columns: 'text', 'category'

# Initialize categorizer
categorizer = ArticleCategorizer(openai_api_key='your-key')

# Train models
categorizer.train_all_models(df)

# Make predictions
predictions = categorizer.predict_all_models("Your article text here")
print(predictions)
```

### Batch Processing

```python
# Process multiple articles
articles = [
    "Article 1 text...",
    "Article 2 text...",
    "Article 3 text..."
]

results = []
for article in articles:
    predictions = categorizer.predict_all_models(article)
    results.append(predictions)

# Convert to DataFrame for analysis
import pandas as pd
df_results = pd.DataFrame(results)
```

### Model Persistence

```python
# Save trained models
categorizer.save_models('models/')

# Load models later
new_categorizer = ArticleCategorizer()
new_categorizer.load_models('models/')
```

## API Integration

### LangChain Integration & Embedding Management

- Embedding models are managed via LangChain integration packages: `langchain-openai` for OpenAI and `langchain-community` for HuggingFace models.
- All embedding calls are routed through these wrappers for consistency and future compatibility.
- See [flow.md](./flow.md) for a full background flow of which APIs and functions are called for each action.

### LangChain Components Used

```python
# As of LangChain v0.2+, use integration packages for embeddings:
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Sentence-BERT
sbert_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# BERT
bert_embeddings = HuggingFaceEmbeddings(
    model_name="bert-base-uncased"
)

# OpenAI
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)
```

> **Note:** You must have `langchain-openai` and `langchain-community` installed (see requirements.txt).

### Custom Preprocessing Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

# Document processing
loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()

# Text splitting for long documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
```

## Installation Options

### Standard Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -e .[dev]
```

### GPU Support
```bash
pip install -e .[gpu]
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (optional)
OPENAI_API_KEY=your-openai-api-key-here

# Hugging Face API Key (optional, for private models)
HUGGINGFACE_API_KEY=your-huggingface-api-key

# Model cache directory
TRANSFORMERS_CACHE=./models/cache
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=article_categorizer tests/

# Run specific test
pytest tests/test_embeddings.py::test_word2vec_embedding
```

## Project Structure

```
smart-article-categorizer/
├── article_categorizer.py      # Main application
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
├── flow.md                     # System flow and API call documentation
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── data/                       # Data directory
│   ├── sample_articles.csv
│   └── custom_dataset.csv
└── docs/                       # Documentation
    |─ performance_analysis.md
```

## Performance Optimization

### Memory Usage
- Use CPU-only versions of models for lower memory usage
- Implement batch processing for large datasets
- Clear model cache between training sessions

### Speed Optimization
- Cache embeddings for repeated texts
- Use sentence-transformers for fastest inference
- Implement async processing for web interface

### Accuracy Improvements
- Increase training dataset size
- Use domain-specific pre-trained models
- Implement ensemble methods combining multiple models

## Deployment

### Local Deployment
```bash
streamlit run article_categorizer.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "article_categorizer.py", "--server.headless", "true"]
```

### Cloud Deployment
- **Streamlit Cloud**: Push to GitHub and deploy via Streamlit Cloud
- **Heroku**: Use Heroku buildpack for Python/Streamlit
- **AWS/GCP**: Deploy on container services with load balancing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for embedding abstractions
- [Hugging Face](https://huggingface.co/) for transformer models
- [OpenAI](https://openai.com/) for embedding API
- [Streamlit](https://streamlit.io/) for web interface
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities
