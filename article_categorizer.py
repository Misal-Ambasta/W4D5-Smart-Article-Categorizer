"""
Smart Article Categorizer using LangChain
Implements 4 different embedding approaches for news classification
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import logging
from datetime import datetime

# Core ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document

# Additional libraries for embeddings
import gensim
from gensim.models import Word2Vec, KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict

# Web UI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ArticleCategorizer:
    """
    Main class for article categorization using multiple embedding approaches
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.embeddings = {}
        self.performance_metrics = {}
        
        # Initialize text preprocessing
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize embedding models
        self._initialize_embeddings(openai_api_key)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_embeddings(self, openai_api_key: Optional[str] = None):
        """Initialize all embedding models"""
        
        # 1. Sentence-BERT using LangChain
        self.sbert_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # 2. BERT using LangChain
        self.bert_embeddings = HuggingFaceEmbeddings(
            model_name="bert-base-uncased",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # 3. OpenAI embeddings (if API key provided)
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
        else:
            self.openai_embeddings = None
            
        # 4. Word2Vec will be initialized after training
        self.word2vec_model = None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for demonstration"""
        sample_data = [
            # Tech
            ("Apple unveils new iPhone with advanced AI capabilities and improved camera system", "Tech"),
            ("Microsoft launches new cloud computing platform for enterprise customers", "Tech"),
            ("Google announces breakthrough in quantum computing research", "Tech"),
            ("Tesla releases software update for autonomous driving features", "Tech"),
            ("Amazon Web Services expands data center infrastructure globally", "Tech"),
            ("Meta introduces new virtual reality headset for consumers", "Tech"),
            ("OpenAI releases new language model with enhanced capabilities", "Tech"),
            ("Nvidia reports record quarterly revenue from AI chip sales", "Tech"),
            ("SpaceX successfully launches satellite constellation for internet connectivity", "Tech"),
            ("IBM develops new quantum processor for commercial applications", "Tech"),
            
            # Finance
            ("Federal Reserve raises interest rates to combat inflation", "Finance"),
            ("Stock market reaches new all-time high amid economic recovery", "Finance"),
            ("Major bank reports strong quarterly earnings and profit growth", "Finance"),
            ("Cryptocurrency market experiences significant volatility this week", "Finance"),
            ("Treasury announces new economic stimulus package for small businesses", "Finance"),
            ("Goldman Sachs upgrades outlook for emerging market investments", "Finance"),
            ("JPMorgan Chase increases dividend payout to shareholders", "Finance"),
            ("Bitcoin price surges following institutional adoption news", "Finance"),
            ("European Central Bank maintains accommodative monetary policy", "Finance"),
            ("Warren Buffett's Berkshire Hathaway makes major acquisition", "Finance"),
            
            # Healthcare
            ("New COVID-19 vaccine shows promising results in clinical trials", "Healthcare"),
            ("FDA approves breakthrough cancer treatment for patients", "Healthcare"),
            ("Researchers develop new gene therapy for rare genetic disorders", "Healthcare"),
            ("Major pharmaceutical company announces drug price reductions", "Healthcare"),
            ("Telemedicine adoption continues to grow post-pandemic", "Healthcare"),
            ("New study reveals benefits of Mediterranean diet for heart health", "Healthcare"),
            ("Artificial intelligence helps doctors diagnose diseases more accurately", "Healthcare"),
            ("Mental health awareness campaigns gain momentum nationwide", "Healthcare"),
            ("Organ transplant success rates improve with new surgical techniques", "Healthcare"),
            ("Personalized medicine becomes more accessible to patients", "Healthcare"),
            
            # Sports
            ("World Cup final attracts record television viewership globally", "Sports"),
            ("Olympic Games feature new sports for younger audiences", "Sports"),
            ("NBA championship series goes to Game 7 with thrilling finish", "Sports"),
            ("Tennis Grand Slam tournament sees upset victories", "Sports"),
            ("Football season begins with high expectations for teams", "Sports"),
            ("Baseball World Series features historic pitching performances", "Sports"),
            ("Swimming world records broken at international competition", "Sports"),
            ("Golf major championship decided by playoff holes", "Sports"),
            ("Marathon race attracts thousands of participants worldwide", "Sports"),
            ("Boxing match generates massive pay-per-view revenue", "Sports"),
            
            # Politics
            ("Presidential election campaign intensifies with debates", "Politics"),
            ("Congress passes bipartisan infrastructure spending bill", "Politics"),
            ("Supreme Court makes landmark decision on constitutional rights", "Politics"),
            ("International summit addresses global climate change policies", "Politics"),
            ("Senate confirms new cabinet member after heated hearings", "Politics"),
            ("Local elections show shifting voter preferences nationwide", "Politics"),
            ("Trade negotiations continue between major economic powers", "Politics"),
            ("United Nations Security Council votes on peacekeeping mission", "Politics"),
            ("Governor signs executive order on environmental protection", "Politics"),
            ("Foreign policy experts debate diplomatic strategies", "Politics"),
            
            # Entertainment
            ("Marvel announces new superhero movie franchise", "Entertainment"),
            ("Streaming platform launches exclusive original series", "Entertainment"),
            ("Music festival features top artists from around the world", "Entertainment"),
            ("Academy Awards ceremony celebrates diverse filmmaking", "Entertainment"),
            ("Broadway shows return with enthusiastic audiences", "Entertainment"),
            ("Celebrity couple announces engagement at red carpet event", "Entertainment"),
            ("Video game industry reports record-breaking sales figures", "Entertainment"),
            ("Film festival showcases independent cinema and documentaries", "Entertainment"),
            ("Television series finale draws millions of viewers", "Entertainment"),
            ("Concert tour announcement generates massive ticket demand", "Entertainment"),
        ]
        
        df = pd.DataFrame(sample_data, columns=['text', 'category'])
        return df
    
    def train_word2vec_embeddings(self, texts: List[str]) -> None:
        """Train Word2Vec model on the dataset"""
        # Preprocess and tokenize texts
        tokenized_texts = [self.preprocess_text(text).split() for text in texts]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=1  # Skip-gram
        )
        
        self.logger.info("Word2Vec model trained successfully")
    
    def get_word2vec_embedding(self, text: str) -> np.ndarray:
        """Get Word2Vec embedding for text by averaging word vectors"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet")
        
        preprocessed_text = self.preprocess_text(text)
        words = preprocessed_text.split()
        
        # Get vectors for words that exist in vocabulary
        word_vectors = []
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        if not word_vectors:
            # Return zero vector if no words found
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average the word vectors
        return np.mean(word_vectors, axis=0)
    
    def get_embeddings(self, texts: List[str], embedding_type: str) -> np.ndarray:
        """Get embeddings for texts using specified embedding type"""
        
        if embedding_type == "word2vec":
            embeddings = [self.get_word2vec_embedding(text) for text in texts]
            return np.array(embeddings)
        
        elif embedding_type == "sbert":
            embeddings = self.sbert_embeddings.embed_documents(texts)
            return np.array(embeddings)
        
        elif embedding_type == "bert":
            embeddings = self.bert_embeddings.embed_documents(texts)
            return np.array(embeddings)
        
        elif embedding_type == "openai":
            if self.openai_embeddings is None:
                raise ValueError("OpenAI API key not provided")
            embeddings = self.openai_embeddings.embed_documents(texts)
            return np.array(embeddings)
        
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, embedding_type: str) -> LogisticRegression:
        """Train logistic regression classifier"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        self.models[embedding_type] = model
        self.logger.info(f"Classifier trained for {embedding_type} embeddings")
        
        return model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      embedding_type: str) -> Dict[str, float]:
        """Evaluate model performance"""
        if embedding_type not in self.models:
            raise ValueError(f"Model not trained for {embedding_type}")
        
        model = self.models[embedding_type]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.performance_metrics[embedding_type] = metrics
        return metrics
    
    def train_all_models(self, df: pd.DataFrame) -> None:
        """Train all embedding models and classifiers"""
        texts = df['text'].tolist()
        labels = df['category'].tolist()
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Word2Vec on all texts
        self.train_word2vec_embeddings(texts)
        
        # Train models for each embedding type
        embedding_types = ["word2vec", "sbert", "bert"]
        if self.openai_embeddings is not None:
            embedding_types.append("openai")
        
        for embedding_type in embedding_types:
            self.logger.info(f"Training {embedding_type} model...")
            
            # Get embeddings
            X_train = self.get_embeddings(X_train_texts, embedding_type)
            X_test = self.get_embeddings(X_test_texts, embedding_type)
            
            # Train classifier
            self.train_classifier(X_train, y_train, embedding_type)
            
            # Evaluate
            metrics = self.evaluate_model(X_test, y_test, embedding_type)
            self.logger.info(f"{embedding_type} - Accuracy: {metrics['accuracy']:.4f}")
    
    def predict(self, text: str, embedding_type: str) -> Tuple[str, float]:
        """Predict category for a single text"""
        if embedding_type not in self.models:
            raise ValueError(f"Model not trained for {embedding_type}")
        
        # Get embedding
        embedding = self.get_embeddings([text], embedding_type)[0]
        
        # Predict
        model = self.models[embedding_type]
        prediction = model.predict([embedding])[0]
        confidence = model.predict_proba([embedding])[0].max()
        
        # Decode label
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        return category, confidence
    
    def predict_all_models(self, text: str) -> Dict[str, Tuple[str, float]]:
        """Predict using all available models"""
        results = {}
        
        for embedding_type in self.models.keys():
            try:
                category, confidence = self.predict(text, embedding_type)
                results[embedding_type] = (category, confidence)
            except Exception as e:
                self.logger.error(f"Error predicting with {embedding_type}: {str(e)}")
                results[embedding_type] = ("Error", 0.0)
        
        return results
    
    def save_models(self, directory: str) -> None:
        """Save trained models to disk"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save label encoder
        joblib.dump(self.label_encoder, f"{directory}/label_encoder.pkl")
        
        # Save classifiers
        for embedding_type, model in self.models.items():
            joblib.dump(model, f"{directory}/classifier_{embedding_type}.pkl")
        
        # Save Word2Vec model
        if self.word2vec_model:
            self.word2vec_model.save(f"{directory}/word2vec_model.bin")
        
        # Save performance metrics
        with open(f"{directory}/performance_metrics.pkl", 'wb') as f:
            pickle.dump(self.performance_metrics, f)
        
        self.logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str) -> None:
        """Load trained models from disk"""
        # Load label encoder
        self.label_encoder = joblib.load(f"{directory}/label_encoder.pkl")
        
        # Load classifiers
        for embedding_type in ["word2vec", "sbert", "bert", "openai"]:
            model_path = f"{directory}/classifier_{embedding_type}.pkl"
            if Path(model_path).exists():
                self.models[embedding_type] = joblib.load(model_path)
        
        # Load Word2Vec model
        word2vec_path = f"{directory}/word2vec_model.bin"
        if Path(word2vec_path).exists():
            self.word2vec_model = Word2Vec.load(word2vec_path)
        
        # Load performance metrics
        metrics_path = f"{directory}/performance_metrics.pkl"
        if Path(metrics_path).exists():
            with open(metrics_path, 'rb') as f:
                self.performance_metrics = pickle.load(f)
        
        self.logger.info(f"Models loaded from {directory}")

def create_streamlit_app():
    """Create Streamlit web application"""
    
    st.set_page_config(
        page_title="Smart Article Categorizer",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("🤖 Smart Article Categorizer")
    st.write("Classify news articles using multiple embedding approaches")
    
    # Initialize session state
    if 'categorizer' not in st.session_state:
        st.session_state.categorizer = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="Enter your OpenAI API key to use OpenAI embeddings"
        )
        
        # Train models button
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Initialize categorizer
                    categorizer = ArticleCategorizer(openai_api_key=openai_api_key)
                    
                    # Create sample dataset
                    df = categorizer.create_sample_dataset()
                    
                    # Train all models
                    categorizer.train_all_models(df)
                    
                    # Store in session state
                    st.session_state.categorizer = categorizer
                    
                    st.success("Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
    
    # Main content
    if st.session_state.categorizer is None:
        st.info("👈 Please train the models first using the sidebar.")
        
        # Show sample data
        st.subheader("Sample Dataset")
        sample_df = ArticleCategorizer().create_sample_dataset()
        st.dataframe(sample_df)
        
        # Show categories
        st.subheader("Categories")
        categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        cols = st.columns(len(categories))
        for i, category in enumerate(categories):
            with cols[i]:
                st.info(f"**{category}**")
    
    else:
        categorizer = st.session_state.categorizer
        
        # Text input for classification
        st.subheader("📝 Article Classification")
        
        # Sample articles for testing
        sample_articles = [
            "Apple announces new iPhone with revolutionary AI features and advanced camera system",
            "Federal Reserve raises interest rates to combat rising inflation concerns",
            "Researchers develop breakthrough gene therapy for treating rare diseases",
            "World Cup final breaks television viewership records worldwide",
            "Congress passes landmark infrastructure bill with bipartisan support",
            "Marvel Studios reveals upcoming superhero movie slate at comic convention"
        ]
        
        # Article input
        selected_sample = st.selectbox(
            "Choose a sample article or enter your own:",
            [""] + sample_articles
        )
        
        article_text = st.text_area(
            "Enter article text:",
            value=selected_sample,
            height=100,
            placeholder="Enter the article text you want to classify..."
        )
        
        if article_text:
            # Get predictions from all models
            predictions = categorizer.predict_all_models(article_text)
            
            # Display predictions
            st.subheader("🎯 Predictions")
            
            cols = st.columns(len(predictions))
            for i, (embedding_type, (category, confidence)) in enumerate(predictions.items()):
                with cols[i]:
                    st.metric(
                        label=f"{embedding_type.upper()}",
                        value=category,
                        delta=f"{confidence:.2%} confidence"
                    )
            
            # Visualization
            st.subheader("📊 Prediction Comparison")
            
            # Create comparison chart
            chart_data = []
            for embedding_type, (category, confidence) in predictions.items():
                chart_data.append({
                    'Model': embedding_type.upper(),
                    'Category': category,
                    'Confidence': confidence
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            # Confidence comparison
            fig = px.bar(
                chart_df,
                x='Model',
                y='Confidence',
                color='Category',
                title='Model Confidence Comparison',
                labels={'Confidence': 'Confidence Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        if categorizer.performance_metrics:
            st.subheader("📈 Model Performance")
            
            # Create performance comparison
            metrics_data = []
            for embedding_type, metrics in categorizer.performance_metrics.items():
                metrics_data.append({
                    'Model': embedding_type.upper(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display metrics table
            st.dataframe(
                metrics_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Performance visualization
            fig = go.Figure()
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics_to_plot:
                fig.add_trace(go.Scatter(
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        with st.expander("ℹ️ About the Models"):
            st.markdown("""
            **Embedding Models Used:**
            
            1. **Word2Vec**: Averages word vectors trained on the dataset
            2. **BERT**: Uses BERT base model embeddings via HuggingFace
            3. **Sentence-BERT**: Uses all-MiniLM-L6-v2 model optimized for sentence embeddings
            4. **OpenAI**: Uses text-embedding-ada-002 API (requires API key)
            
            **Classification**: Logistic Regression is used as the classifier for all embedding types.
            
            **Categories**: Tech, Finance, Healthcare, Sports, Politics, Entertainment
            """)

if __name__ == "__main__":
    # Run Streamlit app
    create_streamlit_app()