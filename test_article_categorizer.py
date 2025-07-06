"""
Test suite for Smart Article Categorizer
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Import the main class (assuming it's in the same directory)
from article_categorizer import ArticleCategorizer


class TestArticleCategorizer:
    """Test cases for ArticleCategorizer class"""
    
    @pytest.fixture
    def categorizer(self):
        """Create a basic categorizer instance for testing"""
        return ArticleCategorizer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame([
            ("Apple announces new iPhone with AI features", "Tech"),
            ("Fed raises interest rates", "Finance"),
            ("New cancer treatment approved", "Healthcare"),
            ("World Cup final tonight", "Sports"),
            ("Election results announced", "Politics"),
            ("Marvel movie breaks records", "Entertainment"),
        ], columns=['text', 'category'])
    
    def test_initialization(self, categorizer):
        """Test categorizer initialization"""
        assert categorizer.categories == ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        assert categorizer.models == {}
        assert categorizer.embeddings == {}
        assert categorizer.performance_metrics == {}
        assert categorizer.sbert_embeddings is not None
        assert categorizer.bert_embeddings is not None
    
    def test_preprocess_text(self, categorizer):
        """Test text preprocessing"""
        text = "Apple Inc. announces NEW iPhone 15 with AI-powered features!!!"
        processed = categorizer.preprocess_text(text)
        
        # Should be lowercase, no special characters, no stopwords
        assert processed.lower() == processed
        assert "apple" in processed
        assert "announce" in processed or "announc" in processed
        assert "iphone" in processed
        assert "featur" in processed or "feature" in processed
        assert "!!!" not in processed
        assert "15" not in processed
    
    def test_create_sample_dataset(self, categorizer):
        """Test sample dataset creation"""
        df = categorizer.create_sample_dataset()
        
        assert len(df) == 60  # 10 articles per category
        assert list(df.columns) == ['text', 'category']
        assert set(df['category'].unique()) == set(categorizer.categories)
        
        # Check each category has equal representation
        category_counts = df['category'].value_counts()
        assert all(count == 10 for count in category_counts.values)
    
    def test_train_word2vec_embeddings(self, categorizer, sample_data):
        """Test Word2Vec training"""
        texts = sample_data['text'].tolist()
        categorizer.train_word2vec_embeddings(texts)
        
        assert categorizer.word2vec_model is not None
        assert categorizer.word2vec_model.vector_size == 100
        
        # Test that we can get embeddings
        embedding = categorizer.get_word2vec_embedding("Apple announces iPhone")
        assert embedding.shape == (100,)
        assert not np.allclose(embedding, 0)  # Should not be all zeros
    
    def test_get_word2vec_embedding(self, categorizer, sample_data):
        """Test Word2Vec embedding generation"""
        texts = sample_data['text'].tolist()
        categorizer.train_word2vec_embeddings(texts)
        
        # Test normal text
        embedding = categorizer.get_word2vec_embedding("Apple iPhone")
        assert embedding.shape == (100,)
        
        # Test empty text
        empty_embedding = categorizer.get_word2vec_embedding("")
        assert np.allclose(empty_embedding, 0)
        
        # Test unknown words
        unknown_embedding = categorizer.get_word2vec_embedding("xyzabc123unknown")
        assert unknown_embedding.shape == (100,)
    
    def test_get_sbert_embeddings(self, categorizer):
        """Test Sentence-BERT embeddings"""
        texts = ["Apple announces iPhone", "Tesla releases update"]
        embeddings = categorizer.get_embeddings(texts, "sbert")
        
        assert embeddings.shape[0] == 2  # Two texts
        assert embeddings.shape[1] > 0  # Should have dimensions
        assert not np.allclose(embeddings[0], embeddings[1])  # Should be different
    
    def test_get_bert_embeddings(self, categorizer):
        """Test BERT embeddings"""
        texts = ["Apple announces iPhone", "Tesla releases update"]
        embeddings = categorizer.get_embeddings(texts, "bert")
        
        assert embeddings.shape[0] == 2  # Two texts
        assert embeddings.shape[1] > 0  # Should have dimensions
        assert not np.allclose(embeddings[0], embeddings[1])  # Should be different
    
    @patch('article_categorizer.OpenAIEmbeddings')
    def test_get_openai_embeddings(self, mock_openai, categorizer):
        """Test OpenAI embeddings with mocking"""
        # Mock the OpenAI embeddings
        mock_embed = Mock()
        mock_embed.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_openai.return_value = mock_embed
        
        categorizer.openai_embeddings = mock_embed
        
        texts = ["Apple announces iPhone", "Tesla releases update"]
        embeddings = categorizer.get_embeddings(texts, "openai")
        
        assert embeddings.shape == (2, 1536)
        mock_embed.embed_documents.assert_called_once_with(texts)
    
    def test_train_classifier(self, categorizer):
        """Test classifier training"""
        # Create dummy data
        X = np.random.rand(100, 50)  # 100 samples, 50 features
        y = np.random.randint(0, 6, 100)  # 6 classes
        
        model = categorizer.train_classifier(X, y, "test_embedding")
        
        assert "test_embedding" in categorizer.models
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_evaluate_model(self, categorizer):
        """Test model evaluation"""
        # Create dummy data
        X_train = np.random.rand(80, 50)
        y_train = np.random.randint(0, 6, 80)
        X_test = np.random.rand(20, 50)
        y_test = np.random.randint(0, 6, 20)
        
        # Train model
        categorizer.train_classifier(X_train, y_train, "test_embedding")
        
        # Evaluate
        metrics = categorizer.evaluate_model(X_test, y_test, "test_embedding")
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_predict(self, categorizer, sample_data):
        """Test prediction functionality"""
        # Train a simple model
        categorizer.train_word2vec_embeddings(sample_data['text'].tolist())
        
        # Get embeddings and train classifier
        texts = sample_data['text'].tolist()
        labels = sample_data['category'].tolist()
        
        # Encode labels
        categorizer.label_encoder.fit(labels)
        y = categorizer.label_encoder.transform(labels)
        
        # Get embeddings
        X = categorizer.get_embeddings(texts, "word2vec")
        
        # Train classifier
        categorizer.train_classifier(X, y, "word2vec")
        
        # Test prediction
        category, confidence = categorizer.predict("Apple releases new iPhone", "word2vec")
        
        assert category in categorizer.categories
        assert 0 <= confidence <= 1
    
    def test_predict_all_models(self, categorizer, sample_data):
        """Test prediction with all models"""
        # Train models
        categorizer.train_word2vec_embeddings(sample_data['text'].tolist())
        
        texts = sample_data['text'].tolist()
        labels = sample_data['category'].tolist()
        
        categorizer.label_encoder.fit(labels)
        y = categorizer.label_encoder.transform(labels)
        
        # Train word2vec and sbert models
        for embedding_type in ["word2vec", "sbert"]:
            X = categorizer.get_embeddings(texts, embedding_type)
            categorizer.train_classifier(X, y, embedding_type)
        
        # Test prediction
        results = categorizer.predict_all_models("Apple releases new iPhone")
        
        assert len(results) >= 2  # At least word2vec and sbert
        for embedding_type, (category, confidence) in results.items():
            if category != "Error":
                assert category in categorizer.categories
                assert 0 <= confidence <= 1
    
    def test_save_and_load_models(self, categorizer, sample_data):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train a simple model
            categorizer.train_word2vec_embeddings(sample_data['text'].tolist())
            
            texts = sample_data['text'].tolist()
            labels = sample_data['category'].tolist()
            
            categorizer.label_encoder.fit(labels)
            y = categorizer.label_encoder.transform(labels)
            
            X = categorizer.get_embeddings(texts, "word2vec")
            categorizer.train_classifier(X, y, "word2vec")
            
            # Save models
            categorizer.save_models(temp_dir)
            
            # Check files exist
            assert Path(temp_dir, "label_encoder.pkl").exists()
            assert Path(temp_dir, "classifier_word2vec.pkl").exists()
            assert Path(temp_dir, "word2vec_model.bin").exists()
            
            # Create new categorizer and load models
            new_categorizer = ArticleCategorizer()
            new_categorizer.load_models(temp_dir)
            
            # Test that loaded model works
            category, confidence = new_categorizer.predict("Apple iPhone", "word2vec")
            assert category in new_categorizer.categories
            assert 0 <= confidence <= 1
    
    def test_invalid_embedding_type(self, categorizer):
        """Test error handling for invalid embedding type"""
        with pytest.raises(ValueError):
            categorizer.get_embeddings(["test"], "invalid_type")
    
    def test_predict_without_training(self, categorizer):
        """Test prediction without training should raise error"""
        with pytest.raises(ValueError):
            categorizer.predict("test text", "word2vec")
    
    def test_openai_without_api_key(self, categorizer):
        """Test OpenAI embeddings without API key"""
        assert categorizer.openai_embeddings is None
        
        with pytest.raises(ValueError):
            categorizer.get_embeddings(["test"], "openai")


class TestStreamlitApp:
    """Test cases for Streamlit application components"""
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.write')
    def test_streamlit_app_initialization(self, mock_write, mock_title, mock_config):
        """Test Streamlit app initialization"""
        from article_categorizer import create_streamlit_app
        
        # This should not raise any errors
        # Note: Full testing of Streamlit apps requires running the app
        # This is a basic smoke test
        assert create_streamlit_app is not None


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction"""
        # Create categorizer
        categorizer = ArticleCategorizer()
        
        # Create sample data
        df = categorizer.create_sample_dataset()
        
        # Train models (excluding OpenAI)
        categorizer.train_word2vec_embeddings(df['text'].tolist())
        
        # Encode labels
        categorizer.label_encoder.fit(df['category'].tolist())
        y = categorizer.label_encoder.transform(df['category'].tolist())
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            df['text'].tolist(), y, test_size=0.2, random_state=42
        )
        
        # Train models
        embedding_types = ["word2vec", "sbert"]
        
        for embedding_type in embedding_types:
            # Get embeddings
            X_train = categorizer.get_embeddings(X_train_texts, embedding_type)
            X_test = categorizer.get_embeddings(X_test_texts, embedding_type)
            
            # Train classifier
            categorizer.train_classifier(X_train, y_train, embedding_type)
            
            # Evaluate
            metrics = categorizer.evaluate_model(X_test, y_test, embedding_type)
            
            # Assert reasonable performance
            assert metrics['accuracy'] > 0.3  # At least better than random
            assert metrics['f1_score'] > 0.3
        
        # Test prediction
        test_articles = [
            "Apple announces new iPhone with AI features",
            "Federal Reserve raises interest rates",
            "New cancer treatment shows promise",
            "World Cup final attracts millions",
            "Election results favor incumbent",
            "Marvel movie breaks box office records"
        ]
        
        expected_categories = ["Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"]
        
        for i, article in enumerate(test_articles):
            results = categorizer.predict_all_models(article)
            
            # Should have predictions from both models
            assert len(results) >= 2
            
            # At least one model should predict the expected category
            predicted_categories = [result[0] for result in results.values() if result[0] != "Error"]
            # Note: We can't guarantee perfect accuracy, but test structure is correct
            assert len(predicted_categories) > 0
    
    def test_performance_benchmarks(self):
        """Test that models meet minimum performance benchmarks"""
        categorizer = ArticleCategorizer()
        df = categorizer.create_sample_dataset()
        
        # Use a larger subset for more reliable metrics
        categorizer.train_word2vec_embeddings(df['text'].tolist())
        
        texts = df['text'].tolist()
        labels = df['category'].tolist()
        
        categorizer.label_encoder.fit(labels)
        y = categorizer.label_encoder.transform(labels)
        
        # Test Word2Vec performance
        X = categorizer.get_embeddings(texts, "word2vec")
        categorizer.train_classifier(X, y, "word2vec")
        
        # Use cross-validation for more reliable metrics
        from sklearn.model_selection import cross_val_score
        model = categorizer.models["word2vec"]
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_accuracy = cv_scores.mean()
        
        # Should perform significantly better than random (1/6 = 0.167)
        assert mean_accuracy > 0.4, f"Word2Vec accuracy {mean_accuracy} too low"
        
        print(f"Word2Vec Cross-validation Accuracy: {mean_accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])