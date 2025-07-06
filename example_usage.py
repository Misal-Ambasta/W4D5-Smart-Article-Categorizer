"""
Example usage of the Smart Article Categorizer
"""

import os
from article_categorizer import ArticleCategorizer
import pandas as pd

def main():
    """Main example function"""
    
    print("🤖 Smart Article Categorizer Example")
    print("=" * 50)
    
    # Initialize categorizer
    # Add your OpenAI API key here if you want to test OpenAI embeddings
    openai_key = os.getenv('OPENAI_API_KEY')  # or your API key
    categorizer = ArticleCategorizer(openai_api_key=openai_key)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    df = categorizer.create_sample_dataset()
    print(f"Dataset created with {len(df)} articles across {len(df['category'].unique())} categories")
    
    # Show sample data
    print("\n📰 Sample articles:")
    for category in df['category'].unique():
        sample = df[df['category'] == category].iloc[0]
        print(f"  {category}: {sample['text'][:80]}...")
    
    # Train models
    print("\n🚀 Training models...")
    print("This may take a few minutes...")
    
    try:
        categorizer.train_all_models(df)
        print("✅ All models trained successfully!")
        
        # Display performance metrics
        print("\n📈 Performance Metrics:")
        print("-" * 60)
        print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for model_name, metrics in categorizer.performance_metrics.items():
            print(f"{model_name.upper():<12} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return
    
    # Test predictions
    print("\n🎯 Testing Predictions:")
    print("=" * 50)
    
    test_articles = [
        "Apple announces groundbreaking AI chip for next generation iPhone",
        "Federal Reserve Chairman signals potential interest rate cuts amid economic uncertainty",
        "Breakthrough gene therapy shows remarkable success in treating inherited blindness",
        "World Cup semifinal match goes to penalty shootout in thrilling finish",
        "Congressional hearing addresses national security implications of social media",
        "Disney's latest animated film breaks opening weekend box office records"
    ]
    
    expected_categories = ["Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"]
    
    for i, article in enumerate(test_articles):
        print(f"\n📄 Article {i+1}:")
        print(f"Text: {article}")
        print(f"Expected: {expected_categories[i]}")
        print("Predictions:")
        
        try:
            predictions = categorizer.predict_all_models(article)
            
            for model_name, (category, confidence) in predictions.items():
                status = "✅" if category == expected_categories[i] else "❌"
                print(f"  {status} {model_name.upper():<12}: {category:<12} (confidence: {confidence:.3f})")
                
        except Exception as e:
            print(f"  ❌ Error making predictions: {e}")
    
    # Save models
    print("\n💾 Saving models...")
    try:
        categorizer.save_models('trained_models')
        print("✅ Models saved to 'trained_models' directory")
    except Exception as e:
        print(f"❌ Error saving models: {e}")
    
    # Demonstrate loading models
    print("\n🔄 Testing model loading...")
    try:
        new_categorizer = ArticleCategorizer()
        new_categorizer.load_models('trained_models')
        
        # Test loaded model
        test_text = "Microsoft launches new cloud computing service"
        predictions = new_categorizer.predict_all_models(test_text)
        
        print(f"✅ Loaded models work! Prediction for '{test_text}':")
        for model_name, (category, confidence) in predictions.items():
            print(f"  {model_name.upper()}: {category} (confidence: {confidence:.3f})")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    
    print("\n🎉 Example completed!")
    print("\n💡 Tips:")
    print("  - Add your OpenAI API key to test OpenAI embeddings")
    print("  - Use larger datasets for better performance")
    print("  - Run 'streamlit run article_categorizer.py' for web interface")


def interactive_demo():
    """Interactive demo allowing user input"""
    print("\n🔮 Interactive Demo")
    print("=" * 30)

    model_dir = 'trained_models'
    if not os.path.isdir(model_dir):
        print(f"❌ No trained models found in '{model_dir}'. Please run the main example first to train and save models.")
        return

    # Load models
    try:
        categorizer = ArticleCategorizer()
        categorizer.load_models(model_dir)
        print(f"✅ Models loaded from '{model_dir}'!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    print("\nEnter article text to get predictions (type 'exit' to quit):")
    while True:
        user_input = input("\n📝 Article: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting interactive demo.")
            break
        if not user_input:
            print("⚠️ Please enter some text.")
            continue
        try:
            predictions = categorizer.predict_all_models(user_input)
            print("Predictions:")
            for model_name, (category, confidence) in predictions.items():
                print(f"  {model_name.upper():<12}: {category:<12} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"❌ Error making predictions: {e}")


if __name__ == "__main__":
    main()
    interactive_demo()