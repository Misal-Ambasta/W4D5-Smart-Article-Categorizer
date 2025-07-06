"""
Setup script for Smart Article Categorizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart-article-categorizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Smart Article Categorizer using multiple embedding approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-article-categorizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "langchain>=0.0.330",
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "nltk>=3.8.0",
        "gensim>=4.3.0",
        "plotly>=5.17.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.7.0",
        "joblib>=1.3.0",
        "pathlib2>=2.3.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "accelerate>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "article-categorizer=article_categorizer.main:main",
        ],
    },
)