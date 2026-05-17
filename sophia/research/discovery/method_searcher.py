"""Method searcher: find methods from external sources."""
import json
import importlib.util
from typing import Dict, List, Optional


# Keyword-to-library mapping for common research methods
KEYWORD_LIBRARY_MAP = {
    "irt": {"library": "girth", "pip": "girth", "description": "Item Response Theory estimation and analysis"},
    "item response theory": {"library": "girth", "pip": "girth", "description": "Item Response Theory estimation and analysis"},
    "sem": {"library": "semopy", "pip": "semopy", "description": "Structural Equation Modeling"},
    "structural equation": {"library": "semopy", "pip": "semopy", "description": "Structural Equation Modeling"},
    "bayesian": {"library": "pymc", "pip": "pymc", "description": "Bayesian statistical modeling with MCMC"},
    "pymc": {"library": "pymc", "pip": "pymc", "description": "Bayesian statistical modeling with MCMC"},
    "survival": {"library": "lifelines", "pip": "lifelines", "description": "Survival analysis (Kaplan-Meier, Cox PH)"},
    "kaplan-meier": {"library": "lifelines", "pip": "lifelines", "description": "Kaplan-Meier survival analysis"},
    "cox": {"library": "lifelines", "pip": "lifelines", "description": "Cox proportional hazards regression"},
    "panel": {"library": "linearmodels", "pip": "linearmodels", "description": "Panel data and instrumental variable estimation"},
    "panel data": {"library": "linearmodels", "pip": "linearmodels", "description": "Panel data estimation (fixed/random effects)"},
    "causal": {"library": "dowhy", "pip": "dowhy", "description": "Causal inference framework"},
    "causalimpact": {"library": "causalimpact", "pip": "causalimpact", "description": "Causal impact analysis (Google's CausalImpact)"},
    "textblob": {"library": "textblob", "pip": "textblob", "description": "Sentiment and text processing"},
    "spacy": {"library": "spacy", "pip": "spacy", "description": "Industrial-strength NLP pipeline"},
    "stan": {"library": "cmdstanpy", "pip": "cmdstanpy", "description": "Stan interface for Bayesian modeling"},
    "hierarchical": {"library": "statsmodels", "pip": "statsmodels", "description": "Hierarchical/mixed effects models"},
    "mixed effects": {"library": "statsmodels", "pip": "statsmodels", "description": "Mixed effects regression models"},
    "time series": {"library": "statsmodels", "pip": "statsmodels", "description": "Time series analysis (ARIMA, VAR, etc.)"},
    "arima": {"library": "statsmodels", "pip": "statsmodels", "description": "ARIMA time series modeling"},
    "forecast": {"library": "prophet", "pip": "prophet", "description": "Facebook Prophet time series forecasting"},
    "prophet": {"library": "prophet", "pip": "prophet", "description": "Facebook Prophet time series forecasting"},
    "nlp": {"library": "nltk", "pip": "nltk", "description": "Natural Language Processing Toolkit"},
    "topic": {"library": "gensim", "pip": "gensim", "description": "Topic modeling with LDA, Word2Vec"},
    "lda": {"library": "gensim", "pip": "gensim", "description": "Latent Dirichlet Allocation topic modeling"},
    "clustering": {"library": "scikit-learn", "pip": "scikit-learn", "description": "K-means, DBSCAN, hierarchical clustering"},
    "deep learning": {"library": "torch", "pip": "torch", "description": "PyTorch deep learning framework"},
    "pytorch": {"library": "torch", "pip": "torch", "description": "PyTorch deep learning framework"},
    "tensorflow": {"library": "tensorflow", "pip": "tensorflow", "description": "TensorFlow deep learning framework"},
    "transformer": {"library": "transformers", "pip": "transformers", "description": "HuggingFace Transformers for NLP"},
    "bert": {"library": "transformers", "pip": "transformers", "description": "BERT and transformer models for NLP"},
    "embedding": {"library": "sentence-transformers", "pip": "sentence-transformers", "description": "Sentence embedding models"},
    "anomaly": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Anomaly detection (Isolation Forest, LOF)"},
    "graph": {"library": "networkx", "pip": "networkx", "description": "Graph and network analysis"},
    "geospatial": {"library": "geopandas", "pip": "geopandas", "description": "Geospatial data analysis"},
    "spatial": {"library": "geopandas", "pip": "geopandas", "description": "Spatial data analysis"},
    "image": {"library": "Pillow", "pip": "Pillow", "description": "Image processing library"},
    "optimization": {"library": "scipy", "pip": "scipy", "description": "Mathematical optimization"},
    "simulation": {"library": "simpy", "pip": "simpy", "description": "Discrete event simulation"},
    "psychometrics": {"library": "girth", "pip": "girth", "description": "Psychometric analysis (IRT, reliability)"},
    "multilevel": {"library": "statsmodels", "pip": "statsmodels", "description": "Multilevel/HLM regression"},
    "mediation": {"library": "statsmodels", "pip": "statsmodels", "description": "Mediation and path analysis"},
    "power": {"library": "statsmodels", "pip": "statsmodels", "description": "Statistical power analysis"},
    # Extended mappings (D5)
    "logistic regression": {"library": "statsmodels", "pip": "statsmodels", "description": "Logistic regression with odds ratios"},
    "logit": {"library": "statsmodels", "pip": "statsmodels", "description": "Logistic regression with odds ratios"},
    "probit": {"library": "statsmodels", "pip": "statsmodels", "description": "Probit regression"},
    "poisson": {"library": "statsmodels", "pip": "statsmodels", "description": "Poisson regression for count data"},
    "negative binomial": {"library": "statsmodels", "pip": "statsmodels", "description": "Negative binomial regression"},
    "ordinal": {"library": "statsmodels", "pip": "statsmodels", "description": "Ordinal regression models"},
    "multinomial": {"library": "statsmodels", "pip": "statsmodels", "description": "Multinomial logistic regression"},
    "glm": {"library": "statsmodels", "pip": "statsmodels", "description": "Generalized Linear Models"},
    "gee": {"library": "statsmodels", "pip": "statsmodels", "description": "Generalized Estimating Equations"},
    "quantile": {"library": "statsmodels", "pip": "statsmodels", "description": "Quantile regression"},
    "robust": {"library": "statsmodels", "pip": "statsmodels", "description": "Robust regression (M-estimators, RLM)"},
    "heckman": {"library": "statsmodels", "pip": "statsmodels", "description": "Heckman selection model"},
    "tobit": {"library": "statsmodels", "pip": "statsmodels", "description": "Tobit censored regression"},
    "difference in differences": {"library": "linearmodels", "pip": "linearmodels", "description": "Difference-in-differences with fixed effects"},
    "diff in diff": {"library": "linearmodels", "pip": "linearmodels", "description": "Difference-in-differences with fixed effects"},
    "fixed effects": {"library": "linearmodels", "pip": "linearmodels", "description": "Panel data fixed effects estimation"},
    "random effects": {"library": "linearmodels", "pip": "linearmodels", "description": "Panel data random effects estimation"},
    "hausman": {"library": "linearmodels", "pip": "linearmodels", "description": "Hausman test for panel models"},
    "granger": {"library": "statsmodels", "pip": "statsmodels", "description": "Granger causality test"},
    "cointegration": {"library": "statsmodels", "pip": "statsmodels", "description": "Cointegration tests (Johansen, Engle-Granger)"},
    "var": {"library": "statsmodels", "pip": "statsmodels", "description": "Vector Autoregression (VAR)"},
    "vecm": {"library": "statsmodels", "pip": "statsmodels", "description": "Vector Error Correction Model"},
    "garch": {"library": "arch", "pip": "arch", "description": "GARCH volatility models"},
    "arch": {"library": "arch", "pip": "arch", "description": "Autoregressive conditional heteroskedasticity"},
    "svm": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Support Vector Machines"},
    "random forest": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Random Forest classifier/regressor"},
    "gradient boosting": {"library": "xgboost", "pip": "xgboost", "description": "Gradient boosting (XGBoost)"},
    "xgboost": {"library": "xgboost", "pip": "xgboost", "description": "Extreme Gradient Boosting"},
    "lightgbm": {"library": "lightgbm", "pip": "lightgbm", "description": "Light Gradient Boosting Machine"},
    "naive bayes": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Naive Bayes classifier"},
    "knn": {"library": "scikit-learn", "pip": "scikit-learn", "description": "K-Nearest Neighbors"},
    "k-means": {"library": "scikit-learn", "pip": "scikit-learn", "description": "K-Means clustering"},
    "dbscan": {"library": "scikit-learn", "pip": "scikit-learn", "description": "DBSCAN density clustering"},
    "pca": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Principal Component Analysis"},
    "svd": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Singular Value Decomposition"},
    "tsne": {"library": "scikit-learn", "pip": "scikit-learn", "description": "t-SNE dimensionality reduction"},
    "umap": {"library": "umap-learn", "pip": "umap-learn", "description": "UMAP dimensionality reduction"},
    "word2vec": {"library": "gensim", "pip": "gensim", "description": "Word2Vec word embeddings"},
    "doc2vec": {"library": "gensim", "pip": "gensim", "description": "Doc2Vec document embeddings"},
    "fasttext": {"library": "fasttext", "pip": "fasttext", "description": "FastText word embeddings"},
    "named entity": {"library": "spacy", "pip": "spacy", "description": "Named Entity Recognition (NER)"},
    "ner": {"library": "spacy", "pip": "spacy", "description": "Named Entity Recognition"},
    "dependency parsing": {"library": "spacy", "pip": "spacy", "description": "Syntactic dependency parsing"},
    "coreference": {"library": "spacy", "pip": "spacy", "description": "Coreference resolution"},
    "summarization": {"library": "transformers", "pip": "transformers", "description": "Text summarization (BART, T5, Pegasus)"},
    "translation": {"library": "transformers", "pip": "transformers", "description": "Neural machine translation"},
    "qa": {"library": "transformers", "pip": "transformers", "description": "Question answering (Extractive/Generative)"},
    "question answering": {"library": "transformers", "pip": "transformers", "description": "Question answering"},
    "zero-shot": {"library": "transformers", "pip": "transformers", "description": "Zero-shot text classification"},
    "few-shot": {"library": "transformers", "pip": "transformers", "description": "Few-shot learning with LLMs"},
    "contrastive learning": {"library": "sentence-transformers", "pip": "sentence-transformers", "description": "Contrastive sentence embeddings"},
    "similarity": {"library": "sentence-transformers", "pip": "sentence-transformers", "description": "Semantic similarity search"},
    "semantic search": {"library": "sentence-transformers", "pip": "sentence-transformers", "description": "Semantic search with embeddings"},
    "image classification": {"library": "torch", "pip": "torch", "description": "Deep learning image classification"},
    "object detection": {"library": "torch", "pip": "torch", "description": "Object detection (YOLO, Faster R-CNN)"},
    "segmentation": {"library": "torch", "pip": "torch", "description": "Image/instance segmentation"},
    "ocr": {"library": "pytesseract", "pip": "pytesseract", "description": "Optical Character Recognition"},
    "table extraction": {"library": "camelot-py", "pip": "camelot-py", "description": "PDF table extraction"},
    "pdf parsing": {"library": "pymupdf", "pip": "pymupdf", "description": "PDF text and image extraction"},
    "web scraping": {"library": "scrapy", "pip": "scrapy", "description": "Web scraping framework"},
    "html parsing": {"library": "beautifulsoup4", "pip": "beautifulsoup4", "description": "HTML/XML parsing"},
    "api client": {"library": "requests", "pip": "requests", "description": "HTTP API client"},
    "database": {"library": "sqlalchemy", "pip": "sqlalchemy", "description": "SQL database ORM and toolkit"},
    "nosql": {"library": "pymongo", "pip": "pymongo", "description": "MongoDB NoSQL driver"},
    "redis": {"library": "redis", "pip": "redis", "description": "Redis in-memory data store"},
    "kafka": {"library": "kafka-python", "pip": "kafka-python", "description": "Apache Kafka client"},
    "celery": {"library": "celery", "pip": "celery", "description": "Distributed task queue"},
    "parallel": {"library": "joblib", "pip": "joblib", "description": "Parallel computing and caching"},
    "dask": {"library": "dask", "pip": "dask", "description": "Parallel computing with Dask"},
    "ray": {"library": "ray", "pip": "ray", "description": "Distributed computing framework"},
    "hypothesis testing": {"library": "scipy", "pip": "scipy", "description": "Statistical hypothesis tests"},
    "nonparametric": {"library": "scipy", "pip": "scipy", "description": "Non-parametric statistical tests"},
    "bootstrap": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Bootstrap resampling and CI"},
    "permutation": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Permutation tests"},
    "mcmc": {"library": "pymc", "pip": "pymc", "description": "Markov Chain Monte Carlo sampling"},
    "variational inference": {"library": "pymc", "pip": "pymc", "description": "Variational inference (ADVI)"},
    "gp": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Gaussian Process regression/classification"},
    "gaussian process": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Gaussian Process models"},
    "mlp": {"library": "torch", "pip": "torch", "description": "Multi-layer perceptron neural network"},
    "cnn": {"library": "torch", "pip": "torch", "description": "Convolutional neural network"},
    "lstm": {"library": "torch", "pip": "torch", "description": "LSTM recurrent neural network"},
    "transformer": {"library": "torch", "pip": "torch", "description": "Transformer neural network architecture"},
    "gan": {"library": "torch", "pip": "torch", "description": "Generative Adversarial Networks"},
    "vae": {"library": "torch", "pip": "torch", "description": "Variational Autoencoder"},
    "diffusion": {"library": "diffusers", "pip": "diffusers", "description": "Diffusion models for generation"},
    "reinforcement learning": {"library": "stable-baselines3", "pip": "stable-baselines3", "description": "Reinforcement learning algorithms"},
    "rl": {"library": "stable-baselines3", "pip": "stable-baselines3", "description": "Reinforcement learning (PPO, DQN, A2C)"},
    "multi-armed bandit": {"library": "mabwiser", "pip": "mabwiser", "description": "Multi-armed bandit algorithms"},
    "ab test": {"library": "statsmodels", "pip": "statsmodels", "description": "A/B testing statistical framework"},
    "split testing": {"library": "statsmodels", "pip": "statsmodels", "description": "A/B and multivariate testing"},
    "mab": {"library": "mabwiser", "pip": "mabwiser", "description": "Multi-armed bandit for online experiments"},
    "uplift": {"library": "causalml", "pip": "causalml", "description": "Uplift modeling for treatment effects"},
    "matching": {"library": "dowhy", "pip": "dowhy", "description": "Propensity score and covariate matching"},
    "synthetic control": {"library": "pysynthdid", "pip": "pysynthdid", "description": "Synthetic Control Method"},
    "synthdid": {"library": "pysynthdid", "pip": "pysynthdid", "description": "Synthetic Difference-in-Differences"},
    "event study": {"library": "linearmodels", "pip": "linearmodels", "description": "Event study with dynamic treatment effects"},
    "staggered did": {"library": "did2s", "pip": "did2s", "description": "Staggered difference-in-differences (Gardner 2022)"},
    "bacon decomposition": {"library": "did2s", "pip": "did2s", "description": "Goodman-Bacon decomposition"},
    "parallel trends": {"library": "linearmodels", "pip": "linearmodels", "description": "Parallel trends testing"},
    "fuzzy rdd": {"library": "rdrobust", "pip": "rdrobust", "description": "Fuzzy Regression Discontinuity"},
    "sharp rdd": {"library": "rdrobust", "pip": "rdrobust", "description": "Sharp Regression Discontinuity"},
    "iv regression": {"library": "linearmodels", "pip": "linearmodels", "description": "Instrumental Variables 2SLS"},
    "weak iv": {"library": "linearmodels", "pip": "linearmodels", "description": "Weak instrument diagnostics"},
    "overidentification": {"library": "linearmodels", "pip": "linearmodels", "description": "Sargan/Hansen overidentification test"},
    "bootstrap inference": {"library": "arch", "pip": "arch", "description": "Bootstrap inference for time series"},
    "wild bootstrap": {"library": "wildboottest", "pip": "wildboottest", "description": "Wild cluster bootstrap"},
    "spatial regression": {"library": "pysal", "pip": "pysal", "description": "Spatial econometrics (SAR, SEM, SLX)"},
    "moran": {"library": "pysal", "pip": "pysal", "description": "Moran's I spatial autocorrelation"},
    "spatial weights": {"library": "pysal", "pip": "pysal", "description": "Spatial weights matrix construction"},
    "network analysis": {"library": "networkx", "pip": "networkx", "description": "Network analysis and graph algorithms"},
    "social network": {"library": "networkx", "pip": "networkx", "description": "Social network analysis (SNA)"},
    "community detection": {"library": "networkx", "pip": "networkx", "description": "Graph community detection"},
    "link prediction": {"library": "networkx", "pip": "networkx", "description": "Network link prediction"},
    "influence maximization": {"library": "networkx", "pip": "networkx", "description": "Influence maximization in networks"},
    " spectral clustering": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Spectral clustering"},
    "hdbscan": {"library": "hdbscan", "pip": "hdbscan", "description": "HDBSCAN hierarchical density clustering"},
    "optics": {"library": "scikit-learn", "pip": "scikit-learn", "description": "OPTICS clustering"},
    "birch": {"library": "scikit-learn", "pip": "scikit-learn", "description": "BIRCH clustering"},
    "affinity propagation": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Affinity propagation clustering"},
    "mean shift": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Mean shift clustering"},
    "gaussian mixture": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Gaussian Mixture Model (GMM)"},
    "lda": {"library": "gensim", "pip": "gensim", "description": "Latent Dirichlet Allocation"},
    "nmf": {"library": "sklearn", "pip": "scikit-learn", "description": "Non-negative Matrix Factorization"},
    "lsa": {"library": "sklearn", "pip": "scikit-learn", "description": "Latent Semantic Analysis"},
    "tf-idf": {"library": "sklearn", "pip": "scikit-learn", "description": "TF-IDF vectorization"},
    "count vectorizer": {"library": "sklearn", "pip": "scikit-learn", "description": "Bag-of-words count vectorization"},
    "bertopic": {"library": "bertopic", "pip": "bertopic", "description": "BERTopic neural topic modeling"},
    "top2vec": {"library": "top2vec", "pip": "top2vec", "description": "Top2Vec topic modeling"},
    "keybert": {"library": "keybert", "pip": "keybert", "description": "Keyword extraction with BERT"},
    "rake": {"library": "rake-nltk", "pip": "rake-nltk", "description": "RAKE keyword extraction"},
    "yake": {"library": "yake", "pip": "yake", "description": "YAKE keyword extraction"},
    "textrank": {"library": "summa", "pip": "summa", "description": "TextRank keyword/summary extraction"},
    "lexrank": {"library": "sumy", "pip": "sumy", "description": "LexRank text summarization"},
    "lsa summarization": {"library": "sumy", "pip": "sumy", "description": "LSA extractive summarization"},
    "k-core": {"library": "networkx", "pip": "networkx", "description": "K-core decomposition"},
    "pagerank": {"library": "networkx", "pip": "networkx", "description": "PageRank centrality"},
    "eigenvector centrality": {"library": "networkx", "pip": "networkx", "description": "Eigenvector centrality"},
    "betweenness": {"library": "networkx", "pip": "networkx", "description": "Betweenness centrality"},
    "closeness": {"library": "networkx", "pip": "networkx", "description": "Closeness centrality"},
    "degree centrality": {"library": "networkx", "pip": "networkx", "description": "Degree centrality"},
    "assortativity": {"library": "networkx", "pip": "networkx", "description": "Degree assortativity"},
    "triadic census": {"library": "networkx", "pip": "networkx", "description": "Triadic census"},
    "clique": {"library": "networkx", "pip": "networkx", "description": "Clique detection"},
    "bipartite": {"library": "networkx", "pip": "networkx", "description": "Bipartite graph analysis"},
    "flow network": {"library": "networkx", "pip": "networkx", "description": "Maximum flow / minimum cut"},
    "minimum spanning tree": {"library": "networkx", "pip": "networkx", "description": "Minimum spanning tree"},
    "shortest path": {"library": "networkx", "pip": "networkx", "description": "Shortest path algorithms"},
    "dijkstra": {"library": "networkx", "pip": "networkx", "description": "Dijkstra shortest path"},
    "astar": {"library": "networkx", "pip": "networkx", "description": "A* pathfinding"},
    "bellman-ford": {"library": "networkx", "pip": "networkx", "description": "Bellman-Ford shortest path"},
    "floyd-warshall": {"library": "networkx", "pip": "networkx", "description": "All-pairs shortest paths"},
    "kruskal": {"library": "networkx", "pip": "networkx", "description": "Kruskal MST"},
    "prim": {"library": "networkx", "pip": "networkx", "description": "Prim MST"},
    "topological sort": {"library": "networkx", "pip": "networkx", "description": "Topological sort"},
    "strongly connected": {"library": "networkx", "pip": "networkx", "description": "Strongly connected components"},
    "weakly connected": {"library": "networkx", "pip": "networkx", "description": "Weakly connected components"},
    "biconnected": {"library": "networkx", "pip": "networkx", "description": "Biconnected components"},
    "bridge": {"library": "networkx", "pip": "networkx", "description": "Bridge edges"},
    "articulation point": {"library": "networkx", "pip": "networkx", "description": "Articulation points"},
    "cycle": {"library": "networkx", "pip": "networkx", "description": "Cycle detection"},
    "eulerian": {"library": "networkx", "pip": "networkx", "description": "Eulerian path/circuit"},
    "hamiltonian": {"library": "networkx", "pip": "networkx", "description": "Hamiltonian path"},
    "isomorphism": {"library": "networkx", "pip": "networkx", "description": "Graph isomorphism"},
    "subgraph matching": {"library": "networkx", "pip": "networkx", "description": "Subgraph matching"},
    "graph edit distance": {"library": "networkx", "pip": "networkx", "description": "Graph edit distance"},
    "weisfeiler-lehman": {"library": "networkx", "pip": "networkx", "description": "Weisfeiler-Lehman graph kernels"},
    "graph neural network": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Graph Neural Networks (GNN)"},
    "gnn": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Graph Neural Networks"},
    "node classification": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Node classification with GNN"},
    "link prediction gnn": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Link prediction with GNN"},
    "graph classification": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Graph-level classification"},
    "graph regression": {"library": "torch-geometric", "pip": "torch-geometric", "description": "Graph-level regression"},
    "molecular graph": {"library": "rdkit", "pip": "rdkit", "description": "Molecular graph analysis (RDKit)"},
    "cheminformatics": {"library": "rdkit", "pip": "rdkit", "description": "Cheminformatics toolkit"},
    "bioinformatics": {"library": "biopython", "pip": "biopython", "description": "Bioinformatics sequence analysis"},
    "sequence alignment": {"library": "biopython", "pip": "biopython", "description": "DNA/protein sequence alignment"},
    "phylogenetic": {"library": "biopython", "pip": "biopython", "description": "Phylogenetic tree analysis"},
    "genomics": {"library": "biopython", "pip": "biopython", "description": "Genomic data processing"},
    "proteomics": {"library": "biopython", "pip": "biopython", "description": "Protein structure analysis"},
    "finance": {"library": "yfinance", "pip": "yfinance", "description": "Financial market data (Yahoo Finance)"},
    "portfolio": {"library": "pyportfolioopt", "pip": "pyportfolioopt", "description": "Portfolio optimization"},
    "risk": {"library": "pyfolio", "pip": "pyfolio", "description": "Financial risk analysis"},
    "backtest": {"library": "backtrader", "pip": "backtrader", "description": "Trading strategy backtesting"},
    "technical analysis": {"library": "ta-lib", "pip": "ta-lib", "description": "Technical indicators (TA-Lib)"},
    "candlestick": {"library": "mplfinance", "pip": "mplfinance", "description": "Financial candlestick charts"},
    "monte carlo": {"library": "numpy", "pip": "numpy", "description": "Monte Carlo simulation"},
    "bootstrap resampling": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Bootstrap confidence intervals"},
    "jackknife": {"library": "scipy", "pip": "scipy", "description": "Jackknife resampling"},
    "permutation test": {"library": "scipy", "pip": "scipy", "description": "Permutation hypothesis testing"},
    "exact test": {"library": "scipy", "pip": "scipy", "description": "Exact statistical tests"},
    "fishers exact": {"library": "scipy", "pip": "scipy", "description": "Fisher's exact test"},
    "mcmc sampling": {"library": "pymc", "pip": "pymc", "description": "MCMC posterior sampling"},
    "hamiltonian monte carlo": {"library": "pymc", "pip": "pymc", "description": "Hamiltonian Monte Carlo (NUTS)"},
    "no-u-turn sampler": {"library": "pymc", "pip": "pymc", "description": "NUTS MCMC sampler"},
    "variational autoencoder": {"library": "torch", "pip": "torch", "description": "VAE for generative modeling"},
    "normalizing flow": {"library": "torch", "pip": "torch", "description": "Normalizing flows for density estimation"},
    "autoregressive model": {"library": "torch", "pip": "torch", "description": "Autoregressive generative models"},
    "energy based model": {"library": "torch", "pip": "torch", "description": "Energy-based generative models"},
    "score matching": {"library": "torch", "pip": "torch", "description": "Score matching for generative modeling"},
    "flow matching": {"library": "torch", "pip": "torch", "description": "Flow matching (Rectified Flow)"},
    "consistency model": {"library": "torch", "pip": "torch", "description": "Consistency models for generation"},
    "neural ode": {"library": "torchdiffeq", "pip": "torchdiffeq", "description": "Neural Ordinary Differential Equations"},
    "neural sde": {"library": "torchsde", "pip": "torchsde", "description": "Neural Stochastic Differential Equations"},
    "deep equilibrium": {"library": "deep-equilibrium", "pip": "deep-equilibrium", "description": "Deep Equilibrium Models"},
    "neural architecture search": {"library": "optuna", "pip": "optuna", "description": "Neural Architecture Search (NAS)"},
    "hyperparameter optimization": {"library": "optuna", "pip": "optuna", "description": "Bayesian hyperparameter optimization"},
    "multi-objective": {"library": "optuna", "pip": "optuna", "description": "Multi-objective optimization"},
    "pruning": {"library": "optuna", "pip": "optuna", "description": "Neural network pruning"},
    "knowledge distillation": {"library": "torch", "pip": "torch", "description": "Knowledge distillation"},
    "model compression": {"library": "torch", "pip": "torch", "description": "Neural network compression"},
    "quantization": {"library": "torch", "pip": "torch", "description": "Model quantization (INT8, etc.)"},
    "onnx": {"library": "onnx", "pip": "onnx", "description": "ONNX model export and inference"},
    "tensorrt": {"library": "tensorrt", "pip": "tensorrt", "description": "NVIDIA TensorRT inference"},
    "openvino": {"library": "openvino", "pip": "openvino", "description": "Intel OpenVINO inference"},
    "coreml": {"library": "coremltools", "pip": "coremltools", "description": "Apple CoreML model conversion"},
    "tflite": {"library": "tensorflow", "pip": "tensorflow", "description": "TensorFlow Lite model conversion"},
    "edge deployment": {"library": "onnx", "pip": "onnx", "description": "Edge AI model deployment"},
    "federated learning": {"library": "pytorch-federated", "pip": "pytorch-federated", "description": "Federated learning framework"},
    "differential privacy": {"library": "opacus", "pip": "opacus", "description": "Differential privacy for deep learning"},
    "fairness": {"library": "fairlearn", "pip": "fairlearn", "description": "Fairness in machine learning"},
    "explainability": {"library": "shap", "pip": "shap", "description": "SHAP model explanations"},
    "lime": {"library": "lime", "pip": "lime", "description": "LIME local explanations"},
    "counterfactual": {"library": "dice-ml", "pip": "dice-ml", "description": "Counterfactual explanations"},
    "concept drift": {"library": "alibi-detect", "pip": "alibi-detect", "description": "Concept drift detection"},
    "outlier": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Outlier/anomaly detection"},
    "isolation forest": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Isolation Forest anomaly detection"},
    "local outlier factor": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Local Outlier Factor (LOF)"},
    "one-class svm": {"library": "scikit-learn", "pip": "scikit-learn", "description": "One-Class SVM anomaly detection"},
    "mahalanobis": {"library": "scipy", "pip": "scipy", "description": "Mahalanobis distance outlier detection"},
    "autoencoder anomaly": {"library": "torch", "pip": "torch", "description": "Autoencoder-based anomaly detection"},
    "time series anomaly": {"library": "adtk", "pip": "adtk", "description": "Time series anomaly detection toolkit"},
    "forecasting": {"library": "statsforecast", "pip": "statsforecast", "description": "Statistical time series forecasting"},
    "exponential smoothing": {"library": "statsforecast", "pip": "statsforecast", "description": "Exponential smoothing (ETS)"},
    "theta model": {"library": "statsforecast", "pip": "statsforecast", "description": "Theta time series forecasting"},
    "croston": {"library": "statsforecast", "pip": "statsforecast", "description": "Croston intermittent demand forecasting"},
    "nhits": {"library": "neuralforecast", "pip": "neuralforecast", "description": "N-HiTS neural forecasting"},
    "nbeats": {"library": "neuralforecast", "pip": "neuralforecast", "description": "N-BEATS neural forecasting"},
    "tft": {"library": "neuralforecast", "pip": "neuralforecast", "description": "Temporal Fusion Transformer forecasting"},
    "deepar": {"library": "neuralforecast", "pip": "neuralforecast", "description": "DeepAR probabilistic forecasting"},
    "tabular data": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Deep learning for tabular data"},
    "tabnet": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "TabNet attention-based tabular model"},
    "node": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Neural Oblivious Decision Ensembles"},
    "category embedding": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Entity embedding for categorical variables"},
    "entity embedding": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Entity embedding for categorical variables"},
    "wide and deep": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Wide & Deep learning model"},
    "factorization machine": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Factorization Machines for tabular data"},
    "deepfm": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "DeepFM for CTR prediction"},
    "dcn": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "Deep & Cross Network"},
    "autoint": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "AutoInt self-attentive tabular model"},
    "ft-transformer": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "FT-Transformer for tabular data"},
    " saint": {"library": "pytorch-tabular", "pip": "pytorch-tabular", "description": "SAINT tabular transformer"},
    "recommender": {"library": "surprise", "pip": "scikit-surprise", "description": "Collaborative filtering recommender systems"},
    "collaborative filtering": {"library": "surprise", "pip": "scikit-surprise", "description": "Collaborative filtering"},
    "matrix factorization": {"library": "surprise", "pip": "scikit-surprise", "description": "Matrix factorization (SVD, NMF)"},
    "content based": {"library": "surprise", "pip": "scikit-surprise", "description": "Content-based recommendation"},
    "hybrid recommender": {"library": "lightfm", "pip": "lightfm", "description": "Hybrid recommender (LightFM)"},
    "session based": {"library": "torch", "pip": "torch", "description": "Session-based recommendation"},
    "sequential recommender": {"library": "torch", "pip": "torch", "description": "Sequential recommendation"},
    "next item prediction": {"library": "torch", "pip": "torch", "description": "Next-item prediction"},
    "active learning": {"library": "modAL", "pip": "modAL", "description": "Active learning for classification"},
    "semi-supervised": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Semi-supervised learning"},
    "self-supervised": {"library": "torch", "pip": "torch", "description": "Self-supervised representation learning"},
    "contrastive": {"library": "torch", "pip": "torch", "description": "Contrastive representation learning"},
    "simclr": {"library": "torch", "pip": "torch", "description": "SimCLR self-supervised learning"},
    "moco": {"library": "torch", "pip": "torch", "description": "MoCo momentum contrast"},
    "byol": {"library": "torch", "pip": "torch", "description": "BYOL self-supervised learning"},
    "swav": {"library": "torch", "pip": "torch", "description": "SwAV self-supervised clustering"},
    "barlow twins": {"library": "torch", "pip": "torch", "description": "Barlow Twins self-supervised learning"},
    "vicreg": {"library": "torch", "pip": "torch", "description": "VICReg self-supervised learning"},
    "dino": {"library": "torch", "pip": "torch", "description": "DINO self-supervised vision transformer"},
    "ibot": {"library": "torch", "pip": "torch", "description": "iBOT self-supervised ViT"},
    "mae": {"library": "torch", "pip": "torch", "description": "Masked Autoencoder (MAE)"},
    "beit": {"library": "torch", "pip": "torch", "description": "BEIT self-supervised vision transformer"},
    "data augmentation": {"library": "albumentations", "pip": "albumentations", "description": "Image augmentation library"},
    "image transform": {"library": "albumentations", "pip": "albumentations", "description": "Image transformations and augmentations"},
    "mixup": {"library": "torch", "pip": "torch", "description": "Mixup data augmentation"},
    "cutmix": {"library": "torch", "pip": "torch", "description": "CutMix data augmentation"},
    "randaugment": {"library": "albumentations", "pip": "albumentations", "description": "RandAugment automatic augmentation"},
    "autoaugment": {"library": "albumentations", "pip": "albumentations", "description": "AutoAugment learned augmentation policy"},
    "test time augmentation": {"library": "albumentations", "pip": "albumentations", "description": "Test-time augmentation (TTA)"},
    "model ensemble": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Voting, stacking, blending ensembles"},
    "stacking": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Stacked generalization"},
    "blending": {"library": "scikit-learn", "pip": "scikit-learn", "description": "Blended ensemble"},
    "snapshot ensemble": {"library": "torch", "pip": "torch", "description": "Snapshot ensemble with cyclic LR"},
    "swa": {"library": "torch", "pip": "torch", "description": "Stochastic Weight Averaging"},
    "ema": {"library": "torch", "pip": "torch", "description": "Exponential Moving Average"},
    "lookahead": {"library": "torch", "pip": "torch", "description": "Lookahead optimizer"},
    "sharpness aware minimization": {"library": "torch", "pip": "torch", "description": "SAM sharpness-aware minimization"},
    "learning rate scheduling": {"library": "torch", "pip": "torch", "description": "LR schedulers (cosine, warm restarts)"},
    "cosine annealing": {"library": "torch", "pip": "torch", "description": "Cosine annealing LR scheduler"},
    "warmup": {"library": "torch", "pip": "torch", "description": "Learning rate warmup"},
    "gradient clipping": {"library": "torch", "pip": "torch", "description": "Gradient clipping for training stability"},
    "mixture of experts": {"library": "torch", "pip": "torch", "description": "Mixture of Experts (MoE)"},
    "switch transformer": {"library": "torch", "pip": "torch", "description": "Switch Transformer MoE"},
    "expert choice": {"library": "torch", "pip": "torch", "description": "Expert Choice routing"},
    "token mixture": {"library": "torch", "pip": "torch", "description": "Token-mixing MLP"},
    "mlp-mixer": {"library": "torch", "pip": "torch", "description": "MLP-Mixer architecture"},
    "convnext": {"library": "torch", "pip": "torch", "description": "ConvNeXt pure ConvNet"},
    "poolformer": {"library": "torch", "pip": "torch", "description": "PoolFormer architecture"},
    "repvgg": {"library": "torch", "pip": "torch", "description": "RepVGG reparameterization"},
    "mobileone": {"library": "torch", "pip": "torch", "description": "MobileOne efficient mobile architecture"},
    "efficientnet": {"library": "torch", "pip": "torch", "description": "EfficientNet scaling"},
    "mobilevit": {"library": "torch", "pip": "torch", "description": "MobileViT mobile vision transformer"},
    "fastvit": {"library": "torch", "pip": "torch", "description": "FastViT efficient vision transformer"},
    "edgenext": {"library": "torch", "pip": "torch", "description": "EdgeNeXt hybrid architecture"},
    "shufflenet": {"library": "torch", "pip": "torch", "description": "ShuffleNet channel shuffle"},
    "mobilenet": {"library": "torch", "pip": "torch", "description": "MobileNet depthwise separable conv"},
    "squeezenet": {"library": "torch", "pip": "torch", "description": "SqueezeNet fire module"},
    "densenet": {"library": "torch", "pip": "torch", "description": "DenseNet dense connections"},
    "resnet": {"library": "torch", "pip": "torch", "description": "ResNet residual connections"},
    "resnext": {"library": "torch", "pip": "torch", "description": "ResNeXt grouped convolutions"},
    "senet": {"library": "torch", "pip": "torch", "description": "SENet squeeze-and-excitation"},
    "sknet": {"library": "torch", "pip": "torch", "description": "SKNet selective kernel"},
    "ecanet": {"library": "torch", "pip": "torch", "description": "ECA-Net efficient channel attention"},
    "cbam": {"library": "torch", "pip": "torch", "description": "CBAM convolutional block attention"},
    "bam": {"library": "torch", "pip": "torch", "description": "BAM bottleneck attention module"},
    "coordinate attention": {"library": "torch", "pip": "torch", "description": "Coordinate attention for mobile networks"},
    "squeeze and excitation": {"library": "torch", "pip": "torch", "description": "Squeeze-and-Excitation attention"},
    "inverted residual": {"library": "torch", "pip": "torch", "description": "Inverted residual block"},
    "ghost module": {"library": "torch", "pip": "torch", "description": "Ghost module cheap operations"},
    "sandglass block": {"library": "torch", "pip": "torch", "description": "Sandglass block for mobile ViT"},
    "patch merging": {"library": "torch", "pip": "torch", "description": "Vision transformer patch merging"},
    "shifted window": {"library": "torch", "pip": "torch", "description": "Swin Transformer shifted window"},
    "hierarchical transformer": {"library": "torch", "pip": "torch", "description": "Hierarchical vision transformer"},
    "cross attention": {"library": "torch", "pip": "torch", "description": "Cross-attention mechanism"},
    "self attention": {"library": "torch", "pip": "torch", "description": "Self-attention mechanism"},
    "multi-head attention": {"library": "torch", "pip": "torch", "description": "Multi-head self-attention"},
    "rotary embedding": {"library": "torch", "pip": "torch", "description": "RoPE rotary position embedding"},
    "alibi": {"library": "torch", "pip": "torch", "description": "ALiBi linear bias attention"},
    "relative position": {"library": "torch", "pip": "torch", "description": "Relative position encoding"},
    "learned position": {"library": "torch", "pip": "torch", "description": "Learned absolute position embedding"},
    "sinusoidal position": {"library": "torch", "pip": "torch", "description": "Sinusoidal position encoding"},
    "layer normalization": {"library": "torch", "pip": "torch", "description": "Layer normalization (Pre-LN, Post-LN)"},
    "rmsnorm": {"library": "torch", "pip": "torch", "description": "RMSNorm root mean square normalization"},
    "group normalization": {"library": "torch", "pip": "torch", "description": "Group normalization"},
    "instance normalization": {"library": "torch", "pip": "torch", "description": "Instance normalization"},
    "batch normalization": {"library": "torch", "pip": "torch", "description": "Batch normalization"},
    "switchable normalization": {"library": "torch", "pip": "torch", "description": "Switchable normalization"},
    "filter response normalization": {"library": "torch", "pip": "torch", "description": "Filter Response Normalization"},
    "evo norm": {"library": "torch", "pip": "torch", "description": "EvoNorm evolutionary normalization"},
    "dropout": {"library": "torch", "pip": "torch", "description": "Dropout regularization"},
    "drop path": {"library": "torch", "pip": "torch", "description": "Stochastic depth (Drop Path)"},
    "mixout": {"library": "torch", "pip": "torch", "description": "Mixout regularization"},
    "dropblock": {"library": "torch", "pip": "torch", "description": "DropBlock structured dropout"},
    "cutout": {"library": "torch", "pip": "torch", "description": "Cutout regularization"},
    "hide and seek": {"library": "torch", "pip": "torch", "description": "Hide-and-Seek augmentation"},
    "grid mask": {"library": "torch", "pip": "torch", "description": "GridMask data augmentation"},
    "random erasing": {"library": "torch", "pip": "torch", "description": "Random Erasing augmentation"},
    "spectral normalization": {"library": "torch", "pip": "torch", "description": "Spectral normalization for GANs"},
    "weight standardization": {"library": "torch", "pip": "torch", "description": "Weight Standardization"},
    "fixup initialization": {"library": "torch", "pip": "torch", "description": "Fixup initialization without normalization"},
    "zero init residual": {"library": "torch", "pip": "torch", "description": "Zero-initialize residual branches"},
    "rezero": {"library": "torch", "pip": "torch", "description": "ReZero residual connection"},
    "deepnorm": {"library": "torch", "pip": "torch", "description": "DeepNorm for deep transformers"},
    "rescale norm": {"library": "torch", "pip": "torch", "description": "RescaleNorm for training stability"},
    "xavier initialization": {"library": "torch", "pip": "torch", "description": "Xavier/Glorot initialization"},
    "kaiming initialization": {"library": "torch", "pip": "torch", "description": "Kaiming/He initialization"},
    "orthogonal initialization": {"library": "torch", "pip": "torch", "description": "Orthogonal weight initialization"},
    "normal initialization": {"library": "torch", "pip": "torch", "description": "Normal/Gaussian initialization"},
    "uniform initialization": {"library": "torch", "pip": "torch", "description": "Uniform weight initialization"},
    "constant initialization": {"library": "torch", "pip": "torch", "description": "Constant weight initialization"},
    "dirichlet initialization": {"library": "torch", "pip": "torch", "description": "Dirichlet weight initialization"},
    "spectral init": {"library": "torch", "pip": "torch", "description": "Spectral initialization"},
}


# Synonym expansion map: common variant → canonical keyword used in KEYWORD_LIBRARY_MAP
SYNONYM_MAP = {
    "t test": "t-test",
    "ttest": "t-test",
    "t-test": "t-test",
    "logistic": "logistic regression",
    "logit": "logistic regression",
    "ols": "regression",
    "linear model": "regression",
    "diff in diff": "difference in differences",
    "diff-in-diff": "difference in differences",
    "svm": "svm",
    "support vector": "svm",
    "random forest": "random forest",
    "rf": "random forest",
    "gradient boost": "gradient boosting",
    "xgb": "xgboost",
    "light gbm": "lightgbm",
    "k means": "k-means",
    "kmeans": "k-means",
    "principal component": "pca",
    "neural network": "mlp",
    "nn": "mlp",
    "natural language": "nlp",
    "topic model": "topic",
    "graph analysis": "graph",
    "forecast": "forecasting",
    "fixed effect": "fixed effects",
    "random effect": "random effects",
    "bayes": "bayesian",
    "hierarchical model": "hierarchical",
    "meta analysis": "meta-analysis",
    "experiment design": "design",
    "power analysis": "power",
    "auto ml": "automl",
    "cross validation": "cross-validation",
    "feature selection": "feature",
    "interpret": "explainability",
    "cluster": "clustering",
    "classifier": "classify",
    "regress": "regression",
    "recommend": "recommender",
    "recommendation": "recommender",
    "matrix factor": "matrix factorization",
    "language model": "llm",
    "generative": "llm",
    "retrieval": "rag",
    "augmented": "rag",
    "computer vision": "image",
    "web scrape": "web scraping",
    "scrape": "web scraping",
    "sql": "database",
    "distributed": "parallel",
    "cuda": "gpu",
}


class MethodSearcher:
    def __init__(self, catalog, provider=None):
        self.catalog = catalog
        self.provider = provider

    def _expand_synonyms(self, description: str) -> str:
        """Expand description with canonical synonyms to improve keyword matching."""
        desc_lower = description.lower()
        extras = []
        for variant, canonical in SYNONYM_MAP.items():
            if variant in desc_lower and canonical not in desc_lower:
                extras.append(canonical)
        if extras:
            return desc_lower + " " + " ".join(extras)
        return desc_lower

    def search(self, method_description: str, category: str = None) -> str:
        """Search for a method. Flow:
        1. Search catalog first
        2. If found, return info
        3. If not, search external sources (keyword-library mapping)
        4. Return candidates

        Args:
            method_description: Description of the method to search for
            category: Optional category filter

        Returns:
            JSON string with found status, candidates list
        """
        if not method_description or not method_description.strip():
            return json.dumps({
                "found": False,
                "message": "Empty method description provided",
                "candidates": [],
            }, ensure_ascii=False)

        # Step 1: Search catalog
        catalog_results = self._search_catalog(method_description, category)
        if catalog_results:
            # Check if any are already installed
            installed = [m for m in catalog_results if m.get("status") == "installed"]
            if installed:
                return json.dumps({
                    "found": True,
                    "source": "catalog",
                    "status": "installed",
                    "methods": installed,
                    "message": f"Found {len(installed)} installed method(s) matching '{method_description}'",
                }, ensure_ascii=False)

            # Known but not installed
            known = [m for m in catalog_results if m.get("status") != "installed"]
            if known:
                return json.dumps({
                    "found": True,
                    "source": "catalog",
                    "status": "known_not_installed",
                    "methods": known,
                    "message": f"Found {len(known)} known but not installed method(s). Consider installing.",
                }, ensure_ascii=False)

        # Step 2: Generate candidates from external sources
        candidates = self._generate_candidates(method_description, category)

        if not candidates:
            return json.dumps({
                "found": False,
                "source": "none",
                "message": f"No methods found matching '{method_description}'. Try a different description.",
                "candidates": [],
            }, ensure_ascii=False)

        # Validate candidates
        validated = []
        for c in candidates:
            c["importable"] = self._validate_candidate(c)
            validated.append(c)

        return json.dumps({
            "found": True,
            "source": "external",
            "status": "candidate",
            "candidates": validated,
            "message": f"Found {len(validated)} candidate method(s). Install dependencies and build to activate.",
        }, ensure_ascii=False)

    def _search_catalog(self, description: str, category: str = None) -> List[dict]:
        """Search local catalog."""
        return self.catalog.search(description, category=category)

    def _generate_candidates(self, description: str, category: str = None) -> List[dict]:
        """Generate candidate methods based on description.
        Uses synonym expansion + keyword matching, then LLM fallback."""
        candidates = []
        seen_libraries = set()
        # Expand with synonyms for broader matching
        desc_lower = self._expand_synonyms(description).strip()

        # Collect all known libraries for direct name detection.
        known_libs = {}
        for info in KEYWORD_LIBRARY_MAP.values():
            lib = info["library"]
            if lib not in known_libs:
                known_libs[lib] = info

        # Keyword matching against the map
        for keyword, info in KEYWORD_LIBRARY_MAP.items():
            if keyword in desc_lower:
                lib = info["library"]
                if lib not in seen_libraries:
                    seen_libraries.add(lib)
                    candidates.append({
                        "name": description.strip().title(),
                        "library": lib,
                        "pip_name": info["pip"],
                        "description": info["description"],
                        "category": category or self._infer_category(description),
                        "source": "keyword_match",
                    })

        # Also detect when user explicitly mentions a library name in description.
        for lib, info in known_libs.items():
            if lib not in seen_libraries and lib.lower() in desc_lower:
                seen_libraries.add(lib)
                candidates.append({
                    "name": description.strip().title(),
                    "library": lib,
                    "pip_name": info["pip"],
                    "description": info["description"],
                    "category": category or self._infer_category(description),
                    "source": "library_name_in_description",
                })

        # If provider (LLM) is available, also ask for suggestions
        if self.provider is not None and hasattr(self.provider, "chat"):
            try:
                llm_candidates = self._ask_llm_for_candidates(description, category)
                for c in llm_candidates:
                    lib = c.get("library", "")
                    if lib and lib not in seen_libraries:
                        seen_libraries.add(lib)
                        c["source"] = "llm_suggestion"
                        candidates.append(c)
            except Exception:
                pass  # LLM failure is non-fatal

        # Fallback: if no keyword match, try the whole description as a library name
        if not candidates:
            words = desc_lower.replace("-", "_").split()
            for word in words:
                if len(word) >= 3 and word not in seen_libraries:
                    candidates.append({
                        "name": description.strip().title(),
                        "library": word,
                        "pip_name": word,
                        "description": f"Potential library for: {description.strip()}",
                        "category": category or self._infer_category(description),
                        "source": "name_heuristic",
                    })
                    break  # Only add the first plausible word

        return candidates

    def _validate_candidate(self, candidate: dict) -> bool:
        """Check if candidate's dependencies can be imported."""
        library = candidate.get("library", "")
        if not library:
            return False
        # Normalize package name for import check
        import_name = library.replace("-", "_").replace(".", "_")
        try:
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def _infer_category(self, description: str) -> str:
        """Infer method category from description with expanded keyword coverage."""
        desc = description.lower()
        category_keywords = {
            "statistics": [
                "test", "mean", "variance", "anova", "t-test", "correlation", "regression",
                "normality", "effect size", "distribution", "hypothesis", "p-value", "significance",
                "confidence interval", "std dev", "standard deviation", "median", "quantile",
                "percentile", "skewness", "kurtosis", "mann-whitney", "wilcoxon", "kruskal",
                "chi-square", "fisher", "levene", "bartlett", "kolmogorov", "shapiro", "anderson",
                "durbin-watson", "jarque-bera", "grubbs", "dixon", "cochran", "friedman",
                "mood", "fligner", "mauchly", "box-cox", "yeo-johnson", "brown-forsythe",
                "logistic regression", "logit", "probit", "poisson", "negative binomial",
                "ordinal", "multinomial", "glm", "gee", "quantile regression", "robust",
                "heckman", "tobit", "survival", "kaplan-meier", "cox", "hazard", "log-rank",
                "time series", "arima", "var", "vecm", "granger", "cointegration", "garch",
                "exponential smoothing", "forecasting", "croston", "theta",
            ],
            "design": [
                "design", "factorial", "experiment", "random", "power", "sample size", "sampling",
                "stratified", "cluster", "systematic", "convenience", "quota", "snowball",
                "randomized controlled", "rct", "control group", "treatment group", "placebo",
                "blind", "double blind", "crossover", "latin square", "orthogonal", "balanced",
                "incomplete block", "split plot", "repeated measures", "within-subject",
                "between-subject", "mixed design", "counterbalanced", "matched pairs",
                "blocking", "randomization", "allocation", "balance", "stratification",
            ],
            "causal": [
                "causal", "treatment", "effect", "did", "rdd", "instrumental", "propensity",
                "mediation", "counterfactual", "difference in differences", "regression discontinuity",
                "synthetic control", "matching", "weighting", "inverse probability", "doubly robust",
                "heckman", "tobit", "selection bias", "endogeneity", "exogeneity", "confounder",
                "omitted variable", "simultaneity", "reverse causality", "natural experiment",
                "quasi-experiment", "policy evaluation", "program evaluation", "impact evaluation",
                "event study", "staggered", "goodman-bacon", "parallel trends",
                "weak iv", "overidentification", "sargan", "hausman", "durbin-wu-hausman",
                "first stage", "f-statistic", "stock-yogo", "anderson-rubin",
            ],
            "survey": [
                "survey", "cronbach", "likert", "questionnaire", "reliability", "factor analysis",
                "item", "scale", "index", "composite", "summated", "rating", "semantic differential",
                "guttman", "thurstone", "mokken", "rasch", "irt", "differential item functioning",
                "test-retest", "inter-rater", "intra-rater", "parallel forms", "split-half",
                "omega", "mcdonald", "composite reliability", "average variance extracted",
                "discriminant validity", "convergent validity", "multitrait-multimethod",
                "item response", "psychometric", "measurement", "validation", "validation study",
            ],
            "qualitative": [
                "qualitative", "thematic", "coding", "sentiment", "content analysis",
                "grounded", "interview", "focus group", "ethnography", "case study",
                "phenomenology", "narrative", "discourse", "conversation", "semiotic",
                "hermeneutic", "phenomenological", "life history", "oral history",
                "participant observation", "field notes", "memo", "axial", "selective",
                "open coding", "in-vivo", "process coding", "emotion coding", "values coding",
                "versus coding", "evaluative", "domain", "taxonomic", "componential",
                "n vivo", "atlas.ti", "nvivo", "maxqda", "dedoose",
            ],
            "meta": [
                "meta-analysis", "meta analysis", "heterogeneity", "publication bias", "forest",
                "pooled", "systematic review", "effect size", "fixed effect", "random effect",
                "mixed effect", "inverse variance", "mantel-haenszel", "peto", "dersimonian",
                "laird", "hunter-schmidt", "rosenthal", "glass", "hedges", "cochran", "egger",
                "begg", "trim and fill", "fail-safe n", "rosenbaum", "selection model",
                "bivariate", "multivariate", "network meta", "cumulative", "individual patient",
                "bayesian meta", "model-based", "multilevel meta", "funnel", "galbraith", "lasso",
            ],
            "computational": [
                "simulation", "agent", "network", "topic", "text", "embedding", "classify",
                "monte carlo", "bootstrap", "permutation", "resampling", "jackknife",
                "cross-validation", "bootstrapping", "numerical", "algorithm", "heuristic",
                "optimization", "genetic algorithm", "simulated annealing", "tabu search",
                "particle swarm", "ant colony", "differential evolution", "evolutionary",
                "swarm intelligence", "cellular automata", "lattice", "ising", "percolation",
                "random walk", "markov chain", "hidden markov", "brownian motion",
                "stochastic process", "queueing", "renewal", "branching", "birth-death",
                "agent-based", "abm", "opinion dynamics", "schelling", "sir", "seir",
                "social network", "community detection", "link prediction", "influence",
            ],
            "ml": [
                "machine learning", "ml", "train", "model", "classify", "predict", "feature",
                "cross-validation", "hyperparameter", "ensemble", "automl", "deep learning", "neural",
                "supervised", "unsupervised", "semi-supervised", "self-supervised", "reinforcement",
                "transfer learning", "few-shot", "zero-shot", "one-shot", "meta-learning", "multi-task",
                "multi-label", "imbalanced", "cost-sensitive", "online learning", "incremental",
                "active learning", "curriculum", "self-training", "co-training", "tri-training",
                "democratic", "boosting", "bagging", "random forest", "gradient boosting", "adaboost",
                "xgboost", "lightgbm", "catboost", "svm", "kernel", "gaussian process", "naive bayes",
                "knn", "decision tree", "random tree", "extra trees", "mlp", "cnn", "rnn", "lstm",
                "gru", "transformer", "attention", "bert", "gpt", "resnet", "vgg", "inception",
                "efficientnet", "mobilenet", "yolo", "rcnn", "fast r-cnn", "mask r-cnn", "ssd",
                "retinanet", "unet", "segnet", "deeplab", "fcn", "pspnet", "ocr", "crnn", "east",
                "clustering", "k-means", "dbscan", "hdbscan", "optics", "birch", "affinity",
                "mean shift", "gaussian mixture", "spectral clustering", "pca", "svd", "tsne", "umap",
                "outlier", "anomaly", "isolation forest", "local outlier", "one-class", "mahalanobis",
                "dimensionality reduction", "manifold learning", "feature engineering", "selection",
                "extraction", "encoding", "scaling", "normalization", "standardization", "imputation",
                "pipeline", "sklearn", "scikit-learn", "torch", "pytorch", "tensorflow", "keras",
            ],
            "llm": [
                "llm", "language model", "prompt", "rag", "benchmark", "generation", "gpt", "bert",
                "transformer", "attention", "fine-tuning", "instruction", "chatbot", "dialogue",
                "summarization", "translation", "qa", "question answering", "code generation",
                "text completion", "embedding", "retrieval", "vector store", "chain", "agent",
                "tool use", "function calling", "react", "reflexion", "tree of thoughts",
                "graph of thoughts", "chain-of-thought", "few-shot prompting", "zero-shot",
                "in-context learning", "alignment", "rlhf", "dpo", "ppo", "constitutional",
                "red teaming", "jailbreak", "hallucination", "faithfulness", "factuality",
                "coherence", "fluency", "perplexity", "bleu", "rouge", "meteor", "bertscore",
                "mauve", "diversity", "novelty", "repetition", "toxicity",
                "bias", "stereotype", "fairness", "privacy", "copyright", "watermark",
                "hugging face", "transformers", "tokenization", "tokenizer", "vocabulary",
            ],
            "visualization": [
                "plot", "chart", "graph", "heatmap", "visualization", "dashboard",
                "figure", "diagram", "scatter", "histogram", "bar chart", "line chart",
                "pie chart", "box plot", "violin", "density", "contour", "surface",
                "3d plot", "animation", "interactive", "tooltip", "legend", "axis",
                "facet", "small multiple", "treemap", "sunburst", "sankey", "chord",
                "network diagram", "word cloud", "timeline", "geographic", "map",
                "choropleth", "cartogram", "bubble map", "flow map", "hexbin",
                "raster", "image", "video", "gif", "svg", "pdf", "png", "export",
                "matplotlib", "seaborn", "plotly", "bokeh", "altair", "ggplot",
                "candlestick", "financial chart", "scientific plot", "publication figure",
            ],
        }
        for cat, keywords in category_keywords.items():
            for kw in keywords:
                if kw in desc:
                    return cat
        return "uncategorized"

    def _ask_llm_for_candidates(self, description: str, category: str = None) -> List[dict]:
        """Ask LLM for library suggestions (when provider is available)."""
        prompt = (
            f"Suggest Python libraries for this research method: '{description}'."
            f"{' Category: ' + category + '.' if category else ''}"
            " Return a JSON array of objects with keys: library, pip_name, description."
            " Only include libraries available on PyPI. Max 3 suggestions."
        )
        try:
            response = self.provider.chat([{"role": "user", "content": prompt}])
            text = response.content if response else ""
            if isinstance(text, str):
                # Try to extract JSON from response
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            return []
        except (json.JSONDecodeError, AttributeError, TypeError):
            return []
