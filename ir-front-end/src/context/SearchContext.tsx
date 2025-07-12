import React, { createContext, useContext, useState, ReactNode } from 'react';
import { searchDocuments, SearchRequest, SearchResponse } from '../services/api';

export type SearchAlgorithm = 'tf-idf' | 'embedding' | 'bm25' | 'hybrid';
export type Dataset = 'antique' | 'beir/quora';

export interface SearchResult {
  id: string;
  title: string;
  snippet: string;
  score: number;
  url: string;
  cluster?: string;
}

export interface SearchState {
  query: string;
  algorithm: SearchAlgorithm;
  dataset: Dataset;
  resultCount: number;
  useIndexing: boolean;
  useVectorStore: boolean;
  results: SearchResult[];
  searchTime: number;
  totalResults: number;
}

interface SearchContextType {
  searchState: SearchState;
  setSearchState: (state: Partial<SearchState>) => void;
  performSearch: (query: string, algorithm: SearchAlgorithm, dataset: Dataset, resultCount: number, useIndexing: boolean, useVectorStore: boolean) => void;
}

const SearchContext = createContext<SearchContextType | undefined>(undefined);

export const useSearch = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error('useSearch must be used within a SearchProvider');
  }
  return context;
};

const mockDocuments = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    snippet: 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
    score: 0.95,
    url: '/document/1',
    fullText: `# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.

## Key Concepts

### Supervised Learning
Supervised learning algorithms learn from labeled training data, helping to predict outcomes for unforeseen data. Common applications include:
- Classification problems
- Regression analysis
- Predictive modeling

### Unsupervised Learning
Unsupervised learning finds hidden patterns or intrinsic structures in input data. It's used for:
- Clustering
- Association rule learning
- Dimensionality reduction

### Reinforcement Learning
Reinforcement learning is an area of machine learning where an agent learns to behave in an environment by performing actions and receiving rewards or penalties.

## Applications

Machine learning has numerous real-world applications:
- Image and speech recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis
- Financial fraud detection

The field continues to evolve rapidly, with new techniques and applications emerging regularly.`
  },
  {
    id: '2',
    title: 'Deep Learning Fundamentals',
    snippet: 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.',
    score: 0.92,
    url: '/document/2',
    fullText: `# Deep Learning Fundamentals

Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It's inspired by the structure and function of the human brain.

## Neural Networks

### Basic Structure
A neural network consists of:
- Input layer: Receives data
- Hidden layers: Process information
- Output layer: Provides results

### Activation Functions
Common activation functions include:
- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh
- Softmax

## Popular Architectures

### Convolutional Neural Networks (CNNs)
Primarily used for image processing and computer vision tasks. They excel at:
- Image classification
- Object detection
- Facial recognition

### Recurrent Neural Networks (RNNs)
Designed for sequential data, including:
- Natural language processing
- Time series analysis
- Speech recognition

### Transformer Networks
Revolutionary architecture for handling sequential data:
- Language translation
- Text generation
- Attention mechanisms

## Training Process

Deep learning models require:
- Large datasets
- Computational power
- Careful hyperparameter tuning
- Regularization techniques

The field has seen remarkable advances in recent years, enabling breakthrough applications in various domains.`
  },
  {
    id: '3',
    title: 'Natural Language Processing Techniques',
    snippet: 'Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.',
    score: 0.88,
    url: '/document/3',
    fullText: `# Natural Language Processing Techniques

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. It bridges the gap between human communication and computer understanding.

## Core Components

### Tokenization
Breaking text into individual words, phrases, or symbols:
- Word tokenization
- Sentence segmentation
- Subword tokenization

### Part-of-Speech Tagging
Identifying grammatical roles:
- Nouns, verbs, adjectives
- Named entity recognition
- Syntactic parsing

### Semantic Analysis
Understanding meaning:
- Word embeddings
- Sentiment analysis
- Topic modeling

## Modern Techniques

### Transformer Models
Revolutionary architecture that has transformed NLP:
- Self-attention mechanisms
- Parallel processing capabilities
- Scalability to large datasets

### Pre-trained Models
Large language models trained on vast corpora:
- BERT (Bidirectional Encoder Representations from Transformers)
- GPT (Generative Pre-trained Transformer)
- T5 (Text-to-Text Transfer Transformer)

### Fine-tuning
Adapting pre-trained models for specific tasks:
- Domain adaptation
- Task-specific optimization
- Transfer learning

## Applications

NLP powers many modern applications:
- Search engines
- Machine translation
- Chatbots and virtual assistants
- Content summarization
- Document classification
- Information extraction

The field continues to advance rapidly, with new models and techniques emerging regularly.`
  },
  {
    id: '4',
    title: 'Computer Vision Applications',
    snippet: 'Computer vision is a field of AI that trains computers to interpret and understand the visual world through digital images and videos.',
    score: 0.86,
    url: '/document/4',
    fullText: `# Computer Vision Applications

Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos, machines can accurately identify and classify objects.

## Fundamental Concepts

### Image Processing
Basic operations on digital images:
- Filtering and enhancement
- Edge detection
- Color space conversion
- Histogram analysis

### Feature Extraction
Identifying important characteristics:
- Corners and edges
- Textures and patterns
- Shape descriptors
- Color features

### Object Detection
Locating and identifying objects:
- Bounding box regression
- Classification confidence
- Multi-object detection
- Real-time processing

## Deep Learning in Computer Vision

### Convolutional Neural Networks
Specialized networks for image processing:
- Convolution layers
- Pooling operations
- Feature maps
- Hierarchical learning

### Popular Architectures
- LeNet: Early CNN architecture
- AlexNet: Breakthrough in image classification
- ResNet: Residual networks for very deep architectures
- YOLO: Real-time object detection

### Transfer Learning
Leveraging pre-trained models:
- Feature extraction
- Fine-tuning
- Domain adaptation

## Real-World Applications

### Medical Imaging
- X-ray analysis
- MRI and CT scan interpretation
- Pathology detection
- Surgical assistance

### Autonomous Vehicles
- Lane detection
- Traffic sign recognition
- Pedestrian detection
- Obstacle avoidance

### Security and Surveillance
- Face recognition
- Activity monitoring
- Anomaly detection
- Access control

### Industrial Applications
- Quality control
- Defect detection
- Robotic guidance
- Inventory management

The field continues to evolve with new techniques and applications emerging across various industries.`
  },
  {
    id: '5',
    title: 'Reinforcement Learning Algorithms',
    snippet: 'Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize reward.',
    score: 0.84,
    url: '/document/5',
    fullText: `# Reinforcement Learning Algorithms

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's inspired by behavioral psychology and how humans and animals learn through trial and error.

## Core Concepts

### Agent and Environment
- Agent: The learner or decision maker
- Environment: The world the agent interacts with
- State: Current situation of the agent
- Action: What the agent can do
- Reward: Feedback from the environment

### Markov Decision Process (MDP)
Mathematical framework for RL:
- States (S)
- Actions (A)
- Transition probabilities (P)
- Rewards (R)
- Discount factor (γ)

## Key Algorithms

### Q-Learning
Model-free algorithm that learns the value of actions:
- Q-table representation
- Exploration vs exploitation
- Bellman equation
- Temporal difference learning

### Deep Q-Network (DQN)
Combines Q-learning with deep neural networks:
- Function approximation
- Experience replay
- Target network
- Stability improvements

### Policy Gradient Methods
Directly optimize the policy:
- REINFORCE algorithm
- Actor-Critic methods
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)

## Advanced Techniques

### Multi-Agent Reinforcement Learning
Multiple agents learning simultaneously:
- Cooperative learning
- Competitive environments
- Nash equilibrium
- Communication protocols

### Hierarchical Reinforcement Learning
Learning at multiple levels:
- Options framework
- Goal-conditioned RL
- Feudal networks
- Temporal abstractions

## Applications

### Game Playing
- Chess and Go
- Video games
- Strategic planning
- Multi-player environments

### Robotics
- Robot navigation
- Manipulation tasks
- Autonomous systems
- Human-robot interaction

### Finance
- Algorithmic trading
- Portfolio optimization
- Risk management
- Market making

### Healthcare
- Treatment optimization
- Drug discovery
- Personalized medicine
- Resource allocation

Reinforcement learning continues to show promise in solving complex sequential decision-making problems across various domains.`
  }
];

export const SearchProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [searchState, setSearchStateInternal] = useState<SearchState>({
    query: '',
    algorithm: 'hybrid',
    dataset: 'antique',
    resultCount: 10,
    useIndexing: true,
    useVectorStore: false,
    results: [],
    searchTime: 0,
    totalResults: 0,
  });

  const setSearchState = (newState: Partial<SearchState>) => {
    setSearchStateInternal(prev => ({ ...prev, ...newState }));
  };

  const performSearch = async (query: string, algorithm: SearchAlgorithm, dataset: Dataset, resultCount: number, useIndexing: boolean, useVectorStore: boolean) => {
    const startTime = Date.now();
    
    try {
      const searchRequest: SearchRequest = {
        query,
        algorithm,
        dataset,
        resultCount,
        useIndexing,
        useVectorStore
      };

      const searchResponse: SearchResponse = await searchDocuments(searchRequest);
      
      setSearchState({
        query,
        algorithm,
        dataset,
        resultCount,
        useIndexing,
        useVectorStore,
        results: searchResponse.results,
        searchTime: Date.now() - startTime,
        totalResults: searchResponse.totalResults,
      });
    } catch (error) {
      console.error('Search failed:', error);
      // Fallback to mock data for development
      const filteredResults = mockDocuments
        .filter(doc => 
          doc.title.toLowerCase().includes(query.toLowerCase()) ||
          doc.snippet.toLowerCase().includes(query.toLowerCase())
        )
        .sort((a, b) => b.score - a.score)
        .slice(0, resultCount);
      
      setSearchState({
        query,
        algorithm,
        dataset,
        resultCount,
        useIndexing,
        useVectorStore,
        results: filteredResults,
        searchTime: Date.now() - startTime,
        totalResults: filteredResults.length,
      });
    }
  };

  return (
    <SearchContext.Provider value={{ searchState, setSearchState, performSearch }}>
      {children}
    </SearchContext.Provider>
  );
};

export const getDocumentById = (id: string) => {
  return mockDocuments.find(doc => doc.id === id);
};