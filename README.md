# Sentiment Analysis with Mixed Sentiment Detection

This project performs sentiment analysis on a text dataset using SpaCy for lemmatization and TextBlob for sentiment scoring. It also detects mixed sentiment within sentences based on certain conjunctions, adversative transitions, and implicit patterns.

## Features
- Detects positive, neutral, negative, and mixed sentiments.
- Refined sentiment labels based on granular thresholds.
- Combines multi-line phrases into single sentences.
- Processes text from `.txt` files and saves the output in a CSV format.

## Getting Started

### Prerequisites

Ensure that you have the following Python libraries installed:

- `spacy==3.5.0`
- `textblob==0.17.1`
- `pandas==1.5.3`
- SpaCy's English language model `en_core_web_sm`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-project.git
   cd sentiment-analysis-project
