import spacy
from textblob import TextBlob
import re
import pandas as pd

# Load SpaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# List of conjunctions indicating contrast or mixed sentiment
CONJUNCTIONS = [
    "but", "although", "though", "even though", "despite", "in spite of", "yet",
    "however", "while", "whereas", "nevertheless", "nonetheless", "still",
    "on the other hand", "on the contrary", "conversely", "even if",
    "despite the fact that", "in contrast", "regardless", "notwithstanding",
    "instead", "in any case", "regardless of", "even so", "albeit", "except",
    "aside from", "alternatively", "contrary to", "other than", "except that",
    "otherwise", "in any event", "notwithstanding the fact", "by contrast",
    "in opposition", "regardless of the fact", "but still", "even though",
    "nonetheless", "irrespective of", "even with", "regardless that",
    "in defiance of", "without regard to", "though it may be", "in the face of",
    "for all that", "otherwise", "with all that", "contrary to expectations",
    "all the same", "still and all", "though it seems", "when in fact",
    "on the flip side", "in reverse", "even so", "contrastingly", "opposingly",
    "yet even", "all things considered", "be that as it may", "even if it seems",
    "at odds with", "paradoxically", "but then", "though it was",
    "though it is", "but nevertheless", "still, however", "aside from the fact",
    "differing from", "diverging from", "notwithstanding that", "alternatively",
    "despite appearances", "albeit in part", "but even", "even with that",
    "in other respects", "nevertheless still", "only that", "unlike",
    "contrary to this", "yet still", "contrarily", "varying from",
    "in direct opposition", "though contrary", "but in contrast", "with a twist",
    "differing opinions", "even against", "not the same", "split on",
    "but at the same time", "if it weren't for", "opposing views"
]

# List of adversative transitions indicating contrasting ideas
ADVERSATIVE_TRANSITIONS = [
    "however", "on the other hand", "conversely", "in contrast", "nevertheless",
    "nonetheless", "but", "yet", "despite that", "in spite of that",
    "still", "whereas", "although", "though", "even so",
    "at the same time", "regardless", "albeit", "notwithstanding",
    "despite", "contrarily", "while", "although", "even though",
    "in any case", "for all that", "but then", "be that as it may",
    "rather", "alternatively", "otherwise", "with that said",
    "despite this", "in light of this", "but still", "despite the fact",
    "in opposition", "rather than", "though it may be",
    "in the face of", "even with", "on the flip side",
    "on the contrary", "as opposed to", "differently",
    "instead", "on the contrary", "notwithstanding that",
    "in other respects", "even if", "though it seems",
    "in a different vein", "contrastingly", "opposingly",
    "contrary to", "though", "while", "rather than",
    "yet even", "even with that", "even still",
    "with all that", "at odds with", "to the contrary",
    "in stark contrast", "but even", "in defiance of",
    "for all that", "not the same", "in view of that",
    "although it is true", "despite the contrary",
    "not only that", "be that as it may",
    "while it may be true", "even when", "though the contrary",
    "against that", "as much as", "differing from",
    "albeit in part", "nonetheless still", "though it is said",
    "that said", "despite evidence to the contrary",
    "while it is true", "in the face of evidence",
    "whereas", "if anything", "albeit",
    "irrespective of", "but on the other hand",
    "with this in mind", "though it appears",
    "even when", "though not", "in a similar manner",
    "by contrast", "though there are", "even then",
    "at the same time", "otherwise stated", "not so",
    "contrary to expectation"
]

# List of implicit contrast patterns with more complex structures
IMPLICIT_PATTERNS = [
    "on the one hand... on the other hand",
    "in some respects... in others",
    "while... it is also true that",
    "despite the advantages... there are drawbacks",
    "although there are benefits... it comes with risks",
    "in contrast to... however",
    "while... yet",
    "despite this... that",
    "even though... it remains true that",
    "in light of... nevertheless",
    "while it may seem... in reality",
    "though... yet",
    "in comparison to... still",
    "whereas... at the same time",
    "although... nonetheless",
    "despite... there is still",
    "while... conversely",
    "although... on the flip side",
    "in some ways... in other ways",
    "though... there are limitations",
    "while... the opposite can also be said",
    "despite being... it can also be",
    "although... it does not mean",
    "in one sense... in another sense",
    "while there are positives... there are also negatives",
    "despite the fact that... still",
    "even if... it doesn’t change",
    "while this is true... that is also true",
    "though... there are challenges",
    "while... there exists a counterpoint",
    "although... it’s also important to note",
    "while some may argue... others believe",
    "though it is true... one must consider",
    "while some benefits exist... there are drawbacks",
    "although it offers... it may lack",
    "while this approach is effective... it has limitations",
    "though it appears... it might not be",
    "while many agree... some disagree",
    "even if it is beneficial... it can be problematic",
    "while there is support for... there is also criticism",
    "in certain cases... in others it might differ",
    "although there is merit... it’s not without flaws",
    "while it seems... the reality is different",
    "even though it is popular... it faces criticism",
    "while some see it as an opportunity... others see it as a risk",
    "though it is advantageous... it requires effort",
    "while it has strengths... it also has weaknesses",
    "despite improvements... there are still issues",
    "while it might help... it can also hurt",
    "although it seems simple... it’s complex",
    "despite being well-received... it has detractors",
    "although it offers insights... it lacks depth",
    "while it aims to assist... it can confuse",
    "even with its advantages... there are pitfalls",
    "although it has potential... results may vary",
    "in theory... in practice",
    "while it addresses... it overlooks",
    "despite its popularity... some remain skeptical",
    "while the goal is clear... the path is uncertain",
    "though it’s designed to be helpful... it may complicate",
    "although it is essential... it can be burdensome",
    "in principle... in practice",
    "while it can be effective... it requires commitment",
    "although it has benefits... it has consequences",
    "even though it is crucial... it may be ignored",
    "while there is enthusiasm... there are concerns",
    "although the proposal is appealing... it has flaws",
    "while the findings suggest... the implications are uncertain",
    "despite the challenges... opportunities exist",
    "while some thrive... others struggle",
    "although it promises... it may not deliver",
    "in certain situations... the opposite may occur",
    "while the process is straightforward... the outcome is unpredictable",
    "although it aims to unify... it can divide",
    "while it opens doors... it may also close them",
    "even if it enhances... it could detract",
    "despite being encouraged... some remain hesitant",
    "while it addresses one issue... it may create another",
    "though it can be beneficial... it requires caution",
    "while it seems clear... there are nuances",
    "although it offers clarity... it can also confuse",
    "while it provides support... it can also constrain",
    "though the intent is good... results may vary",
    "while it fosters collaboration... it can lead to conflict",
    "even if it is innovative... it can be disruptive",
    "while some endorse it... others criticize it",
    "although it promises efficiency... it may introduce complexity"
]

# Utility function to check if a sentence contains mixed sentiment
def detect_mixed_sentiment(sentence):
    """
    Detects if a sentence contains mixed sentiment using conjunctions, adversative transitions, or implicit patterns.
    Returns True if a mixed sentiment is detected, otherwise False.
    """
    cleaned_sentence = sentence.lower()

    # Check for conjunctions or adversative transitions
    if any(conj in cleaned_sentence for conj in CONJUNCTIONS):
        return True

    if any(trans in cleaned_sentence for trans in ADVERSATIVE_TRANSITIONS):
        return True

    # Check for implicit contrast patterns
    if any(pattern in cleaned_sentence for pattern in IMPLICIT_PATTERNS):
        return True

    return False

# Function to clean and process an utterance
def clean_utterance(utterance):
    utterance = utterance.lower()
    utterance = re.sub(r'http\S+|www\S+|https\S+', '', utterance, flags=re.MULTILINE)
    utterance = re.sub(r'\d+', '', utterance)
    utterance = re.sub(r'[^\w\s]', '', utterance)
    doc = nlp(utterance)
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

# Function to detect refined sentiment tone and return score and tone label
def detect_sentiment_tone(utterance):
    analysis = TextBlob(utterance)
    sentiment_score = analysis.sentiment.polarity

    # Refined thresholds for more granularity in sentiment labels
    if sentiment_score > 0.85:
        sentiment_label = "strong positive"
    elif 0.60 < sentiment_score <= 0.85:
        sentiment_label = "positive"
    elif 0.30 < sentiment_score <= 0.60:
        sentiment_label = "mild positive"
    elif 0.10 < sentiment_score <= 0.30:
        sentiment_label = "neutral positive"
    elif -0.10 <= sentiment_score <= 0.10:
        sentiment_label = "neutral"
    elif -0.30 <= sentiment_score < -0.10:
        sentiment_label = "neutral negative"
    elif -0.60 <= sentiment_score < -0.30:
        sentiment_label = "mild negative"
    elif -0.85 <= sentiment_score < -0.60:
        sentiment_label = "negative"
    else:
        sentiment_label = "strong negative"

    # Check for mixed sentiment
    if detect_mixed_sentiment(utterance):
        sentiment_label = "mixed sentiment"

    return sentiment_score, sentiment_label

# Function to combine phrases and sentences that are split across lines
def combine_phrases(lines):
    combined_lines = []
    temp_phrase = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(('.', '?', '!')):
            if temp_phrase:
                temp_phrase += " " + line
                combined_lines.append(temp_phrase.strip())
                temp_phrase = ""
            else:
                combined_lines.append(line.strip())
        else:
            temp_phrase += " " + line if temp_phrase else line
    if temp_phrase:
        combined_lines.append(temp_phrase.strip())
    return combined_lines

# Function to preprocess and extract dialogues from .txt file content
def preprocess_dataset(file_path):
    cleaned_conversation = []
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = file.read()

    lines = dataset.split('\n')
    combined_lines = combine_phrases(lines)

    for line in combined_lines:
        original_utterance = line
        cleaned_utterance = clean_utterance(line)

        # Run sentiment analysis
        sentiment_score, sentiment_label = detect_sentiment_tone(line)

        cleaned_conversation.append({
            'Original Sentence/Phrase': original_utterance,
            'Sentiment Score': sentiment_score,
            'Sentiment Tone': sentiment_label
        })

    return cleaned_conversation

# Path to your .txt file
file_path = '/your_file_path'

# Preprocess the dataset from the .txt file
processed_conversation = preprocess_dataset(file_path)

# Convert the processed data into a Pandas DataFrame
df = pd.DataFrame(processed_conversation)

# Save the DataFrame to a CSV file (optional)
csv_output_path = 'output_file.csv'
df.to_csv(csv_output_path, index=False)

# Display the DataFrame
print(df)
