"""
Twitter Emotion Classifier Dashboard

This module creates an interactive emotion classification system that analyzes text
and predicts emotions using a pre-trained DistilBERT model. The system includes:
- Text emotion classification with probability scores
- Interactive dashboard with Gradio
- Model performance visualization (confusion matrix, t-SNE)

The classifier uses DistilBERT embeddings combined with Logistic Regression
to classify text into 6 emotions: sadness, joy, love, anger, fear, surprise.

"""

# Import statements with explanations
from datasets import load_dataset  # Hugging Face datasets library for loading emotion dataset
from transformers import DistilBertTokenizer, DistilBertModel  # Pre-trained transformer models for text processing
import torch  # PyTorch library for tensor operations and neural networks
import numpy as np  # Numerical computing library for array operations
from sklearn.linear_model import LogisticRegression  # Machine learning classifier
from sklearn.metrics import classification_report, confusion_matrix  # Model evaluation metrics
import seaborn as sns  # Statistical data visualization library
import matplotlib.pyplot as plt  # Plotting library for creating charts and graphs
from sklearn.manifold import TSNE  # Dimensionality reduction technique for visualization
import gradio as gr  # Library for creating interactive web interfaces for ML models

# =============================================================================
# 1. DATASET LOADING
# =============================================================================
"""
Load the emotion dataset from Hugging Face.

The 'dair-ai/emotion' dataset contains English Twitter messages labeled with emotions.
It includes 6 emotion categories: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).

Dataset structure:
- train: 16,000 examples for training
- validation: 2,000 examples for validation
- test: 2,000 examples for testing
- Each example has 'text' (string) and 'label' (integer 0-5)
"""
dataset = load_dataset("dair-ai/emotion")

# =============================================================================
# 2. MODEL AND TOKENIZER LOADING
# =============================================================================
"""
Load the pre-trained DistilBERT model and tokenizer.

DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder Representations
from Transformers) that retains 97% of BERT's performance while being 60% smaller.

- Tokenizer: Converts text into tokens (subword units) that the model can understand
- Model: Pre-trained transformer that generates contextual embeddings for text
- "distilbert-base-uncased": The base uncased version (no capital letters distinction)
"""
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# =============================================================================
# 3. EMBEDDING EXTRACTION FUNCTION
# =============================================================================
def get_embedding(text):
    """
    Extract contextual embeddings from text using DistilBERT.

    This function converts input text into dense vector representations (embeddings)
    that capture semantic meaning. The embeddings can then be used for classification.

    Args:
        text (str): Input text to convert into embeddings

    Returns:
        numpy.ndarray: A 768-dimensional embedding vector representing the text

    Process:
    1. Tokenize text and convert to PyTorch tensors
    2. Pass through DistilBERT model to get contextual representations
    3. Extract the [CLS] token embedding (first token) as text representation
    4. Convert to numpy array for compatibility with scikit-learn
    """
    # Tokenize the input text with specific parameters:
    inputs = tokenizer(
        text,
        return_tensors="pt",      # Return PyTorch tensors (alternatives: "tf" for TensorFlow, "np" for numpy)
                                  # "pt" is required because our DistilBERT model expects PyTorch tensors
        truncation=True,          # Truncate text longer than max_length to avoid memory issues
        padding=True,             # Pad shorter sequences to max_length for batch processing
        max_length=64             # Maximum sequence length (shorter than BERT's 512 for speed)
    )

    # Disable gradient computation for inference (saves memory and computation)
    with torch.no_grad():
        outputs = model(**inputs)  # Pass tokenized inputs through DistilBERT model

    # Extract embeddings from the [CLS] token (first token, index 0)
    # last_hidden_state shape: [batch_size, sequence_length, hidden_size]
    # [:, 0, :] selects the first token ([CLS]) from all batches
    # .numpy() converts PyTorch tensor to numpy array for scikit-learn compatibility
    return outputs.last_hidden_state[:, 0, :].numpy()

# =============================================================================
# 4. DATA PREPARATION - EMBEDDING GENERATION
# =============================================================================
"""
Generate embeddings for training and testing data.

We use a subset of the full dataset for faster training and demonstration:
- Training: 2,000 examples (out of 16,000 available)
- Testing: 500 examples (out of 2,000 available)

This process converts text into numerical vectors that machine learning
algorithms can work with. Each text becomes a 768-dimensional vector.
"""
print("Generating training embeddings...")
X_train = np.vstack([get_embedding(text) for text in dataset["train"]["text"][:2000]])
y_train = np.array(dataset["train"]["label"][:2000])

print("Generating testing embeddings...")
X_test = np.vstack([get_embedding(text) for text in dataset["test"]["text"][:500]])
y_test = np.array(dataset["test"]["label"][:500])

print(f"Training data shape: {X_train.shape}")  # Should be (2000, 768)
print(f"Testing data shape: {X_test.shape}")    # Should be (500, 768)

# =============================================================================
# 5. CLASSIFIER TRAINING
# =============================================================================
"""
Train a Logistic Regression classifier on the DistilBERT embeddings.

Logistic Regression is chosen because:
- It's fast and efficient for high-dimensional data (768 features)
- Works well with pre-trained embeddings
- Provides probability scores for each class
- Less prone to overfitting compared to complex models

max_iter=200: Maximum number of iterations for the solver to converge
"""
print("Training classifier...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Calculate and display training accuracy
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Testing accuracy: {test_accuracy:.3f}")

# =============================================================================
# 6. LABEL MAPPING AND VISUALIZATION SETUP
# =============================================================================
"""
Define emotion labels and their corresponding emojis for better user experience.

Label mapping (from dataset):
0: sadness - 😢
1: joy - 😊
2: love - ❤️
3: anger - 😡
4: fear - 😨
5: surprise - 😲

These emojis make the predictions more intuitive and user-friendly.
"""
label_names = dataset["train"].features["label"].names  # ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emoji_map = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

print(f"Emotion classes: {label_names}")

# =============================================================================
# 7. PREDICTION AND VISUALIZATION FUNCTIONS
# =============================================================================
def predict_emotion_with_chart(text):
    """
    Predict emotion from text and create a probability chart.

    This is the main prediction function used by the Gradio interface.
    It combines emotion prediction with visualization.

    Args:
        text (str): Input text to classify

    Returns:
        tuple: (matplotlib.figure.Figure, str)
            - Figure: Bar chart showing probability for each emotion
            - String: Predicted emotion with emoji (e.g., "joy 😊")

    Process:
    1. Convert text to embedding using DistilBERT
    2. Get probability scores for all 6 emotions using trained classifier
    3. Find the emotion with highest probability
    4. Create visualization showing all probabilities
    5. Return both chart and prediction text
    """
    # Get embedding for the input text
    emb = get_embedding(text)

    # Get probability scores for all emotions (shape: [6,])
    probs = clf.predict_proba(emb)[0]

    # Find the emotion with highest probability
    pred_label = np.argmax(probs)
    label_name = label_names[pred_label]
    emoji = emoji_map.get(label_name, "")

    # Create probability bar chart for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(label_names, probs, color='skyblue', alpha=0.7)

    # Highlight the predicted emotion
    bars[pred_label].set_color('orange')

    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f"Emotion Classification Results\nPrediction: {label_name} {emoji} ({probs[pred_label]:.2f})", fontsize=14)
    ax.set_ylim(0, 1)  # Probabilities range from 0 to 1

    # Add probability values on top of bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, f"{label_name} {emoji}"

def show_confusion_matrix():
    """
    Generate and display a confusion matrix for model performance evaluation.

    A confusion matrix shows how well the model performs for each emotion class.
    - Diagonal elements: Correct predictions
    - Off-diagonal elements: Misclassifications

    Returns:
        matplotlib.figure.Figure: Heatmap visualization of confusion matrix

    Reading the matrix:
    - Rows represent true/actual emotions
    - Columns represent predicted emotions
    - Higher values on diagonal = better performance
    - High off-diagonal values indicate common misclassifications
    """
    # Generate predictions for test set
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,              # Show numbers in cells
                fmt="d",                 # Format as integers
                cmap="Blues",            # Color scheme
                xticklabels=label_names, # Emotion names as x-axis labels
                yticklabels=label_names, # Emotion names as y-axis labels
                ax=ax)

    ax.set_xlabel("Predicted Emotion", fontsize=12)
    ax.set_ylabel("True Emotion", fontsize=12)
    ax.set_title("Confusion Matrix - Model Performance by Emotion", fontsize=14)

    # Calculate and display overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)  # Sum of diagonal / total
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.3f}',
            transform=ax.transAxes, ha='center', fontsize=12)

    plt.tight_layout()
    return fig

def show_tsne():
    """
    Create t-SNE visualization of text embeddings.

    t-SNE (t-Distributed Stochastic Neighbor Embedding) reduces the 768-dimensional
    DistilBERT embeddings to 2D for visualization. This helps us understand:
    - How well emotions cluster in the embedding space
    - Whether similar emotions are close together
    - If the model can distinguish between different emotions

    Returns:
        matplotlib.figure.Figure: 2D scatter plot of embeddings colored by emotion

    Interpretation:
    - Points close together have similar embeddings
    - Well-separated clusters indicate the model can distinguish emotions
    - Overlapping clusters suggest the model may confuse those emotions
    """
    print("Computing t-SNE embeddings (this may take a moment)...")

    # Reduce 768D embeddings to 2D using t-SNE
    tsne = TSNE(
        n_components=2,    # Reduce to 2 dimensions for plotting
        random_state=42,   # For reproducible results
        perplexity=30,     # Balance between local and global structure
        n_iter=1000        # Number of iterations for optimization
    )
    X_embedded = tsne.fit_transform(X_test)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each emotion class with different colors
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for i, (label, color) in enumerate(zip(label_names, colors)):
        idx = y_test == i  # Boolean mask for current emotion
        ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                  label=f"{label} ({emoji_map[label]})",
                  alpha=0.6,
                  c=color,
                  s=50)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("t-SNE Visualization of DistilBERT Embeddings\nEach point represents a text sample", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# =============================================================================
# 8. GRADIO INTERFACE SETUP
# =============================================================================
"""
Create an interactive web interface using Gradio.

Gradio automatically creates a web interface for machine learning models.
The interface includes three tabs:
1. Predict Emotion: Main classifier interface for user input
2. Confusion Matrix: Model performance evaluation
3. t-SNE Visualization: Embedding space visualization

gr.Blocks: Allows custom layout with tabs and components
gr.Tab: Creates tabbed interface for organizing different functionalities
"""

# Create the main interface using Gradio Blocks (allows custom layouts)
with gr.Blocks(title="Emotion Classifier Dashboard", theme=gr.themes.Soft()) as demo:
    # Main title
    gr.Markdown("""
    # 🎭 Emotion Classifier Dashboard

    This interactive tool classifies text emotions using DistilBERT embeddings and Logistic Regression.
    Enter any text to see predicted emotions with confidence scores!
    """)

    # Tab 1: Main prediction interface
    with gr.Tab("🔮 Predict Emotion"):
        gr.Markdown("### Enter text below to classify its emotion")

        # Input components
        inp = gr.Textbox(
            label="Enter text to analyze",
            placeholder="Example: I'm so happy today! or I feel really sad...",
            lines=3
        )

        # Output components
        out_plot = gr.Plot(label="Emotion Probabilities")
        out_text = gr.Textbox(label="Predicted Emotion", interactive=False)

        # Action button
        btn = gr.Button("🎯 Classify Emotion", variant="primary")
        btn.click(
            fn=predict_emotion_with_chart,
            inputs=inp,
            outputs=[out_plot, out_text]
        )

        # Example inputs for user testing
        gr.Examples(
            examples=[
                ["I love spending time with my family!"],
                ["This makes me so angry and frustrated."],
                ["I'm scared about the upcoming exam."],
                ["What a wonderful surprise!"],
                ["I feel so sad and lonely today."]
            ],
            inputs=inp
        )

    # Tab 2: Model performance evaluation
    with gr.Tab("📊 Model Performance"):
        gr.Markdown("### Confusion Matrix - See how well the model performs for each emotion")
        confusion_btn = gr.Button("📈 Generate Confusion Matrix", variant="secondary")
        confusion_plot = gr.Plot(label="Confusion Matrix")
        confusion_btn.click(fn=show_confusion_matrix, outputs=confusion_plot)

    # Tab 3: Embedding visualization
    with gr.Tab("🔍 Embedding Visualization"):
        gr.Markdown("""
        ### t-SNE Visualization of Text Embeddings
        This shows how the model represents different emotions in a 2D space.
        Similar emotions should cluster together.
        """)
        tsne_btn = gr.Button("🎨 Generate t-SNE Plot", variant="secondary")
        tsne_plot = gr.Plot(label="t-SNE Visualization")
        tsne_btn.click(fn=show_tsne, outputs=tsne_plot)

# Launch the interface
print("Launching Gradio interface...")
print("The dashboard will be available at: http://localhost:7860")
demo.launch(
    share=False,          # Set to True to create a public shareable link
    inbrowser=True,       # Automatically open in browser
    server_name="0.0.0.0" # Allow access from other devices on network
)
