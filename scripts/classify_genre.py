import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score
import os
import warnings
import pickle
from wordcloud import WordCloud
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# Ensure output directory exists
os.makedirs('../data/stats2', exist_ok=True)
os.makedirs('../models', exist_ok=True)  # Create directory for saving models

# Function to load and clean data
def load_and_clean_data(file_path):
    print("Loading and cleaning data...")
    # Load data
    df = pd.read_csv(file_path)
    
    # Check for missing values
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Drop rows with missing values in important columns
    df = df.dropna(subset=['description', 'genres', 'rating'])
    
    # Convert text columns to lowercase
    df['description'] = df['description'].str.lower()
    df['genres'] = df['genres'].str.lower()
    
    # Process genres
    df['genres_list'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])
    
    # Remove duplicates
    duplicates_before = df.shape[0]
    df = df.drop_duplicates(subset=['title', 'description'])
    duplicates_removed = duplicates_before - df.shape[0]
    
    print(f"Rows removed due to missing values: {duplicates_before - df.shape[0]}")
    print(f"Rows removed due to duplicates: {duplicates_removed}")
    print(f"Cleaned data shape: {df.shape}")
    
    return df

# Function to visualize genre distribution
def visualize_genre_distribution(df):
    print("Visualizing genre distribution...")
    
    # Explode the genres_list column to get one row per genre
    genre_df = df.explode('genres_list')
    
    # Count occurrences of each genre
    genre_counts = genre_df['genres_list'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    
    # Get top 20 genres
    top_genres = genre_counts.head(20)
    
    # Create barplot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='count', y='genre', data=top_genres, palette='viridis')
    
    # Add count labels to the bars
    for i, count in enumerate(top_genres['count']):
        ax.text(count + 5, i, str(count), va='center')
    
    plt.title('Top 20 Most Common Movie Genres', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.tight_layout()
    plt.savefig('../data/stats2/genre_distribution.png')
    plt.close()
    
    # Create a pie chart for top 10 genres
    plt.figure(figsize=(10, 10))
    top10 = genre_counts.head(10)
    plt.pie(top10['count'], labels=top10['genre'], autopct='%1.1f%%', 
            shadow=True, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, 10)))
    plt.axis('equal')
    plt.title('Distribution of Top 10 Movie Genres', fontsize=16)
    plt.tight_layout()
    plt.savefig('../data/stats2/genre_pie_chart.png')
    plt.close()

# Function to visualize rating distribution
def visualize_rating_distribution(df):
    print("Visualizing rating distribution...")
    
    # Count occurrences of each rating
    rating_counts = df['rating'].value_counts().reset_index()
    rating_counts.columns = ['rating', 'count']
    
    # Create barplot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='rating', y='count', data=rating_counts, palette='rocket')
    
    # Add count labels to the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', fontsize=10)
    
    plt.title('Distribution of Movie Ratings', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../data/stats2/rating_distribution.png')
    plt.close()
    
    # Create a heatmap of ratings vs. top genres
    genre_df = df.explode('genres_list')
    top_genres = genre_df['genres_list'].value_counts().head(10).index
    
    # Filter for top genres
    genre_df = genre_df[genre_df['genres_list'].isin(top_genres)]
    
    # Create a crosstab
    cross_tab = pd.crosstab(genre_df['genres_list'], genre_df['rating'])
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Relationship Between Top Genres and Ratings', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.tight_layout()
    plt.savefig('../data/stats2/genre_rating_heatmap.png')
    plt.close()

# Function to visualize description text data
def visualize_description_text(df):
    print("Visualizing description text data...")
    
    # Combine all descriptions
    all_descriptions = ' '.join(df['description'])
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=200, contour_width=3, contour_color='steelblue',
                         colormap='viridis').generate(all_descriptions)
    
    # Display word cloud
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../data/stats2/description_wordcloud.png')
    plt.close()
    
    # Get top words using CountVectorizer
    cv = CountVectorizer(max_features=20, stop_words='english')
    cv_matrix = cv.fit_transform(df['description'])
    
    # Sum up word counts across all documents
    word_freq = np.array(cv_matrix.sum(axis=0)).flatten()
    word_freq = [(word, word_freq[idx]) for word, idx in cv.vocabulary_.items()]
    word_freq.sort(key=lambda x: x[1], reverse=True)
    
    # Create DataFrame for visualization
    word_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
    
    # Create barplot of top words
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='word', y='frequency', data=word_df, palette='viridis')
    plt.title('Top 20 Most Frequent Words in Movie Descriptions', fontsize=16)
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add frequency labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../data/stats2/top_words_barchart.png')
    plt.close()

# Function to visualize TFIDF and feature extraction
def visualize_feature_extraction(tfidf, X_train_processed, le):
    print("Visualizing feature extraction...")
    
    # Get feature names from TF-IDF
    feature_names = tfidf.get_feature_names_out()
    
    # Get top 20 TF-IDF features by weight across the corpus
    tfidf_sum = np.sum(X_train_processed[:, :len(feature_names)], axis=0)
    top_indices = tfidf_sum.argsort()[-20:][::-1]
    top_values = tfidf_sum[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create DataFrame for visualization
    tfidf_df = pd.DataFrame({'feature': top_features, 'importance': top_values})
    
    # Create barplot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='feature', y='importance', data=tfidf_df, palette='cool')
    plt.title('Top 20 TF-IDF Features (Words) by Importance', fontsize=16)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance (Sum of TF-IDF Scores)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../data/stats2/tfidf_feature_importance.png')
    plt.close()
    
    # Visualize encoding for ratings
    if hasattr(le, 'classes_'):
        # Create a mapping of original values to encoded values
        rating_mapping = {original: encoded for original, encoded in zip(le.classes_, range(len(le.classes_)))}
        
        # Create DataFrame for visualization
        rating_df = pd.DataFrame(list(rating_mapping.items()), columns=['Rating', 'Encoded Value'])
        
        # Create a visualization of the encoding
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Rating', y='Encoded Value', data=rating_df, palette='rocket')
        
        # Add labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
        
        plt.title('Label Encoding for Movie Ratings', fontsize=16)
        plt.xlabel('Original Rating', fontsize=12)
        plt.ylabel('Encoded Value', fontsize=12)
        plt.tight_layout()
        plt.savefig('../data/stats2/rating_encoding.png')
        plt.close()

# Function to visualize SVD reduction and feature importance
def visualize_dimension_reduction(svd, X_reduced):
    print("Visualizing dimension reduction...")
    
    # Plot explained variance by components
    explained_var = svd.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    components = range(1, len(explained_var) + 1)
    
    # Bar chart for individual explained variance
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(components[:20], explained_var[:20], alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, 20)))
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Explained Variance Ratio')
    ax1.set_title('Explained Variance by Component (Top 20)')
    
    # Line chart for cumulative explained variance
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(components, cum_explained_var, marker='o', linestyle='-', color='royalblue')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Explained Variance')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Explained Variance')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../data/stats2/svd_explained_variance.png')
    plt.close()
    
    # Visualize the first two components as a scatter plot with density
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot
    sns.kdeplot(x=X_reduced[:, 0], y=X_reduced[:, 1], 
               cmap="viridis", fill=True, thresh=0.05)
    
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
               s=20, c=X_reduced[:, 0], cmap="viridis", alpha=0.2)
    
    plt.colorbar(label='Component 1 Value')
    plt.title('Data Distribution after Dimension Reduction (First 2 Components)', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../data/stats2/svd_data_distribution.png')
    plt.close()

# Function to extract features
def extract_features(df):
    print("Extracting features...")
    # Create feature dataframe
    X = df[['description', 'rating']]
    
    # Use MultiLabelBinarizer to convert genres list to binary matrix
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres_list'])
    
    # Get the genre names
    genre_names = mlb.classes_
    print(f"Total number of unique genres: {len(genre_names)}")
    print(f"Some example genres: {genre_names[:10]}")
    
    # Get distribution of genres
    genre_counts = np.sum(y, axis=0)
    genre_dist = dict(zip(genre_names, genre_counts))
    
    # Print the top 10 most common genres
    print("\nTop 10 most common genres:")
    for genre, count in sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{genre}: {count}")
    
    # Visualize the extracted features
    visualize_genre_distribution(df)
    visualize_rating_distribution(df)
    visualize_description_text(df)
    
    return X, y, mlb

# Function to process features
def process_features(X_train, X_test):
    print("Processing features...")
    # TF-IDF for description with n-grams for better text representation
    tfidf = TfidfVectorizer(
        max_features=5000, 
        stop_words='english',
        ngram_range=(1, 2),  # Include unigrams and bigrams
        min_df=3,  # Minimum document frequency
        max_df=0.9  # Maximum document frequency
    )
    X_train_description = tfidf.fit_transform(X_train['description'])
    X_test_description = tfidf.transform(X_test['description'])
    
    # Label Encoding for ratings
    le = LabelEncoder()
    X_train_rating = le.fit_transform(X_train['rating']).reshape(-1, 1)
    X_test_rating = le.transform(X_test['rating']).reshape(-1, 1)
    
    # Convert sparse matrix to dense for description
    X_train_description_dense = X_train_description.toarray()
    X_test_description_dense = X_test_description.toarray()
    
    # Combine features
    X_train_processed = np.hstack((X_train_description_dense, X_train_rating))
    X_test_processed = np.hstack((X_test_description_dense, X_test_rating))
    
    # Visualize feature extraction
    visualize_feature_extraction(tfidf, X_train_processed, le)
    
    return X_train_processed, X_test_processed, tfidf, le

# Function to reduce dimensions
def reduce_dimensions(X_train, X_test, n_components=100):
    print("Reducing dimensions...")
    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    
    # Apply MinMaxScaler to ensure non-negative values for Logistic Regression
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    
    # Explained variance
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance with {n_components} components: {explained_variance:.2f}")
    
    # Visualize the dimension reduction
    visualize_dimension_reduction(svd, X_train_reduced)
    
    return X_train_scaled, X_test_scaled, svd, scaler

# Function to visualize data
def visualize_data(X_reduced, y, mlb, title="Dimensionality Reduction Visualization"):
    print("Visualizing data...")
    # Create a 2D plot using the first two components
    plt.figure(figsize=(12, 8))
    
    # Since we have multi-label data, we'll color points based on the most frequent genre
    # For each sample, find the index of the genre with value 1
    genre_indices = [np.where(sample == 1)[0] for sample in y]
    
    # Choose the first genre for each sample (if available)
    chosen_genres = [indices[0] if len(indices) > 0 else -1 for indices in genre_indices]
    
    # Get unique genre indices
    unique_genres = np.unique(chosen_genres)
    
    # Define a colormap
    cmap = plt.cm.get_cmap('viridis', len(unique_genres))
    
    # Plot each genre
    for i, genre_idx in enumerate(unique_genres):
        if genre_idx == -1:
            continue  # Skip samples with no genres
        
        mask = np.array(chosen_genres) == genre_idx
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   color=cmap(i), label=mlb.classes_[genre_idx] if genre_idx >= 0 else 'Unknown', 
                   alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../data/stats2/data_visualization.png')
    plt.close()

# Function to compute partial accuracy (at least one genre correct)
def accuracy(y_true, y_pred):
    """
    Calculate partial accuracy - percentage of samples that have at least one genre predicted correctly
    
    Args:
        y_true: Matrix of true genre labels (one-hot encoded)
        y_pred: Matrix of predicted genre labels (one-hot encoded)
    
    Returns:
        Partial accuracy score (0.0 to 1.0)
    """
    correct_count = 0
    total_samples = y_true.shape[0]
    
    for i in range(total_samples):
        # Get true and predicted genres
        true_genres = set(np.where(y_true[i] == 1)[0])
        pred_genres = set(np.where(y_pred[i] == 1)[0])
        
        # Check if at least one genre is correctly predicted
        if len(true_genres.intersection(pred_genres)) > 0:
            correct_count += 1
    
    return correct_count / total_samples if total_samples > 0 else 0

# Function to calculate top-K accuracy in top-N predictions
def top_k_in_top_n_accuracy(y_true, y_pred_proba, mlb, k=3, n=5):
    """
    Calculate the percentage of samples where at least k out of the top n predicted genres are correct
    
    Args:
        y_true: Matrix of true genre labels (one-hot encoded)
        y_pred_proba: Matrix of prediction probabilities for each genre
        mlb: MultiLabelBinarizer used for encoding
        k: Minimum number of correct genres needed in top-n (default: 3)
        n: Number of top predicted genres to consider (default: 5)
    
    Returns:
        Percentage of samples meeting the criteria
    """
    correct_count = 0
    total_samples = y_true.shape[0]
    
    for i in range(total_samples):
        # Get the actual genres for this sample
        true_genres = set(np.where(y_true[i] == 1)[0])
        
        if len(true_genres) == 0:
            # If there are no true genres, skip this sample
            total_samples -= 1
            continue
        
        # Get top-N genres with highest probabilities
        top_n_indices = np.argsort(y_pred_proba[i])[::-1][:n]
        top_n_genres = set(top_n_indices)
        
        # Count correct genres in top-N
        correct_genres = len(true_genres.intersection(top_n_genres))
        
        # If at least K genres are correct, increment counter
        if correct_genres >= k:
            correct_count += 1
    
    return correct_count / total_samples if total_samples > 0 else 0

# Function to predict with adjusted threshold
def predict_with_adjusted_threshold(model, X, threshold=0.15):
    """
    Predict with lowered threshold focusing on getting at least 1 out of 3 genres correct
    """
    # Get probabilities for each genre
    probs = []
    for estimator in model.estimators_:
        if hasattr(estimator, 'predict_proba'):
            probs.append(estimator.predict_proba(X)[:, 1])
    
    probs = np.array(probs).T  # Shape: [n_samples, n_labels]
    
    # Create initial prediction matrix with threshold
    y_pred = (probs >= threshold).astype(int)
    
    # For each sample, ensure exactly 3 genres are predicted
    for i in range(y_pred.shape[0]):
        # Count predicted genres
        n_predicted = np.sum(y_pred[i])
        
        if n_predicted == 3:
            # Already at 3 genres - keep as is
            continue
        elif n_predicted < 3:
            # Need to add more genres
            # Find indices of zeros (unpredicted genres)
            zero_indices = np.where(y_pred[i] == 0)[0]
            # Sort by probability
            sorted_indices = sorted([(idx, probs[i, idx]) for idx in zero_indices], 
                                   key=lambda x: x[1], reverse=True)
            # Add top genres to reach 3
            to_add = 3 - n_predicted
            for j in range(min(to_add, len(sorted_indices))):
                y_pred[i, sorted_indices[j][0]] = 1
        else:
            # Too many genres predicted - keep only top 3
            # Get all predicted genre indices
            pred_indices = np.where(y_pred[i] == 1)[0]
            # Sort by probability
            sorted_indices = sorted([(idx, probs[i, idx]) for idx in pred_indices], 
                                   key=lambda x: x[1], reverse=True)
            # Reset predictions
            y_pred[i] = 0
            # Keep top 3
            for j in range(min(3, len(sorted_indices))):
                y_pred[i, sorted_indices[j][0]] = 1
    
    return y_pred, probs

# Function to train models
def train_models(X_train, y_train, X_val, y_val, mlb):
    print("Training models...")
    
    # Multi-label models with Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = MultiOutputClassifier(LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'))
    lr_model.fit(X_train, y_train)
    
    # Predict with adjusted threshold
    lr_val_pred, lr_val_probs = predict_with_adjusted_threshold(lr_model, X_val, threshold=0.15)
    
    # Evaluate metrics
    lr_val_jaccard = jaccard_score(y_val, lr_val_pred, average='samples')
    lr_val_hamming = hamming_loss(y_val, lr_val_pred)
    lr_val_partial_acc = accuracy(y_val, lr_val_pred)
    
    print(f"Logistic Regression Validation Jaccard Score: {lr_val_jaccard:.4f}")
    print(f"Logistic Regression Validation Hamming Loss: {lr_val_hamming:.4f}")
    print(f"Logistic Regression Validation Partial Accuracy: {lr_val_partial_acc:.4f}")
    
    # Try with C=2.0 for Logistic Regression
    lr_c2_model = MultiOutputClassifier(LogisticRegression(max_iter=1000, C=2.0, solver='liblinear'))
    lr_c2_model.fit(X_train, y_train)
    lr_c2_val_pred, lr_c2_val_probs = predict_with_adjusted_threshold(lr_c2_model, X_val, threshold=0.12)
    lr_c2_val_jaccard = jaccard_score(y_val, lr_c2_val_pred, average='samples')
    lr_c2_val_hamming = hamming_loss(y_val, lr_c2_val_pred)
    lr_c2_val_partial_acc = accuracy(y_val, lr_c2_val_pred)
    
    print(f"LR (C=2.0) Validation Jaccard Score: {lr_c2_val_jaccard:.4f}")
    print(f"LR (C=2.0) Validation Hamming Loss: {lr_c2_val_hamming:.4f}")
    print(f"LR (C=2.0) Validation Partial Accuracy: {lr_c2_val_partial_acc:.4f}")
    
    # Multi-label models with SVM
    print("Training SVM model...")
    svm_model = MultiOutputClassifier(SVC(kernel='linear', probability=True, C=1.0))
    svm_model.fit(X_train, y_train)
    
    # Predict with adjusted threshold
    svm_val_pred, svm_val_probs = predict_with_adjusted_threshold(svm_model, X_val, threshold=0.15)
    
    # Evaluate metrics
    svm_val_jaccard = jaccard_score(y_val, svm_val_pred, average='samples')
    svm_val_hamming = hamming_loss(y_val, svm_val_pred)
    svm_val_partial_acc = accuracy(y_val, svm_val_pred)
    
    print(f"SVM Validation Jaccard Score: {svm_val_jaccard:.4f}")
    print(f"SVM Validation Hamming Loss: {svm_val_hamming:.4f}")
    print(f"SVM Validation Partial Accuracy: {svm_val_partial_acc:.4f}")
    
    return {
        'logistic_regression': {
            'model': lr_model,
            'jaccard': lr_val_jaccard,
            'hamming_loss': lr_val_hamming,
            'partial_accuracy': lr_val_partial_acc,
            'predictions': lr_val_pred,
            'probabilities': lr_val_probs
        },
        'logistic_regression_c2': {
            'model': lr_c2_model,
            'jaccard': lr_c2_val_jaccard,
            'hamming_loss': lr_c2_val_hamming,
            'partial_accuracy': lr_c2_val_partial_acc,
            'predictions': lr_c2_val_pred,
            'probabilities': lr_c2_val_probs
        },
        'svm': {
            'model': svm_model,
            'jaccard': svm_val_jaccard,
            'hamming_loss': svm_val_hamming,
            'partial_accuracy': svm_val_partial_acc,
            'predictions': svm_val_pred,
            'probabilities': svm_val_probs
        }
    }

# Function to evaluate models
def evaluate_models(models, X_test, y_test, mlb):
    print("Evaluating models...")
    results = {}
    
    # List of genres to remove from visualization (if they exist)
    genres_to_exclude = ['western', 'sports & fitness', 'short', 'lgbtq+']
    
    for name, model_info in models.items():
        model = model_info['model']
        
        # Predict with adjusted threshold
        y_pred, y_probs = predict_with_adjusted_threshold(model, X_test, threshold=0.15)
        
        # Calculate metrics
        jaccard = jaccard_score(y_test, y_pred, average='samples')
        hamming = hamming_loss(y_test, y_pred)
        partial_acc = accuracy(y_test, y_pred)
        
        print(f"{name.upper()} Test Jaccard Score: {jaccard:.4f}")
        print(f"{name.upper()} Test Hamming Loss: {hamming:.4f}")
        print(f"{name.upper()} Test Partial Accuracy (≥1 correct in 3): {partial_acc:.4f}")
        
        # Display classification report for a few selected genres (top 5 most common)
        genre_counts = np.sum(y_test, axis=0)
        top_genre_indices = np.argsort(genre_counts)[-5:]
        
        print(f"\nClassification Report for Top 5 Most Common Genres ({name.upper()}):")
        for idx in top_genre_indices:
            genre_name = mlb.classes_[idx]
            print(f"\nGenre: {genre_name}")
            report = classification_report(y_test[:, idx], y_pred[:, idx])
            print(report)
        
        # Create and plot confusion matrix for a selected genre (most common genre)
        most_common_genre_idx = np.argmax(genre_counts)
        most_common_genre = mlb.classes_[most_common_genre_idx]
        
        cm = confusion_matrix(y_test[:, most_common_genre_idx], y_pred[:, most_common_genre_idx])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not ' + most_common_genre, most_common_genre],
                   yticklabels=['Not ' + most_common_genre, most_common_genre])
        plt.title(f'Confusion Matrix for "{most_common_genre}" Genre - {name.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../data/stats2/{name}_confusion_matrix.png')
        plt.close()
        
        # Create performance by genre
        genre_performance = {}
        for i, genre in enumerate(mlb.classes_):
            genre_jaccard = jaccard_score(y_test[:, i], y_pred[:, i])
            # Only include the genre if it's not in the exclusion list
            if genre.lower() not in genres_to_exclude:
                genre_performance[genre] = genre_jaccard
        
        # Plot all genres by Jaccard score
        plt.figure(figsize=(14, 14))
        
        # Sort genres by performance
        sorted_genres = sorted(genre_performance.items(), key=lambda x: x[1], reverse=True)
        
        # All genres
        all_names = [g[0] for g in sorted_genres]
        all_scores = [g[1] for g in sorted_genres]
        
        # Calculate number of genres
        n_genres = len(all_names)
        
        # Create colormap based on score values
        colors = plt.cm.viridis(np.linspace(0, 1, n_genres))
        
        # Plot all genres
        plt.barh(all_names, all_scores, color=colors)
        plt.title(f'Jaccard Score for All Genres - {name.title()}')
        plt.xlabel('Jaccard Score')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add a vertical line for average Jaccard score
        avg_score = np.mean(all_scores)
        plt.axvline(x=avg_score, color='red', linestyle='--', 
                   label=f'Average: {avg_score:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'../data/stats2/{name}_all_genre_performance.png')
        plt.close()
        
        results[name] = {
            'jaccard': jaccard,
            'hamming_loss': hamming,
            'partial_accuracy': partial_acc,
            'predictions': y_pred,
            'probabilities': y_probs,
            'genre_performance': genre_performance
        }
    
    return results

# Function to visualize results
def visualize_results(models, X_test, y_test, mlb):
    print("Visualizing results...")
    # Get best model based on partial accuracy
    best_model_name = max(models, key=lambda k: models[k].get('partial_accuracy', 0))
    
    # Calculate metrics if missing
    for model_name, model_info in models.items():
        if 'partial_accuracy' not in model_info:
            print(f"Calculating metrics for {model_name} on the fly...")
            
            # Predict with adjusted threshold
            y_pred, y_probs = predict_with_adjusted_threshold(model_info['model'], X_test, threshold=0.15)
            
            # Calculate metrics
            model_info['partial_accuracy'] = accuracy(y_test, y_pred)
            model_info['probabilities'] = y_probs
    
    # Plot metrics comparison
    plt.figure(figsize=(14, 8))
    
    # Metrics for each model
    jaccard_scores = [models[model]['jaccard'] for model in models]
    hamming_scores = [models[model]['hamming_loss'] for model in models]
    partial_acc_scores = [models[model]['partial_accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, jaccard_scores, width, color='blue', label='Jaccard Score')
    plt.bar(x, hamming_scores, width, color='orange', label='Hamming Loss')
    plt.bar(x + width, partial_acc_scores, width, color='red', label='Partial Accuracy (≥1 correct in 3)')
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(x, models.keys())
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target Score (85%)')
    plt.ylim(0, 1.0)
    
    # Add values on bars
    for i, v in enumerate(jaccard_scores):
        plt.text(i - width, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(hamming_scores):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(partial_acc_scores):
        plt.text(i + width, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('../data/stats2/model_comparison.png')
    plt.close()
    
    # Create a radar chart to compare model performance across multiple metrics
    plt.figure(figsize=(10, 8))
    
    # Define metrics to compare
    metrics = ['Jaccard Score', 'Hamming Loss', 'Partial Accuracy']
    
    # Number of metrics
    N = len(metrics)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Radar chart plot
    ax = plt.subplot(111, polar=True)
    
    # For each model
    for i, model_name in enumerate(models.keys()):
        model = models[model_name]
        
        # Get metric values
        values = [
            model['jaccard'],
            model['hamming_loss'],
            model['partial_accuracy']
        ]
        values += values[:1]  # Close the loop
        
        # Plot model values
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    plt.xticks(angles[:-1], metrics)
    
    # Y axis limits
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Radar Chart')
    plt.tight_layout()
    plt.savefig('../data/stats2/model_radar_chart.png')
    plt.close()

# Function to predict genre
def predict_genre(title, description, rating, models, tfidf, le, svd, scaler, mlb, model_choice=None):
    """
    Predict genres using traditional ML models
    """
    # List of genres to exclude from results
    genres_to_exclude = ['western', 'sports & fitness', 'short', 'lgbtq+']
    
    # Process input
    description = description.lower()
    description_tfidf = tfidf.transform([description])
    
    try:
        rating_encoded = le.transform([rating]).reshape(-1, 1)
    except:
        print(f"Warning: Rating '{rating}' not seen during training. Using most common rating instead.")
        rating_encoded = np.array([[0]])  # Default to first class
    
    # Combine features
    X_input = np.hstack((description_tfidf.toarray(), rating_encoded))
    
    # Reduce dimensions and scale
    X_input_reduced = svd.transform(X_input)
    X_input_scaled = scaler.transform(X_input_reduced)
    
    # Select model based on user choice or default to best model
    if model_choice and model_choice in models:
        chosen_model = models[model_choice]['model']
        model_name = model_choice
    else:
        # Default to best model based on partial accuracy
        model_name = max(models, key=lambda k: models[k].get('partial_accuracy', 0))
        chosen_model = models[model_name]['model']
        
    print(f"Using {model_name.upper()} model for prediction")
    
    # Get probabilities for all genres
    probs = []
    for estimator in chosen_model.estimators_:
        if hasattr(estimator, 'predict_proba'):
            probs.append(estimator.predict_proba(X_input_scaled)[0][1])
    
    probs = np.array(probs)
    
    # Filter out excluded genres from results
    valid_indices = [i for i, genre in enumerate(mlb.classes_) if genre.lower() not in genres_to_exclude]
    
    # Get top 3 genres with highest probabilities from valid genres
    valid_probs = probs[valid_indices]
    valid_indices_sorted = np.argsort(valid_probs)[::-1]
    top3_indices = [valid_indices[i] for i in valid_indices_sorted[:3]]
    predicted_genres = mlb.classes_[top3_indices]
    
    # Convert to (genre, probability) pairs for all valid genres
    genre_probs = [(mlb.classes_[i], probs[i]) for i in valid_indices]
    
    # Sort by probability
    genre_probs.sort(key=lambda x: x[1], reverse=True)
    
    result = {
        'title': title,
        'predicted_genres': predicted_genres,
        'top_genres': genre_probs[:3],  # Top 3 genres with probabilities
        'model_used': model_name,
        'partial_accuracy': models[model_name]['partial_accuracy']
    }
    
    return result

# Function to save models and components
def save_models(models, tfidf, le, svd, scaler, mlb):
    print("Saving models and components...")
    # Create a dict with all components
    components = {
        'models': models,
        'tfidf': tfidf,
        'label_encoder': le,
        'svd': svd,
        'scaler': scaler,
        'multilabel_binarizer': mlb
    }
    
    # Save to pickle file
    with open('../data/models/genre_classifier.pkl', 'wb') as f:
        pickle.dump(components, f)
    
    print("Models and components saved to '../data/models/genre_classifier.pkl'")

# Function to load models and components
def load_models():
    print("Loading models and components...")
    try:
        with open('../data/models/genre_classifier.pkl', 'rb') as f:
            components = pickle.load(f)
        
        print("Models and components loaded successfully.")
        return components
    except FileNotFoundError:
        print("No saved models found. Will train new models.")
        return None

# Main function
def main():
    # File path
    file_path = '../data/processed/movies_data_2.csv'
    
    # Check if saved models exist
    saved_components = load_models()
    
    if saved_components:
        # Use saved models
        models = saved_components['models']
        tfidf = saved_components['tfidf']
        le = saved_components['label_encoder']
        svd = saved_components['svd']
        scaler = saved_components['scaler']
        mlb = saved_components['multilabel_binarizer']
        
        print("Using pre-trained models. Skipping training process.")
        
        # Load data just for visualization and evaluation
        df = load_and_clean_data(file_path)
        X, y, _ = extract_features(df)  # Use existing mlb
        
        # Check if metrics exist in model info, calculate if missing
        for model_name, model_info in models.items():
            # Skip single-label models - we're not using them anymore
            if model_name.endswith('_single'):
                continue
                
            if 'partial_accuracy' not in model_info:
                print(f"Calculating metrics for {model_name} model...")
                # Split data to get a test set
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
                _, X_test_processed, _, _ = process_features(_, X_test)
                X_test_reduced = svd.transform(X_test_processed)
                X_test_reduced = scaler.transform(X_test_reduced)
                
                # Calculate metrics
                y_pred, y_probs = predict_with_adjusted_threshold(model_info['model'], X_test_reduced, threshold=0.15)
                model_info['partial_accuracy'] = accuracy(y_test, y_pred)
                model_info['probabilities'] = y_probs
                
        # Remove single-label models if they exist
        keys_to_remove = [k for k in list(models.keys()) if k.endswith('_single')]
        for key in keys_to_remove:
            if key in models:
                del models[key]
        
        # Show best model based on partial accuracy
        best_model = max(models, key=lambda k: models[k].get('partial_accuracy', 0))
        print(f"\nBest model based on Accuracy: {best_model.upper()}")
        print(f"Accuracy: {models[best_model]['partial_accuracy']:.4f}")
        print(f"Jaccard score: {models[best_model]['jaccard']:.4f}")
        print(f"Hamming loss: {models[best_model]['hamming_loss']:.4f}")
        
        # Save updated models
        save_models(models, tfidf, le, svd, scaler, mlb)
    else:
        # Load and clean data
        df = load_and_clean_data(file_path)
        
        # Extract features and prepare multi-label target
        X, y, mlb = extract_features(df)
        
        # Split data: 70% train, 15% validation, 15% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)
        
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Process features
        X_train_processed, X_val_processed, tfidf, le = process_features(X_train, X_val)
        _, X_test_processed, _, _ = process_features(X_train, X_test)
        
        # Reduce dimensions
        n_components = min(100, min(X_train_processed.shape[0], X_train_processed.shape[1]) - 1)
        X_train_reduced, X_val_reduced, svd, scaler = reduce_dimensions(X_train_processed, X_val_processed, n_components)
        X_test_reduced = svd.transform(X_test_processed)
        X_test_reduced = scaler.transform(X_test_reduced)
        
        # Visualize data
        visualize_data(X_train_reduced, y_train, mlb, "Training Data Visualization (Reduced Dimensions)")
        
        # Train models
        models = train_models(X_train_reduced, y_train, X_val_reduced, y_val, mlb)
        
        # Evaluate models
        results = evaluate_models(models, X_test_reduced, y_test, mlb)
        
        # Visualize results
        visualize_results(models, X_test_reduced, y_test, mlb)
        
        # Show best model based on partial accuracy
        best_model = max(models, key=lambda k: models[k].get('partial_accuracy', 0))
        print(f"\nBest model based on Accuracy: {best_model.upper()}")
        print(f"Accuracy: {models[best_model]['partial_accuracy']:.4f}")
        print(f"Jaccard score: {models[best_model]['jaccard']:.4f}")
        print(f"Hamming loss: {models[best_model]['hamming_loss']:.4f}")
        
        # Save models and components
        save_models(models, tfidf, le, svd, scaler, mlb)
    
    # Interactive prediction
    print("\n" + "="*50)
    print("Genre Prediction System")
    print("="*50)
    
    # Display metrics for the best model
    best_model = max(models, key=lambda k: models[k].get('partial_accuracy', 0))
    print(f"\nBest model: {best_model.upper()}")
    print(f"Accuracy: {models[best_model]['partial_accuracy']:.4f}")
    print(f"Jaccard score: {models[best_model]['jaccard']:.4f}")
    print(f"Hamming loss: {models[best_model]['hamming_loss']:.4f}")
    
    while True:
        print("\nEnter movie details (or 'exit' to quit):")
        title = input("Title: ")
        if title.lower() == 'exit':
            break
        
        description = input("Description: ")
        rating = input("Rating (e.g., PG, PG-13, R): ")
        
        # Let user choose model
        print("\nChoose a model for prediction:")
        print("1. Logistic Regression (C=1.0)")
        print("2. Logistic Regression (C=2.0)")
        print("3. SVM")
        print("4. Best model (based on Accuracy)")
        
        model_choice = input(f"Enter choice (1-4): ")
        
        if model_choice == '1':
            model_name = 'logistic_regression'
        elif model_choice == '2':
            model_name = 'logistic_regression_c2'
        elif model_choice == '3':
            model_name = 'svm'
        else:
            model_name = None  # Will use best model based on partial accuracy
        
        result = predict_genre(title, description, rating, models, tfidf, le, svd, scaler, mlb, model_name)
        
        print("\nPrediction Result:")
        print(f"Movie: {result['title']}")
        print(f"Model used: {result['model_used'].upper()}")
        print(f"Model's Accuracy: {result['partial_accuracy']:.4f}")
        
        print("\nPredicted Genres (Top 3):")
        for genre in result['predicted_genres']:
            print(f"  {genre}")
        
        print("\nGenres with Probabilities:")
        for genre, prob in result['top_genres']:
            prob_percent = prob * 100
            stars = "*" * int(prob_percent / 10)
            print(f"  {genre}: {prob_percent:.2f}% {stars}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
