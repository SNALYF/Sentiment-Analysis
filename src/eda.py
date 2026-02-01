import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def perform_eda(df, output_dir='img'):
    """
    Performs EDA on the training set and saves plots to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Performing EDA...")
    
    # Create a copy to avoid modifying the original dataframe
    train_eda = df.copy()
    train_eda['review_length'] = train_eda['content'].apply(lambda x: len(str(x).split()))

    # Rating Distribution
    plt.figure(figsize=(8, 6))
    rating_counts = train_eda['rating'].value_counts().sort_index()
    # Ensure order is 1star to 5star
    order = ['1star', '2star', '3star', '4star', '5star']
    sns.countplot(x='rating', data=train_eda, order=order, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
    plt.close()
    print(f"Saved rating_distribution.png to {output_dir}")

    # Review Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_eda, x='review_length', bins=60, kde=True, hue='rating', hue_order=order, palette='viridis', multiple='stack')
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Word Count per Review')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'review_length_distribution.png'))
    plt.close()
    print(f"Saved review_length_distribution.png to {output_dir}")
