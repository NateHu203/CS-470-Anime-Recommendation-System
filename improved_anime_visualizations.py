import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global font parameters for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Load the datasets
anime = pd.read_csv(os.path.join("data/raw/","anime.csv"))
ratings = pd.read_csv(os.path.join("data/raw/","rating.csv"))

# --- 1. Top Anime Community Bar Plot ---
def plot_top_anime_by_members(anime_df, top_n=15):
    top_anime = anime_df.sort_values('members', ascending=False).head(top_n)
    
    plt.figure(figsize=(12,8))  # Increased figure size
    bars = sns.barplot(x='name', y='members', data=top_anime, palette='Paired')
    plt.xticks(rotation=90)
    plt.title('Top Anime Community', fontsize=20, fontweight='bold')  # Larger title
    plt.xlabel('Anime Name', fontsize=16)
    plt.ylabel('Total Member', fontsize=16)
    
    for bar, value in zip(bars.patches, top_anime['members']):
        bars.annotate(f'{int(value):,}', (bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                      ha='center', va='center', fontsize=10, color='black',  # Increased annotation size
                      bbox=dict(boxstyle="round,pad=0.3", fc="orange", ec="black", lw=1))
    
    plt.tight_layout()
    plt.show()

# --- 2. Improved Anime Categories Donut Plot with better colors ---
def plot_category_distribution(anime_df):
    category_counts = anime_df['type'].value_counts()
    
    # Using a more visually appealing color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_counts)))
    
    plt.figure(figsize=(12,10))  # Increased figure size
    wedges, texts, autotexts = plt.pie(category_counts, labels=None, autopct='%1.2f%%', 
                                       startangle=140, wedgeprops=dict(width=0.5),
                                       colors=colors, textprops={'fontsize': 14})  # Increased text size
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    plt.title('Anime Categories Distribution', fontsize=22, fontweight='bold')  # Larger title
    
    # Create more visually attractive legend
    plt.legend(wedges, category_counts.index, title="Category Types", 
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
               fontsize=14, title_fontsize=18)  # Larger legend
    
    plt.tight_layout()
    plt.show()

# --- 3. Overall Anime Ratings Distribution ---
def plot_anime_ratings_distribution(anime_df):
    plt.figure(figsize=(12,7))  # Increased figure size
    sns.histplot(anime_df['rating'].dropna(), kde=True, color='gold', bins=20)
    plt.title("Anime's Average Ratings Distribution", fontsize=20, fontweight='bold')  # Larger title
    plt.xlabel('Rating', fontsize=16)
    plt.ylabel('Total', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

# --- 4. Overall Users Anime Ratings Distribution ---
def plot_users_ratings_distribution(ratings_df):
    valid_ratings = ratings_df[ratings_df['rating'] != -1]
    plt.figure(figsize=(12,7))  # Increased figure size
    sns.histplot(valid_ratings['rating'], kde=True, color='gold', bins=20)
    plt.title("Users Anime Ratings Distribution", fontsize=20, fontweight='bold')  # Larger title
    plt.xlabel('Rating', fontsize=16)
    plt.ylabel('Total', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

# --- 5. Consolidated Category-Wise Rating Distributions ---
def plot_all_category_distributions(anime_df, ratings_df):
    categories = anime_df['type'].dropna().unique()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))  # Increased figure size
    
    # Color palette for different categories
    palette = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    # First subplot: Anime's Average Ratings by Category
    for i, category in enumerate(categories):
        anime_filtered = anime_df[anime_df['type'] == category]
        if len(anime_filtered) > 0:
            sns.kdeplot(anime_filtered['rating'].dropna(), ax=ax1, 
                      label=f"{category} (n={len(anime_filtered)})", 
                      color=palette[i], fill=True, alpha=0.2)
    
    ax1.set_title("Anime's Average Ratings Distribution by Category", fontsize=20, fontweight='bold')  # Larger title
    ax1.set_xlabel('Rating', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.legend(title="Category", fontsize=14, title_fontsize=16)  # Larger legend
    ax1.tick_params(axis='both', which='major', labelsize=14)  # Larger tick labels
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Second subplot: Users' Ratings by Category
    for i, category in enumerate(categories):
        anime_filtered = anime_df[anime_df['type'] == category]
        merged = pd.merge(ratings_df, anime_filtered[['anime_id', 'name']], on='anime_id')
        valid_user_ratings = merged[merged['rating'] != -1]
        
        if len(valid_user_ratings) > 0:
            sns.kdeplot(valid_user_ratings['rating'], ax=ax2, 
                       label=f"{category} (n={len(valid_user_ratings)})", 
                       color=palette[i], fill=True, alpha=0.2)
    
    ax2.set_title("Users' Anime Ratings Distribution by Category", fontsize=20, fontweight='bold')  # Larger title
    ax2.set_xlabel('Rating', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.legend(title="Category", fontsize=14, title_fontsize=16)  # Larger legend
    ax2.tick_params(axis='both', which='major', labelsize=14)  # Larger tick labels
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the visualizations
    plot_top_anime_by_members(anime)
    plot_category_distribution(anime)
    plot_anime_ratings_distribution(anime)
    plot_users_ratings_distribution(ratings)
    plot_all_category_distributions(anime, ratings) 