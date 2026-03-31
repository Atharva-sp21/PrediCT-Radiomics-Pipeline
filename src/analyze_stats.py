import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    if not os.path.exists('radiomics_features.csv'):
        print("radiomics_features.csv not found!")
        return

    df = pd.read_csv('radiomics_features.csv')
    
    # 1. Separate numeric radiomics features
    basic_cols = ['Patient_ID', 'Agatston_Score', 'Agatston_Category']
    feature_cols = [c for c in df.columns if c not in basic_cols]
    
    # 2. Spearman correlation
    correlations = []
    agatston = df['Agatston_Score'].values
    for col in feature_cols:
        # Check if the feature array has standard deviation zero
        if np.std(df[col]) == 0:
            continue
        stat, p = stats.spearmanr(df[col], agatston)
        if not np.isnan(stat):
            correlations.append((col, stat, p))
            
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top 3 most correlated features:")
    for feat, stat, p in correlations[:3]:
        print(f"{feat}: stat={stat:.4f}, p={p:.4e}")
        
    # Feature Importance (XAI) Barplot
    top_10_corr = correlations[:10]
    top_10_corr_feats = [x[0] for x in top_10_corr]
    top_10_corr_vals = [abs(x[1]) for x in top_10_corr]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_corr_vals, y=top_10_corr_feats, palette='mako')
    plt.xlabel('Absolute Spearman Correlation')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importance (Correlation with Agatston Score)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("\nSaved feature_importance.png")
        
    # 3. Kruskal-Wallis test on shape_Sphericity
    sphericity_col = None
    for c in feature_cols:
        if 'Sphericity' in c:
            sphericity_col = c
            break
            
    if sphericity_col:
        groups = [group[sphericity_col].values for name, group in df.groupby('Agatston_Category')]
        if len(groups) > 1:
            h_stat, p_kw = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis on {sphericity_col}: H={h_stat:.4f}, p={p_kw:.4e}")
        else:
            print(f"\nNot enough groups for Kruskal-Wallis on {sphericity_col}")
    else:
        print("\nSphericity feature not found!")
        
    # 4. Standardize numeric features
    X = df[feature_cols].values
    X = np.nan_to_num(X) # handle any zero variance issues silently
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]
    
    # 6. Scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x='UMAP1', 
        y='UMAP2', 
        hue='Agatston_Category', 
        palette='viridis',
        s=100
    )
    plt.title('UMAP Clustering of Radiomics Features')
    plt.legend(title='Agatston Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('calcium_phenotypes_umap.png', dpi=300)
    print("\nSaved UMAP plot to calcium_phenotypes_umap.png")

    # 7. Correlation Matrix of top 10 most variable features
    feature_variances = np.var(X_scaled, axis=0)
    top_10_idx = np.argsort(feature_variances)[-10:][::-1]
    top_10_features = [feature_cols[i] for i in top_10_idx]

    top_10_data = df[top_10_features]
    corr_matrix, _ = stats.spearmanr(top_10_data)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, xticklabels=top_10_features, yticklabels=top_10_features, annot=True, cmap='coolwarm')
    plt.title('Spearman Correlation Matrix (Top 10 Most Variable Features)')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    print("\nSaved correlation_matrix.png")

if __name__ == '__main__':
    main()
