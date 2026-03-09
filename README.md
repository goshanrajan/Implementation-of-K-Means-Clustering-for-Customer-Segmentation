# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1: Import the required Python libraries and load the customer dataset containing features such as customer income and spending score.

2: Select the relevant features from the dataset and determine the optimal number of clusters (K) using the Elbow Method.

3: Train the K-Means clustering model with the chosen value of K and group the customers into clusters based on similarity.

4: Visualize the clusters using a scatter plot and display the customer segmentation results. 

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: T.Goshanrajan
RegisterNumber:  212225040098

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== K-Means Clustering for Customer Segmentation ===")
print("Loading Mall_Customers.csv dataset...")

# 1. Load the dataset
df = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset columns:", df.columns.tolist())

# 2. Select features for clustering (Annual Income & Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nClustering features selected: Annual Income (k$) & Spending Score (1-100)")

# 3. Elbow Method - Find optimal number of clusters
print("\nFinding optimal number of clusters using Elbow Method...")
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)

# Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.grid(True)
plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Elbow plot saved as 'elbow_plot.png'")
print("Optimal K appears to be around 5 clusters")

# 4. Train K-Means model with optimal clusters (K=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("\nK-Means model trained with 5 clusters")
print("Cluster centers:")
print(pd.DataFrame(kmeans.cluster_centers_, 
                   columns=['Annual Income (k$)', 'Spending Score (1-100)']))

# 5. Visualize Customer Segments
plt.figure(figsize=(12, 5))

# Cluster scatter plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                     c=df['Cluster'], cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments (K=5)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Cluster distribution
plt.subplot(1, 2, 2)
cluster_counts = df['Cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Customers per Cluster')
plt.xticks(range(5))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')
plt.show()

print("Customer segments plot saved as 'customer_segments.png'")

# 6. Interactive Plotly visualization
fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                 color='Cluster', size_max=8, opacity=0.7,
                 title='Customer Segments - Interactive Plot',
                 hover_data=['CustomerID'])
fig.show()
fig.write_html('interactive_segments.html')

# 7. Prediction function for new customers
def predict_segment(annual_income, spending_score):
    """
    Predict customer segment for new customer input
    Input: annual_income (k$), spending_score (1-100)
    Output: cluster number (0-4)
    """
    new_customer = np.array([[annual_income, spending_score]])
    cluster = kmeans.predict(new_customer)[0]
    
    # Interpret cluster characteristics
    cluster_info = {
        0: "Conservative: Low Income, Low Spending",
        1: "Balanced: Medium Income, Medium Spending", 
        2: "High Earners, Low Spenders",
        3: "Target: High Income, High Spending",
        4: "Young Spenders: Low Income, High Spending"
    }
    
    return cluster, cluster_info.get(cluster, "Unknown Segment")

# 8. Interactive prediction from user input
print("\n" + "="*50)
print("CUSTOMER SEGMENT PREDICTION")
print("="*50)

try:
    print("\nEnter customer details:")
    annual_income = float(input("Annual Income (k$): "))
    spending_score = float(input("Spending Score (1-100): "))
    
    cluster_num, description = predict_segment(annual_income, spending_score)
    print(f"\nCustomer belongs to Cluster {cluster_num}")
    print(f"Segment: {description}")
    print(f"Marketing Strategy: Target this segment with appropriate campaigns")
    
except ValueError:
    print("Please enter valid numeric values!")
except Exception as e:
    print(f"Error: {e}")

# 9. Summary Statistics
print("\n" + "="*50)
print("CLUSTER SUMMARY")
print("="*50)
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary.round(2))

print(f"\nTotal customers segmented: {len(df)}")
print(f"Clusters created: 5")
print("\n Program executed successfully!")

print("\nFiles generated:")
print("- elbow_plot.png")
print("- customer_segments.png") 
print("- interactive_segments.html")

```

## Output:
<img width="1240" height="416" alt="image" src="https://github.com/user-attachments/assets/757e4ce8-868b-4aba-9cea-7fa7182bdd4d" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
