# Retail Customer Segmentation

## Project Description
Segmented 200 mall customers into 5 distinct groups based on 
annual income and spending score using K-Means clustering, 
enabling targeted marketing strategies.

## Dataset
Mall Customer Segmentation Dataset (Kaggle)
- 200 customers, 5 features: CustomerID, Gender, Age, Annual Income, Spending Score

## Technologies Used
- Python, Pandas, NumPy
- Scikit-learn (K-Means)
- Matplotlib, Seaborn

## Approach
1. Data cleaning & Exploratory Data Analysis
2. Feature scaling using StandardScaler
3. Elbow method to determine optimal clusters (K=5)
4. K-Means clustering applied on Income & Spending Score
5. Visualized and interpreted customer segments

## Visualizations


![Elbow Method](elbow%20method%20graph.png)




![Customer Segments](final%20cluster%20graph.png)




![Cluster Summary](terminal%20cluster%20summary%20table.png)



## Key Insights

| Cluster | Avg Income (k$) | Avg Spending Score | Segment Type |
|---------|-----------------|---------------------|---------------|
| 0 | 55.3 | 49.5 | Average customers |
| 1 | 86.5 | 82.1 | Premium customers |
| 2 | 25.7 | 79.4 | Careful spenders |
| 3 | 88.2 | 17.1 | Potential targets |
| 4 | 26.3 | 20.9 | Budget customers |

## Result
Identified 5 actionable customer segments to support 
data-driven marketing decisions.
