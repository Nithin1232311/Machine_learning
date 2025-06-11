
# Advanced KMeans: Customer Segmentation
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Customer data: Annual Income and Spending Score
df = pd.DataFrame({
    'Income': [15, 16, 17, 25, 30, 40, 55, 65, 70, 80, 90, 100],
    'Spending': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99]
})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_data)

plt.scatter(df['Income'], df['Spending'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
