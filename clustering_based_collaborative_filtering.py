import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
books = pd.read_csv('C:/Users/ahmednabil/Documents/Downloads/AIE425_Assignment_Submission/Dataset/BX-Users.csv', sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False)
users = pd.read_csv('C:/Users/ahmednabil/Documents/Downloads/AIE425_Assignment_Submission/Dataset/BX-Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('C:/Users/ahmednabil/Documents/Downloads/AIE425_Assignment_Submission/Dataset/BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False)

# Remove duplicates
books = books.drop_duplicates()
users = users.drop_duplicates()
ratings = ratings.drop_duplicates()

# Replace out-of-bound values with the mean (if applicable)
lower_bound = 0
upper_bound = 10
valid_mean = ratings[(ratings['Book-Rating'] >= lower_bound) & (ratings['Book-Rating'] <= upper_bound)]['Book-Rating'].mean()
ratings['Book-Rating'] = ratings['Book-Rating'].apply(lambda x: valid_mean if not lower_bound <= x <= upper_bound else x)

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
ratings_scaled = ratings.copy()
ratings_scaled['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])

# Create a pivot table for user-item interactions
user_item_matrix = ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)
user_item_matrix_scaled = pd.DataFrame(user_item_matrix_scaled, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Clustering Users
# Apply K-means clustering on users
kmeans_user = KMeans(n_clusters=5, random_state=42)
user_clusters_kmeans = kmeans_user.fit_predict(user_item_matrix_scaled)
user_item_matrix_scaled['UserCluster_KMeans'] = user_clusters_kmeans

# Evaluate the clustering using silhouette score
silhouette_kmeans_user = silhouette_score(user_item_matrix_scaled, user_clusters_kmeans)
print(f"Silhouette Score for K-means user clustering: {silhouette_kmeans_user}")

# Apply Mean-Shift clustering on users
meanshift_user = MeanShift()
user_clusters_meanshift = meanshift_user.fit_predict(user_item_matrix_scaled)
user_item_matrix_scaled['UserCluster_MeanShift'] = user_clusters_meanshift

# Evaluate the clustering using silhouette score
silhouette_meanshift_user = silhouette_score(user_item_matrix_scaled, user_clusters_meanshift)
print(f"Silhouette Score for Mean-Shift user clustering: {silhouette_meanshift_user}")

# Clustering Items
# Transpose the dataset for item clustering
item_user_matrix_scaled = user_item_matrix_scaled.T

# Apply K-means clustering on items
kmeans_item = KMeans(n_clusters=5, random_state=42)
item_clusters_kmeans = kmeans_item.fit_predict(item_user_matrix_scaled)
item_user_matrix_scaled['ItemCluster_KMeans'] = item_clusters_kmeans

# Evaluate the clustering using silhouette score
silhouette_kmeans_item = silhouette_score(item_user_matrix_scaled, item_clusters_kmeans)
print(f"Silhouette Score for K-means item clustering: {silhouette_kmeans_item}")

# Apply Mean-Shift clustering on items
meanshift_item = MeanShift()
item_clusters_meanshift = meanshift_item.fit_predict(item_user_matrix_scaled)
item_user_matrix_scaled['ItemCluster_MeanShift'] = item_clusters_meanshift

# Evaluate the clustering using silhouette score
silhouette_meanshift_item = silhouette_score(item_user_matrix_scaled, item_clusters_meanshift)
print(f"Silhouette Score for Mean-Shift item clustering: {silhouette_meanshift_item}")

# Collaborative Filtering
def user_based_cf(user_id, item_id, user_similarity, ratings):
    similar_users = user_similarity[user_id].drop(user_id).dropna()
    similar_users = similar_users[similar_users > 0].sort_values(ascending=False)
    
    weighted_sum = 0
    sum_of_weights = 0
    
    for other_user, similarity in similar_users.items():
        if not np.isnan(ratings[other_user][item_id]):
            weighted_sum += similarity * ratings[other_user][item_id]
            sum_of_weights += similarity
            
    if sum_of_weights == 0:
        return np.nan
    
    return weighted_sum / sum_of_weights

# Calculate Pearson Correlation for user-based CF
user_similarity_kmeans = user_item_matrix_scaled.drop(columns=['UserCluster_KMeans']).T.corr(method='pearson')
user_similarity_meanshift = user_item_matrix_scaled.drop(columns=['UserCluster_MeanShift']).T.corr(method='pearson')

# Predict ratings for a user-item pair using K-means user clusters
user_id = 7  
item_id = 3  
actual_rating_normalized = user_item_matrix_scaled.loc[user_id, item_id]
actual_rating = actual_rating_normalized * 10  # Convert to original range
predicted_rating_kmeans_user_normalized = user_based_cf(user_id, item_id, user_similarity_kmeans, user_item_matrix_scaled.values)
predicted_rating_kmeans_user = predicted_rating_kmeans_user_normalized * 10  # Convert to original range
print(f"Actual rating for user {user_id} on item {item_id}: {actual_rating}")
print(f"Predicted rating for user {user_id} on item {item_id} using K-means user-based CF: {predicted_rating_kmeans_user}")

# Predict ratings for a user-item pair using Mean-Shift user clusters
predicted_rating_meanshift_user_normalized = user_based_cf(user_id, item_id, user_similarity_meanshift, user_item_matrix_scaled.values)
predicted_rating_meanshift_user = predicted_rating_meanshift_user_normalized * 10  # Convert to original range
print(f"Predicted rating for user {user_id} on item {item_id} using Mean-Shift user-based CF: {predicted_rating_meanshift_user}")

# Calculate errors for user-based CF
error_kmeans_user = mean_absolute_error([actual_rating], [predicted_rating_kmeans_user])
error_meanshift_user = mean_absolute_error([actual_rating], [predicted_rating_meanshift_user])
print(f"Mean Absolute Error for K-means user-based CF: {error_kmeans_user}")
print(f"Mean Absolute Error for Mean-Shift user-based CF: {error_meanshift_user}")

# Item-Based Collaborative Filtering using Cosine Similarity
def item_based_cf(user_id, item_id, item_similarity, ratings):
    similar_items = item_similarity[item_id]
    rated_items = np.where(~np.isnan(ratings[user_id]))[0]
    
    weighted_sum = 0
    sum_of_weights = 0
    
    for other_item in rated_items:
        if not np.isnan(ratings[user_id][other_item]):
            weighted_sum += similar_items[other_item] * ratings[user_id][other_item]
            sum_of_weights += similar_items[other_item]
            
    if sum_of_weights == 0:
        return np.nan
    
    return weighted_sum / sum_of_weights

# Calculate Cosine Similarity for item-based CF
item_similarity_kmeans = cosine_similarity(item_user_matrix_scaled.drop(columns=['ItemCluster_KMeans']))
item_similarity_meanshift = cosine_similarity(item_user_matrix_scaled.drop(columns=['ItemCluster_MeanShift']))

# Predict ratings for a user-item pair using K-means item clusters
predicted_rating_kmeans_item_normalized = item_based_cf(user_id, item_id, item_similarity_kmeans, user_item_matrix_scaled.values)
predicted_rating_kmeans_item = predicted_rating_kmeans_item_normalized * 10  # Convert to original range
print(f"Predicted rating for user {user_id} on item {item_id} using K-means item-based CF: {predicted_rating_kmeans_item}")

# Predict ratings for a user-item pair using Mean-Shift item clusters
predicted_rating_meanshift_item_normalized = item_based_cf(user_id, item_id, item_similarity_meanshift, user_item_matrix_scaled.values)
predicted_rating_meanshift_item = predicted_rating_meanshift_item_normalized * 10  # Convert to original range
print(f"Predicted rating for user {user_id} on item {item_id} using Mean-Shift item-based CF: {predicted_rating_meanshift_item}")

# Calculate errors for item-based CF
error_kmeans_item = mean_absolute_error([actual_rating], [predicted_rating_kmeans_item])
error_meanshift_item = mean_absolute_error([actual_rating], [predicted_rating_meanshift_item])
print(f"Mean Absolute Error for K-means item-based CF: {error_kmeans_item}")
print(f"Mean Absolute Error for Mean-Shift item-based CF: {error_meanshift_item}")

# Visualizations
# Bar Plot of Silhouette Scores
silhouette_scores = {
    'K-means User': silhouette_kmeans_user,
    'Mean-Shift User': silhouette_meanshift_user,
    'K-means Item': silhouette_kmeans_item,
    'Mean-Shift Item': silhouette_meanshift_item
}
colors = ['red', 'darkred', 'blue', 'darkblue']

fig, ax = plt.subplots()
bars = ax.bar(silhouette_scores.keys(), silhouette_scores.values(), color=colors)
ax.set_xlabel('Clustering Method')
ax.set_ylabel('Silhouette Score')
ax.set_title('Comparison of Silhouette Scores for Clustering Methods')

# Adding value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.show()

# Scatter Plot of Actual vs. Predicted Ratings
actual_ratings = [actual_rating]
predicted_ratings_kmeans_user = [predicted_rating_kmeans_user]
predicted_ratings_meanshift_user = [predicted_rating_meanshift_user]
predicted_ratings_kmeans_item = [predicted_rating_kmeans_item]
predicted_ratings_meanshift_item = [predicted_rating_meanshift_item]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(actual_ratings, predicted_ratings_kmeans_user, color='r', label='K-means User CF', alpha=0.6)
ax.scatter(actual_ratings, predicted_ratings_meanshift_user, color='g', label='Mean-Shift User CF', alpha=0.6)
ax.scatter(actual_ratings, predicted_ratings_kmeans_item, color='b', label='K-means Item CF', alpha=0.6)
ax.scatter(actual_ratings, predicted_ratings_meanshift_item, color='y', label='Mean-Shift Item CF', alpha=0.6)
ax.plot([min(actual_ratings), max(actual_ratings)], [min(actual_ratings), max(actual_ratings)], color='k', linestyle='--', label='Ideal')
ax.set_xlabel('Actual Ratings')
ax.set_ylabel('Predicted Ratings')
ax.set_title('Actual vs. Predicted Ratings')
ax.legend()
plt.show()

# PCA with 3 Components for 3D Visualization
pca = PCA(n_components=3)
pca_result = pca.fit_transform(user_item_matrix_scaled.drop(columns=['UserCluster_KMeans', 'UserCluster_MeanShift']))

fig = plt.figure(figsize=(14, 6))

# K-means Clustering 3D Plot
ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=user_clusters_kmeans, cmap='viridis', alpha=0.6)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title('3D PCA Visualization of K-means Clustering')
ax.view_init(elev=20., azim=15)

# Mean-Shift Clustering 3D Plot
ax = fig.add_subplot(122, projection='3d')
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=user_clusters_meanshift, cmap='viridis', alpha=0.6)
legend2 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend2)
ax.set_title('3D PCA Visualization of Mean-Shift Clustering')
ax.view_init(elev=20., azim=15)

plt.show()
