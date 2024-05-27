# Clustering-Based-Collaborative-Filtering
This project implements a collaborative filtering recommendation system using clustering algorithms to enhance the recommendation accuracy. The system employs K-means and Mean-Shift clustering techniques to group users and items, followed by collaborative filtering to predict user ratings for unseen items.

Features
Data Preprocessing: Loading datasets, handling duplicates, and normalizing data.
User and Item Clustering: Applying K-means and Mean-Shift clustering algorithms to users and items.
Evaluation: Using silhouette scores to evaluate clustering performance.
Collaborative Filtering: Implementing user-based and item-based collaborative filtering.
Error Calculation: Measuring prediction accuracy using Mean Absolute Error (MAE).
Visualizations: Generating various plots to visualize clustering results and prediction accuracy.
Installation
Clone the repository:
sh
Copy code
git clone https://github.com/yourusername/clustering-based-collaborative-filtering.git
Navigate to the project directory:
sh
Copy code
cd clustering-based-collaborative-filtering
Install the required packages:
sh
Copy code
pip install -r requirements.txt
Usage
Place the dataset files (BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv) in the appropriate directory.
Run the script:
sh
Copy code
python clustering_based_collaborative_filtering.py
Dataset
BX-Users.csv: Contains user information.
BX-Books.csv: Contains book information.
BX-Book-Ratings.csv: Contains book ratings by users.
Code Structure
Data Loading and Preprocessing: Load datasets, handle missing values, and normalize data.
Clustering: Apply K-means and Mean-Shift clustering on users and items.
Collaborative Filtering: Implement user-based and item-based collaborative filtering methods.
Evaluation and Visualization: Evaluate clustering performance and visualize results.
Examples
Clustering Users and Items
python
Copy code
# Apply K-means clustering on users
kmeans_user = KMeans(n_clusters=5, random_state=42)
user_clusters_kmeans = kmeans_user.fit_predict(user_item_matrix_scaled)
user_item_matrix_scaled['UserCluster_KMeans'] = user_clusters_kmeans

# Evaluate the clustering using silhouette score
silhouette_kmeans_user = silhouette_score(user_item_matrix_scaled, user_clusters_kmeans)
print(f"Silhouette Score for K-means user clustering: {silhouette_kmeans_user}")
Collaborative Filtering Prediction
python
Copy code
# Predict ratings for a user-item pair using K-means user clusters
predicted_rating_kmeans_user_normalized = user_based_cf(user_id, item_id, user_similarity_kmeans, user_item_matrix_scaled.values)
predicted_rating_kmeans_user = predicted_rating_kmeans_user_normalized * 10  # Convert to original range
print(f"Predicted rating for user {user_id} on item {item_id} using K-means user-based CF: {predicted_rating_kmeans_user}")
Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
