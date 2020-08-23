"""
Descriptions:
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their
previous actions or explicit feedback.

Cons:
The system is limit to recommend the same category items to user based on their past purchase

EXPLAIN ALGORITHM:
* User 1101 bought items [A, B, C, D]
-----------------------------------------------
     |   A      |    B    |    C     |     D  |
-----------------------------------------------
1101 |   10     |    15   |    32    |    10  | 
-----------------------------------------------

* Look thru a dictionary of items, found item E & F with highest similarity. Thus, refer E & F for user 1101
---------------------------------------------
    |   char_1  |   char_2   |   char_3     |
---------------------------------------------
A   |    1      |      1     |    0         |
---------------------------------------------
E   |    1      |      1     |    0         |
---------------------------------------------
F   |    1      |      1     |    0         |
---------------------------------------------
"""
import numpy as np
import os

from surprise import AlgoBase
from surprise import PredictionImpossible
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from .constants import ITEM_SIMILARITIES_DIR


class ContentKNNAlgo(AlgoBase):
    def __init__(self, items, k=20, verbose=False):
        super().__init__()
        self.similarities = None
        self.k = k
        self.items = items
        self.verbose = verbose

    def fit(self, train_set):
        # Initialize train_set
        AlgoBase.fit(self, train_set)

        if os.path.isdir(ITEM_SIMILARITIES_DIR):
            if self.verbose:
                print('Loading similarities between items....')
            with open(ITEM_SIMILARITIES_DIR, 'rb') as file:
                self.similarities = np.load(file)
            if self.verbose:
                print('Finish loading similarities between items')
        else:
            # Load groceries items vectors to compute similarities between items
            # Note: this takes loooooong time to run
            if self.verbose:
                print('Computing similarities between items, will take a while....')
            self.compute_cosine_similarity()
            if self.verbose:
                print('Finish computing similarities between items')

            with open(ITEM_SIMILARITIES_DIR, 'wb') as file:
                np.save(file, self.similarities)

    def compute_cosine_similarity(self):
        # Compute from items vector to similarity
        # It's gonna take roughly 90 mins for 182,726 items to be processed
        items_mat = self.items.drop(['itemID'], axis=1).values
        n = items_mat.shape[0]
        self.similarities = np.zeros((n, n))
        for i in range(n):
            if self.verbose:
                print(i)
            self.similarities[i, :] = cosine_similarity([items_mat[i]], items_mat)

    def estimate(self, user_id, item_id):
        if not (self.trainset.knows_user(user_id) and self.trainset.knows_item(item_id)):
            raise PredictionImpossible('Unknown user ID and/or item ID')

        neighbors = []
        for rating in self.trainset.ur[user_id]:
            item_similarity = self.similarities[item_id, rating[0]]
            neighbors.append((item_similarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        sim_total = weighted_sum = 0
        for (sim_score, rating) in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating

        if sim_total == 0:
            raise PredictionImpossible('No neighbors')

        predicted_rating = weighted_sum / sim_total

        return predicted_rating

    def recommend_per_item(self, item_id, num_result=5):
        # Given item ID, return n recommended items that are similar
        if not (self.trainset.knows_user(item_id)):
            raise PredictionImpossible('Unknown item ID')

        result = {}
        for idx, row in self.trainset.iterrows():
            similar_indices = self.similarities[idx].argsort()[:-100:-1]
            similar_items = [(self.similarities[idx][i], self.trainset['id'][i]) for i in similar_indices]
            result[row['id']] = similar_items[1:]
        return result

    def recommend_per_user(self, user_id, num_result=5):
        # Given user ID, return n recommended items for each item the user ID has brought
        if not (self.trainset.knows_user(user_id)):
            raise PredictionImpossible('Unknown user ID')
