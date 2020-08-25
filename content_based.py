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
import pandas as pd
import numpy as np
import os

from surprise import AlgoBase
from surprise import PredictionImpossible
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from .constants import ITEM_SIMILARITIES_DIR, CATEGORY_MAPPING


class ContentKNNAlgo(AlgoBase):
    def __init__(self, items, k=8, verbose=False):
        super().__init__()
        self.similarities = None
        self.k = k
        self.items = items
        self.verbose = verbose

    def fit(self, train_set):
        # Initialize train_set
        AlgoBase.fit(self, train_set)

        if os.path.isfile(ITEM_SIMILARITIES_DIR):
            if self.verbose:
                print('Loading similarities between items....')
            self.similarities = np.load(ITEM_SIMILARITIES_DIR, mmap_mode='r+')
            if self.verbose:
                print('Finish loading similarities between items')
        else:
            # Load groceries items vectors to compute similarities between items
            if self.verbose:
                print('Computing similarities between items, will take a while....')
            self.compute_cosine_similarity()
            if self.verbose:
                print('Finish computing similarities between items')

            with open(ITEM_SIMILARITIES_DIR, 'wb') as file:
                np.save(file, self.similarities)

    def compute_cosine_similarity(self):
        # Compute from items vector to similarity
        iter = 0
        n = self.items.shape[0]
        self.similarities = np.zeros((n, n))
        for name, k in CATEGORY_MAPPING.items():
            dt = self.items.loc[self.items['category'] == k]
            step = dt.shape[0]
            # TODO: Only salt snack has memory error, pass for now. Come back to fix later
            if k != 24:
                dt_mat = dt.drop(['itemID', 'category'], axis=1).values
                sim = cosine_similarity(dt_mat)
                self.similarities[iter: iter+step, iter: iter + step] = sim
            iter += step
        self.items = None

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
