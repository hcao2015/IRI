from surprise import accuracy
import numpy as np
from collections import defaultdict
from surprise.model_selection import train_test_split
from six import iteritems


class Evaluator(object):

    def __init__(self, algorithm, data, random_state=53421):
        self.algorithm = algorithm
        self.data = data
        self.random_state = random_state
        self.build_data()

    def build_data(self):
        # Build train/test sets as 75/25
        self.train_set, self.test_set = train_test_split(self.data, test_size=.25, random_state=self.random_state)

        # Build Leave Portion Out to stimulate user behavior
        # This will use the whole dataset and leave some items out per user
        LPO = LeavePortionOut(random_state=self.random_state)
        self.lpo_train, self.lpo_test = LPO.split(self.data)

    def rmse(self, predictions):
        return accuracy.rmse(predictions, verbose=False)

    def mae(self, predictions):
        return accuracy.mae(predictions, verbose=False)

    def hit_rate(self, top_n_recs, lpo_predictions):
        # The total number of items that are recommended to correct users
        # TODO: Need to implement
        total_hit = 0
        return

    def evaluate(self, top_n=5):
        metrics = {}

        self.algorithm.fit(self.train_set)
        predictions = self.algorithm.test(self.test_set)

        print("Evaluating with accuracy...")
        metrics["Root Mean Squared Error (RMSE)"] = self.rmse(predictions)
        metrics["Mean Absolute Error (MAE)"] = self.mae(predictions)

        precisions, recalls = self.precision_recall_at_k(predictions, k=8, threshold=0.1)
        metrics['precisions'] = sum(prec for prec in precisions.values()) / len(precisions)
        metrics['recalls'] = sum(rec for rec in recalls.values()) / len(recalls)

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

        if top_n:
            print("Evaluating with leave portion out...")
            self.algorithm.fit(self.lpo_train)

            # This will evaluate and return k-top recommended items from lpo_test
            lpo_predictions = self.algorithm.test(self.lpo_test)

            # Build anti-test set that for each user in trainset, get all items that are not purchase
            # Note: this will include a portion of items being leaveout in lpo_test
            anti_testset = self.lpo_train.build_anti_testset()

            # Build predictions for all ratings not in the training set
            all_predictions = self.algorithm.test(anti_testset)

            print("Evaluating with top n recommendations per user")
            top_n_recs = self.get_top_n_recommendation(all_predictions, top_n)

            print("Evaluating with hit-rate....")
            hit_rate = self.hit_rate(top_n_recs, lpo_predictions)
        return metrics

    @staticmethod
    def get_top_n_recommendation(predictions, n=6):
        top_n = defaultdict(list)
        for user_id, item_id, actual_rating, estimated_rating, _ in predictions:
            top_n[int(user_id)].append((int(item_id), estimated_rating))

        for user_id, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[user_id] = user_ratings[:n]
        return top_n

    def get_top_n_recommendation_for_user(self, user_id=None):
        self.algorithm.fit(self.train_set)
        test_set = self.get_anti_testset_for_user(user_id)
        predictions = self.algorithm.test(test_set)
        return self.get_top_n_recommendation(predictions)

    @staticmethod
    def precision_recall_at_k(predictions, k=10, threshold=0.1):
        # This is modified from examples in package to accommodate for content-based
        # Threshold is used to either accept or reject a rating within {x} deviation from actual rating

        # For each user, get all tuples of estimated and true ratings
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum(bool(true_r) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((abs(est-true_r)/true_r <= threshold) for (est, true_r) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum((bool(true_r) and (abs(est-true_r)/true_r <= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

    def get_anti_testset_for_user(self, user_id=None):
        fill = self.train_set.global_mean
        anti_testset = []
        if not user_id:
            uid = np.random.randint(0, self.train_set.n_users)
        else:
            uid = self.train_set.to_inner_uid(user_id)
        user_items = set([j for (j, _) in self.train_set.ur[uid]])
        anti_testset += [(self.train_set.to_raw_uid(uid), self.train_set.to_raw_iid(i), fill) for
                         i in self.train_set.all_items() if
                         i not in user_items]
        return anti_testset


class LeavePortionOut(object):
    def __init__(self, percent=0.1, random_state=None, rating_threshold=0):
        self.percent = percent
        self.random_state = random_state
        self.rating_threshold = rating_threshold

    def split(self, data):
        user_ratings = defaultdict(list)
        for uid, iid, r_ui, _ in data.raw_ratings:
            user_ratings[uid].append((uid, iid, r_ui, None))

        # For each user, select x percent of items to be include in test_set and exclude in train_set
        raw_trainset, raw_testset = [], []
        for uid, ratings in iteritems(user_ratings):
            if len(ratings) > self.rating_threshold:
                leave_these_out_idx = np.random.choice(len(ratings), int(round(len(ratings)*self.percent)))
                for idx, rating in enumerate(ratings):
                    if idx in leave_these_out_idx:
                        raw_testset.append(ratings[idx])
                    else:
                        raw_trainset.append(ratings[idx])

        if not raw_trainset:
            raise ValueError('Could not build any trainset. Probaby rating_threshold is too high!')
        train_set = data.construct_trainset(raw_trainset)
        test_set = data.construct_testset(raw_testset)

        return train_set, test_set
