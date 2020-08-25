import os
import pandas as pd
import numpy as np
import pprint

from IRI.clean_data import extract_data
from IRI.constants import BASE_PATH, CATEGORY_MAPPING
from IRI.content_based import ContentKNNAlgo
from IRI.evaluation import Evaluator

from sklearn.preprocessing import MultiLabelBinarizer
from surprise import Dataset, Reader

pp = pprint.PrettyPrinter(indent=2, width=150)


def vectorize_data(users_items, items):
    users_items = users_items.groupby(['PANID', 'COLUPC'])['total_purchase'].sum().reset_index()

    users_items.rename(columns={'PANID': 'userID', 'COLUPC': 'itemID'}, inplace=True)
    users_items['total_purchase'] = np.ceil(users_items['total_purchase']).astype(int)

    # Normalize and Convert purchases into rating scale [1, 10]
    users_items['normalized_purchases'] = users_items.groupby('userID')['total_purchase'].transform(lambda x: x/x.sum())
    old_min = users_items['normalized_purchases'].min()
    users_items['rating'] = users_items['normalized_purchases'].apply(lambda x: round(((x-old_min) * (10-1) / 1) + 1))
    users_items = users_items[['userID', 'itemID', 'rating']]

    items.rename(columns={'COLUPC': 'itemID', 'PRODUCT TYPE': 'product_type'}, inplace=True)
    items['flavor_profile'] = items['flavor_profile'].astype(str)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(items['flavor_profile'].str.split(', '))
    mult_columns = pd.DataFrame(labels,
                                columns=mlb.classes_,
                                index=items.index).add_prefix('flavor_')
    items = pd.merge(items, mult_columns, how="left", left_index=True, right_index=True)
    items.drop(['FLAVOR/SCENT', 'flavor_profile', 'product_subtype'], axis=1, inplace=True)
    items = pd.get_dummies(items, columns=["product_type", "fat_level", "calorie_level", "sugar_level",
                                           "caffeine_level", "user_info"])

    items['category'] = items['category'].map(lambda x: CATEGORY_MAPPING[x])
    return users_items, items


def process_data():
    if not os.path.isdir(os.path.join(BASE_PATH, 'cleaned_data')):
        extract_data()

    original_users_items = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_data/panels_items.csv'))
    original_items = pd.read_csv(os.path.join(BASE_PATH, 'cleaned_data/items.csv'), low_memory=False)

    # items shape (182726, 945)
    # users_items shape (476808, 3)
    users_items, items = vectorize_data(original_users_items, original_items)

    # Scale is from 1 to 10
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(users_items, reader)

    algo = ContentKNNAlgo(items=items, k=10, verbose=False)

    evaluator = Evaluator(algo, data)
    evaluator.evaluate(None)

    # Get top 6 recommended items for 5 random users
    result = {}

    for _ in range(5):
        top_n_recoms = evaluator.get_top_n_recommendation_for_user()
        user_id = list(top_n_recoms.keys())[0]
        item_ids = [item_id for item_id, _ in top_n_recoms[user_id]]
        user_item = original_items.loc[original_items['itemID'].isin(item_ids)]
        user_res = user_item.loc[:, user_item.columns.isin(['category', 'product_type', 'FLAVOR/SCENT',
                                                            "fat_level", "calorie_level", "sugar_level",
                                                            "caffeine_level", "user_info"])]
        result[user_id] = []
        for item in list(user_res.values):
            result[user_id].append([item_spec for item_spec in item if not pd.isnull(item_spec)])
    print('Top 6 products for each 5 random users are:')
    pp.pprint(result)


if __name__ == "__main__":
    process_data()







