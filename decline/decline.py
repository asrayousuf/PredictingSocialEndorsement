import pickle
import numpy as np

from decline_preprocess import filter_by_review_count

with open('./data/ratings_by_business.pkl', 'rb') as f:
    ratings_by_business = pickle.load(f)

def find_businesses_with_decline(ratings_by_business):
    businesses_with_decline = {}
    for business_id, business_ratings in ratings_by_business.items():
        idx = len(business_ratings) // 4
        first_quarter_rating = np.mean([stars for stars, review_date in business_ratings[:idx]])
        last_quarter_rating = np.mean([stars for stars, review_date in business_ratings[-idx:]])
        if first_quarter_rating - last_quarter_rating >= 1:
            businesses_with_decline[business_id] = {
                'first_quarter': first_quarter_rating,
                'last_quarter': last_quarter_rating,
                'num_ratings': len(business_ratings),
            }
    businesses_with_decline = dict(sorted(
            businesses_with_decline.items(),
            key=lambda x: x[1]['first_quarter'] - x[1]['last_quarter'],
            reverse=True
        )
    )
    return businesses_with_decline

ratings_by_business = filter_by_review_count(ratings_by_business, 500)
businesses_with_decline = find_businesses_with_decline(ratings_by_business)
