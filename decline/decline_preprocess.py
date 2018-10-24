import json
import pickle
from collections import defaultdict
from datetime import datetime

def filter_by_review_count(ratings_by_business, min_count):
    return {
        business_id: business_ratings
        for business_id, business_ratings in ratings_by_business.items()
        if len(business_ratings) >= min_count
    }

def parse_datestrings(ratings_by_business):
    return {
        business_id: [
            (stars, datetime.strptime(datestring, '%Y-%m-%d'))
            for stars, datestring in business_ratings
        ]
        for business_id, business_ratings in ratings_by_business.items()
    }

def sort_by_date(ratings_by_business):
    return {
        business_id: sorted(business_ratings, key=lambda x: x[1])
        for business_id, business_ratings in ratings_by_business.items()
    }

def main():
    ratings_by_business = defaultdict(list)

    with open('./data/yelp_academic_dataset_review.json', 'r') as f:
        for i, line in enumerate(f, start=1):
            if not (i % 10000):
                print(f'Parsed {i} lines')
            line_data = json.loads(line)
            ratings_by_business[line_data['business_id']].append(
                (line_data['stars'], line_data['date'])
            )

    print(f'Collected data for {len(ratings_by_business)} businesses')

    ratings_by_business = filter_by_review_count(ratings_by_business, 50)
    print(f'Filtered to {len(ratings_by_business)} businesses')

    ratings_by_business = parse_datestrings(ratings_by_business)
    print(f'Parsed datestrings')

    ratings_by_business = sort_by_date(ratings_by_business)
    print(f'Sorted by date')

    with open('./data/ratings_by_business.pkl', 'wb') as f:
        pickle.dump(ratings_by_business, f)

if __name__ == "__main__":
    main()
