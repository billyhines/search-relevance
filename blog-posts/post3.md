# Introduction

Welcome to the third post in our search relevance series! In our previous posts, we explored text-based search and vector-based search separately. We saw how each approach has its strengths: text search excels at exact matches and handling specific terms, while vector search captures semantic meaning and handles synonyms and misspellings well. In this post, we'll combine these approaches to create a hybrid search system that leverages the best of both worlds.

Our hybrid approach will aim to:

1. Combine the precision of text-based search with the semantic understanding of vector-based search.
2. Improve our ability to handle a wider range of queries, including those with misspellings or without exact matches.
3. Further enhance our search relevance metrics.

We'll cover the following key topics:

1. Implementing a basic hybrid search by combining text and vector queries
2. Tuning the hybrid search through boosting
3. Evaluating the performance of our hybrid approach
4. Exploring the advantages of hybrid search, particularly for handling edge cases

Let's dive in and see how we can create a more robust and effective search system through hybridization.

# Running Hybrid Searches

With Elasticsearch's kNN query, we can simply add the kNN matching clause into the list of text matches that we had originally used for our text search. One thing to note is that the text match scores are often orders of magnitude larger than the vector match scores. In order to give the vector match a fair chance in the final rankings, we will need to boost their scores. Similar to how we tuned the text scores, we will iterate over boost values for the vector scores in search of the highest-ranking evaluation metrics.

We can create our new hybrid search queries in the following manner:

```python
python
Copy
query_body = {
    "size": num_results,
    "query": {
        "bool": {
            "should": [
                {
                    "match": {
                        "product_title": {
                            "query": search_text,
                        }
                    }
                },
                {
                    "match": {
                        "product_description": {
                            "query": search_text,
                        }
                    }
                },
                {
                    "nested": {
                        "path": "product_attributes",
                        "query": {
                            "match": {
                                "product_attributes.name_value": {
                                    "query": search_text,
                                }
                            }
                        }
                    }
                },
                {
                    "knn": {
                        "field": "product_vector",
                        "query_vector": search_vector,
                        "num_candidates": 50,
                    }
                }
            ]
        }
    }
}

```

By tuning our vector match boosts, we're effectively able to balance the quality of the text matches between the original text and the products with the semantic relationship captured by the vector embeddings.

## Hybrid Search Results

Let’s take a look at a sample query in our three methods so far. The user query in this case is “hanging shelves”

**Text**

| Position | Score | Product ID | Product Title | Relevance |
| --- | --- | --- | --- | --- |
| 1 | 115.4625 | 107182 | Ameriwood Wardrobe Storage Closet with Hanging Rod and 2-Shelves in American Cherry | 2.67 |
| 2 | 87.12547 | 117672 | Whitmor 19.50 in. x 45.38 in. x 68.00 in. Double Rod Closet Shelves |  |
| 3 | 84.42163 | 174420 | 4-Shelves Tier Pole Caddy in Bronze |  |
| 4 | 83.25287 | 106280 | Rolling Shelves 17 in. Express Pullout Shelf |  |
| 5 | 83.25287 | 114168 | Rolling Shelves 21 in. Express Pullout Shelf |  |
| 6 | 81.89681 | 116589 | Zenith Premium Bathtub and Shower Pole Caddy with 4 Shelves in White |  |
| 7 | 80.95274 | 193107 | Stack-On 42 in. DIY Workbench with Full Length Steel Shelves |  |
| 8 | 80.55455 | 192343 | Command Picture Hanging Solution Kit |  |
| 9 | 80.42712 | 190110 | Martha Stewart Living 24 in. Espresso Shelves (2-Pack) |  |
| 10 | 80.26959 | 181845 | Fresca Allier 16 in. W Bathroom Linen Cabinet with 2 Glass Shelves in White |  |

The top result here for the “Ameriwood Wardrobe Storage Closet with Hanging Rod and 2-Shelves” comes in with a relatively high relevance score of 2.67 (out of three). The rest of the results seem to be for mostly shelves and are missing our “hanging” component.

**Vector**

| Position | Score | Product ID | Product Title | Relevance |
| --- | --- | --- | --- | --- |
| 1 | 0.702164 | 128711 | Honey-Can-Do 8-Shelf PEVA hanging organizer | 3 |
| 2 | 0.700011 | 104901 | 4D Concepts Hanging Wall Corner Shelf Storage | 3 |
| 3 | 0.699165 | 197453 | Martha Stewart Living Solutions 70 in. Silhouette 2-entryway Shelf with Hooks |  |
| 4 | 0.699132 | 119985 | Design House 12 in. x 10-3/16 in. White Shelf-Hanging Rod Bracket |  |
| 5 | 0.699011 | 139090 | 4D Concepts Hanging Wall Corner Shelf Storage | 2.67 |
| 6 | 0.696995 | 134901 | Prepac 36 in. W Hanging Entryway Shelf | 3 |
| 7 | 0.696304 | 136407 | Prepac 60 in. Wall-Mounted Coat Rack in White |  |
| 8 | 0.696301 | 188206 | New Age Industrial 15 in. D x 48 in. L 12-Gauge Aluminum Wall Shelf |  |
| 9 | 0.695971 | 136535 | Prepac 48.5 in. x 19.25 in. Floating Entryway Shelf and Coat Rack in Black |  |
| 10 | 0.695865 | 161429 | Houseworks 34 in. x 5-1/4 in. Unfinished Wood Decor Shelf with Pegs |  |

Our vector query is able to embed the “hanging shelves” into a single representation, and because of that is able to return more products that look like a hanging shelf solution.

**Hybrid**

| Position | Score | Product ID | Product Title | Relevance |
| --- | --- | --- | --- | --- |
| 1 | 115.4625 | 107182 | Ameriwood Wardrobe Storage Closet with Hanging Rod and 2-Shelves in American Cherry | 2.67 |
| 2 | 110.6256 | 163204 | ClosetMaid 54 in. Canteen 8-Shelf Hanging Organizer |  |
| 3 | 110.371 | 119437 | ClosetMaid 54 in. Mocha 8-Shelf Hanging Organizer | 2.67 |
| 4 | 109.9637 | 186249 | ClosetMaid 54 in. Canteen 10-Shelf Hanging Organizer |  |
| 5 | 109.8537 | 141074 | ClosetMaid 54 in. Mocha 10-Shelf Hanging Organizer | 3 |
| 6 | 107.4734 | 124425 | ClosetMaid 24 in. White Versatile Hanging Shelf | 3 |
| 7 | 106.2367 | 109858 | ClosetMaid 24 in. Hanging Wire Shelf | 3 |
| 8 | 104.0418 | 154365 | Home Decorators Collection 2-Shelves and Towel Rack in Chrome |  |
| 9 | 103.2512 | 128711 | Honey-Can-Do 8-Shelf PEVA hanging organizer | 3 |
| 10 | 101.6648 | 212563 | Martha Stewart Living Garage 6 in. H x 24 in. W White Metal Shelves |  |

Our hybrid solution shows us that we really can get the best of both worlds. The top result is our top result from the text search. The 9th result is our top result from the vector search. In between these two are four other high-relevance products that weren’t in the top ten results of our text or vector searches alone.

# Evaluation

We’ve run a lot of additional queries up to this point, let’s check back into the evaluation scores to see how we’re shaping up. Recall from the first post that we will be using Mean Recipricol Rank, Mean Average Precision, and Normalized Discounted Cumulative Gain.

| Name | MRR | MAP | NDCG | Run Time |
| --- | --- | --- | --- | --- |
| textsearch | 0.261 | 0.113 | 0.170 | 178.5 |
| textsearch_boosted | 0.318 | 0.149 | 0.218 | 207.4 |
| vectorsearch | 0.331 | 0.159 | 0.237 | 509.6 |
| vectorsearch_multifield | 0.241 | 0.097 | 0.156 | 623.0 |
| vectorsearch_multifield_tuned | 0.255 | 0.106 | 0.168 | 662.4 |
| hybrid | 0.325 | 0.163 | 0.238 | 662.6 |
| hybrid_boosted | 0.342 | 0.170 | 0.251 | 716.3 |
| Increase over Text | 7.7% | 14.1% | 15.5% | 245.4% |
| Increase over Vector | 3.4% | 6.9% | 6.2% | 40.6% |

We can see that our tuned hybrid really does give us that “best of both worlds” and the best performance across all three metrics. By tuning our hybrid search we were able to make a 7%-16% gain across all three of our metrics from the tuned text search and a 3%-7% gain across all three of our metrics from the vector search. It worth noting, however, that we take a 245% hit on the time it takes to run these queries over the basic text query.

# Conclusion

In this post, we combined the additional capability of vector based search to our text searches across the search relevance dataset. We tuned our hybrid queries using a similar method to how we tuned our text search.

This added capability allowed us to surface better results across our set of queries and even surface results where there weren’t any previously available due to the limitation of text based search.

While Hybrid approaches offer clear advantages in terms of relevance and semantic understanding, it's once again important to note that these techniques often come with increased computational complexity and indexing costs. The trade-off between performance and relevance should be carefully evaluated based on the specific requirements of the search application.

In the next post, we'll take our exploration of hybrid search one step further by exploring an additional ranking method called Reciprocal Rank Fusion.

If you want to dive deeper into the code, the notebooks for all of the work above can be found here:
* [3-hybrid.ipynb](https://github.com/billyhines/search-relevance/blob/main/3-hybrid.ipynb)