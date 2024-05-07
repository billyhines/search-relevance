# Introduction

Welcome to the second post in our search relevance series! In the first post we explored our product search relevance dataset. We’ve defined our evaluation metrics and we established a baseline performance for text search out of the box. We then ran some optimizations to increase the performance of text search. In this post we’ll use the same framework but use it to create and run vector based searches.

A vector based search solution gives us a few advantages over a text based solution:

- The text embeddings can capture **semantic meaning** of words and phrases which can allow for matches that might not have the same text
- Text embeddings can handle **synonyms or even misspellings** which may be common in the short text queries of our dataset

The world of text embedding models has been rapidly evolving over the past few years. In a previous post I discussed creating text embeddings using Google’s Universal Sentence Encoder. These days (4/17/2024 for reference), it seems like a new text embedding model smashes the benchmarks every other day.

In this post, we’ll show how to embed text to create vector representation of both our queries and products. We’ll then show how to run a vector based query in Elasticsearch and compare the results to our previous text based results. Finally, we’ll combine the two methods to generate a hybrid of the results, hopefully giving us the best of both worlds!

- **Vector Embeddings**
    - Introduce the concept of vector embeddings and their role in representing text data.
    - Explain the advantages of vector embeddings for search tasks (e.g., capturing semantic similarities, handling misspellings, etc.).
    - Discuss the specific embedding model you'll be using (e.g., pre-trained models like BERT, sentence-transformers, etc.).
- **Generating Embeddings**
    - Walk through the process of generating vector embeddings for your product data (titles, descriptions, attributes).
    - Share any preprocessing steps, batching techniques, or optimizations you employed.
    - Discuss how you indexed the embeddings in Elasticsearch (or any other vector store you're using).

# Vector Embeddings

## Generating Embeddings

We’re going to first start off with an open source pretrained embedding model. We’ve selected the `snowflake-arctic-embed-m` model as it is the highest ranked models on retrieval tasks on the Massive Text Embedding Benchmark that has a size under 1 GB.

The next step is decide *what* to embed. It’s easy on the query side: for each query we’ll embed the text “Represent this sentence for searching relevant passages: “ and then the query text itself.

On the product side of things, we have some more options. We could embed titles, descriptions, attributes or any combination of these three. To start with, we’ll embed a concatenation of all three.

We can create these embeddings relatively simply using the `sentence-transformers` package as we show below for embedding the queries:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m", trust_remote_code=True)

prompt = 'Represent this sentence for searching relevant passages: '
df_queries_docs = [prompt + str(x) for x in df_queries['search_term']]
query_embeddings = model.encode(df_queries_docs)
```

## Indexing Embeddings

Elasticsearch has the ability to store and then run queries against vectors, but we need to make changes to our mapping to allow for the vector storage. We’ll set up our mapping like this:

```python
# Initialize the index

index_name = 'products-embeddings'

mapping = {
    "properties": {
        "product_uid": {
            "type": "integer"
        },
        "product_title": {
            "type": "text"
        },
        "product_description": {
            "type": "text"
        },
        "product_attributes": {
            "type": "nested",
            "properties": {
                "name": {
                    "type": "text"
                },
                "value": {
                    "type": "text"
                },
                "name_value": {
                    "type": "text"
                },
            }
        },
        "product_vector": {
            "type": "dense_vector",
            "dims": 768
        },
        "query_scores": {
            "type": "nested",
            "properties": {
                "query_id": {
                    "type": "text"
                },
                "relevance": {
                    "type": "float"
                },
            }
        }
    }
}

es.indices.create(index=index_name, mappings=mapping)
```

## Running Vector Searches

Finally, we need to make some changes to our query to run a vector search of our query vector against our product vectors:

```python
query_body = {
    "size": num_results,
    "query": {
        "bool": {
            "should": [
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

To run this query, Elasticsearch first uses an Approximate Nearest Neighbors algorithm to search across the indexed vectors and pull the “num_candidates” number of products that should be closest to the “search_vector”. Once those initial candidates are returned, the similarity between vectors is calculated and the documents are ranked according to these similarity values.

Let’s take a look at some sample results. First consider the results for the text query of "real flame gel fuel”:

| position | product_title | score | relevance |
| --- | --- | --- | --- |
| 1 | Real Flame Porter 50 in. Ventless Gel Fuel Fireplace in Walnut | 295.2595 |  |
| 2 | Real Flame Chateau 41 in. Ventless Gel Fuel Fireplace in White | 294.4405 |  |
| 3 | Real Flame Ashley 48 in. Gel Fuel Fireplace in Mahogany | 292.371 |  |
| 4 | Real Flame Ashley 48 in. Gel Fuel Fireplace in White | 290.451 |  |
| 5 | Real Flame Ashley 48 in. Gel Fuel Fireplace in Blackwash | 290.451 |  |
| 6 | Real Flame Chateau 41 in. Corner Ventless Gel Fuel Fireplace in Espresso | 286.5626 |  |
| 7 | Real Flame Chateau 41 in. Corner Ventless Gel Fuel Fireplace in White | 286.5626 | 2.33 |
| 8 | Real Flame Silverton 48 in. Gel Fuel Fireplace in White | 280.7104 |  |
| 9 | Real Flame Chateau 41 in. Corner Ventless Gel Fuel Fireplace in Dark Walnut | 279.9077 |  |
| 10 | Real Flame 15 in. 2-Can Outdoor Gel Fuel Conversion Kit in Oak | 279.469 |  |

Many of the results in this list have the words “gel fuel”, but are actually full fireplaces that utilize the gel fuel. However, our semantic vector based search is able to more closely infer that the query is looking for the fuel itself:

| position | product_title | score | relevance |
| --- | --- | --- | --- |
| 1 | Real Flame 13 oz. 18.5 lb. Gel Fuel Cans (16-Pack) | 0.809501 | 3 |
| 2 | Real Flame 13 oz. 24 lb. Gel Fuel Cans (24-Pack) | 0.804812 | 3 |
| 3 | Real Flame 13 oz. 15 lb. Gel Fuel Cans (12-Pack) | 0.803864 | 3 |
| 4 | Real Flame 24 in. Oak Convert to Gel Fireplace Logs | 0.776211 |  |
| 5 | Real Flame Fresno 72 in. Media Console Gel Fuel Fireplace in Black | 0.775063 | 2.67 |
| 6 | Real Flame Silverton 48 in. Gel Fuel Fireplace in White | 0.769471 |  |
| 7 | Real Flame Hawthorne 75 in. Media Console Gel Fuel Fireplace in Dark Espresso | 0.76487 |  |
| 8 | Real Flame 18 in. Oak Convert to Gel Fireplace Logs | 0.763937 | 2 |
| 9 | Real Flame Chateau 41 in. Ventless Gel Fuel Fireplace in Espresso | 0.76364 |  |
| 10 | Real Flame 15 in. 2-Can Outdoor Gel Fuel Conversion Kit in Oak | 0.763134 |  |

One interesting observation about this specific query is the inclusion of what appears to be a brand name, "Real Flame." In this dataset, we seem to have primarily "Real Flame" gel fuels, which has likely contributed to the relevant results in the vector based search. However, it's worth noting that a text embedding approach may not necessarily capture brand name matches as effectively as a text-based search. What if we could combine the best of both worlds?

## Running Hybrid Searches

With Elasticsearch’s kNN query, we can simply add the kNN matching clause into the list of text matches that we had originally used for our text search. One thing to note in the example above is that the text match scores are orders of magnitude larger than the vector match scores. In order to give the vector match a fighting chance in the final rankings, we will surely need to boost their scores. In a similar way we did when searching for the boost values in the text scores, we will iterate over boost values for the vector scores in search of the highest ranking evaluation metrics.

We can create our new hybrid search queries in the following manner:

```python
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
            ]
        }
    }
}
```

By tuning our vector match boosts, we’re effectively able to balance the quality of the text matches between the original text and the products as well as the semantic relationship between the two.

# Evaluation

We’ve run a lot of additional queries up to this point, let’s check back into the evaluation scores to see how we’re shaping up. Recall from the first post that we will be using Mean Recipricol Rank, Mean Average Precision, and Normalized Discounted Cumulative Gain.

| run_name | MRR | MAP | NDCG | run_time |
| --- | --- | --- | --- | --- |
| textsearch | 0.261 | 0.113 | 0.170 | 178.5 |
| textsearch_boosted | 0.318 | 0.149 | 0.218 | 207.4 |
| vectorsearch | 0.343 | 0.166 | 0.246 | 306.3 |
| hybrid | 0.326 | 0.163 | 0.239 | 439.5 |
| hybrid_boosted | 0.349 | 0.175 | 0.256 | 521.3 |
|  | 10% | 17% | 18% | 151% |

We can see that our tuned hybrid really does give us that “best of both worlds” and the best performance across all three metrics. By tuning our hybrid search we were able to make a 10%-18% gain across all three of our metrics from the tuned text search. It worth noting, however, that we take a 151% hit on the time it takes to run these queries.

One of the advantages we highlighted for vector-based search was its ability to handle misspellings and queries without exact text matches, which are common in short text queries like those in our dataset. There is indeed a small subset of 1.5% of our queries that did not return a text based result. However, with the introduction of the vector-based component in our hybrid search approach, we were able to surface relevant results for all of these previously unsuccessful queries. While the quality of these results may not be optimal in every case, the vector-based search ensures that we can now provide the user with at least some potentially relevant products, as opposed to an empty result set. This capability to handle queries with imperfect text matches is a key strength of incorporating vector embeddings into our search system.

# Conclusion

In this post, we added the additional capability of vector based search to our search relevance dataset. We were able to run strictly vector-based searches as well as a hybrid based search that we tuned in a similar manner to how we tuned our text search.

This added capability allowed us to surface better results across our set of queries and even surface results where there weren’t any previously available due to the limitation of text based search.

While vector search and hybrid approaches offer clear advantages in terms of relevance and semantic understanding, it's important to note that these techniques often come with increased computational complexity and indexing costs. The trade-off between performance and relevance should be carefully evaluated based on the specific requirements of the search application.

In the next post, we'll take our exploration of hybrid search one step further by exploring an additional ranking method called Reciprocal Rank Fusion.

If you want to dive deeper into the code, the notebooks for all of the work above can be found here:
* [2-vector-search.ipynb](https://github.com/billyhines/search-relevance/blob/main/2-vector-search.ipynb)
* [3-hybrid.ipynb](https://github.com/billyhines/search-relevance/blob/main/3-hybrid.ipynb)