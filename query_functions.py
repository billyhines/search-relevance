import pandas as pd

def query_elasticsearch_hybrid(
    es_client,
    index_name,
    search_text=None,
    search_vector=None,
    num_results=10,
    boost_values=None,
):
    """Queries Elasticsearch with search_text and or a search_vector with optional boost values.

    Args:
        es_client: An Elasticsearch client object.
        index_name: The name of the index to query.
        search_text: The text to search for.
        search_vector: The vector to search with.
        num_results: The maximum number of results to return (default 10).
        boost_values: A dictionary containing the boost values for title, description, and attributes.

    Returns:
        The search results from Elasticsearch.
    """

    # Default boost values
    boost_values = boost_values or {
        "title_boost": 1,
        "description_boost": 1,
        "attributes_boost": 1,
        "vector_boost": 1,
    }

    should_clauses = []

    # Text match clauses:
    if search_text is not None:
        text_match_clauses = [
            {
                "match": {
                    "product_title": {
                        "query": search_text,
                        "boost": boost_values["title_boost"],
                    }
                }
            },
            {
                "match": {
                    "product_description": {
                        "query": search_text,
                        "boost": boost_values["description_boost"],
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
                                "boost": boost_values["attributes_boost"],
                            }
                        }
                    },
                }
            },
        ]

        should_clauses.extend(text_match_clauses)
    
    # Vector match clauses:
    if search_vector is not None:
        vetor_match_clause = {
                        "knn": {
                            "field": "product_vector",
                            "query_vector": search_vector,
                            "num_candidates": 150,
                            "boost": boost_values["vector_boost"],
                        }
                    }
        should_clauses.append(vetor_match_clause)

    query_body = {
        "size": num_results,
        "query": {
            "bool": {
                "should": should_clauses
            }
        },
    }

    results = es_client.search(index=index_name, body=query_body)
    return results