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
        boost_values: A dictionary containing the boost values for the fields.

    Returns:
        The search results from Elasticsearch.
    """

    # Default boost values
    boost_values = boost_values or {
        "title_boost": 1,
        "description_boost": 1,
        "attributes_boost": 1,
        "product_text_string_vector_boost": 1,
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
    if search_vector is not None and "product_text_string_vector_boost" in boost_values:
        vetor_match_clause = {
            "knn": {
                "field": "product_text_string_vector",
                "query_vector": search_vector,
                "num_candidates": 150,
                "boost": boost_values["product_text_string_vector_boost"],
            }
        }
        should_clauses.append(vetor_match_clause)

    if search_vector is not None and "product_title_vector_boost" in boost_values:
        vetor_match_clause = {
            "knn": {
                "field": "product_title_vector",
                "query_vector": search_vector,
                "num_candidates": 150,
                "boost": boost_values["product_title_vector_boost"],
            }
        }
        should_clauses.append(vetor_match_clause)

    if search_vector is not None and "product_description_vector_boost" in boost_values:
        vetor_match_clause = {
            "knn": {
                "field": "product_description_vector",
                "query_vector": search_vector,
                "num_candidates": 150,
                "boost": boost_values["product_description_vector_boost"],
            }
        }
        should_clauses.append(vetor_match_clause)

    if (
        search_vector is not None
        and "product_attributes_string_vector_boost" in boost_values
    ):
        vetor_match_clause = {
            "knn": {
                "field": "product_attributes_string_vector",
                "query_vector": search_vector,
                "num_candidates": 150,
                "boost": boost_values["product_attributes_string_vector_boost"],
            }
        }
        should_clauses.append(vetor_match_clause)

    query_body = {
        "size": num_results,
        "query": {"bool": {"should": should_clauses}},
    }

    results = es_client.search(index=index_name, body=query_body)
    return results


def query_elasticsearch_rrf(
    es_client,
    index_name,
    search_text=None,
    search_vector=None,
    num_results=10,
    num_query_results=50,
    k=60,
    boost_values=None,
):
    """Combines a vector search and a text search with RRF.

    Args:
        es_client: An Elasticsearch client object.
        index_name: The name of the index to query.
        search_text: The text to search for.
        search_vector: The vector to search with.
        num_results: The maximum number of results to return (default 10).
        num_results: The maximum number of results to return from each query type (default 50).
        k: the RRF ranking constant.
        boost_values: A dictionary containing the boost values for title, description, and attributes.


    Returns:
        The search a dataframe of results from Elasticsearch ranked by RRF.
    """

    # Default boost values
    boost_values = boost_values or {
        "title_boost": 1,
        "description_boost": 1,
        "attributes_boost": 1,
        "product_text_string_vector_boost": 1,
    }

    # Text search
    text_results = query_elasticsearch_hybrid(
        es_client,
        index_name,
        search_text=search_text,
        num_results=num_query_results,
        boost_values=boost_values,
    )
    text_hits = pd.DataFrame(text_results["hits"]["hits"])

    # Vector seach
    vector_results = query_elasticsearch_hybrid(
        es_client,
        index_name,
        search_vector=search_vector,
        num_results=num_query_results,
        boost_values=boost_values,
    )
    vector_hits = pd.DataFrame(vector_results["hits"]["hits"])

    # Combine with Recipricol Rank fUsIoN

    if len(text_hits) == 0:
        rrf_results = vector_hits
        rrf_results["rrf_score"] = 1 / (rrf_results["_score"].rank(ascending=False) + k)
        rrf_results["product_uid"] = [x["product_uid"] for x in rrf_results["_source"]]
    else:

        text_hits["score"] = 1 / (text_hits["_score"].rank(ascending=False) + k)
        vector_hits["score"] = 1 / (vector_hits["_score"].rank(ascending=False) + k)

        rrf_results = text_hits.merge(
            vector_hits, how="outer", left_on="_id", right_on="_id"
        )
        rrf_results["rrf_score"] = rrf_results["score_x"].fillna(0) + rrf_results[
            "score_y"
        ].fillna(0)

        rrf_results["source"] = rrf_results["_source_x"].combine_first(
            rrf_results["_source_y"]
        )
        rrf_results["product_uid"] = [x["product_uid"] for x in rrf_results["source"]]

    rrf_results.sort_values("rrf_score", inplace=True, ascending=False)
    rrf_results.reset_index(inplace=True, drop=True)

    return rrf_results[:num_results]


def query_elasticsearch_rrf_multi(
    es_client,
    index_name,
    search_text=None,
    search_vector=None,
    num_results=10,
    num_query_results=50,
    k=60,
    boost_values=None,
):
    """Combines a vector search and a text search with RRF. Allows for multiple vector search fields.

    Args:
        es_client: An Elasticsearch client object.
        index_name: The name of the index to query.
        search_text: The text to search for.
        search_vector: The vector to search with.
        num_results: The maximum number of results to return (default 10).
        num_results: The maximum number of results to return from each query type (default 50).
        k: the RRF ranking constant.
        boost_values: A dictionary containing the boost values for title, description, and attributes.


    Returns:
        The search a dataframe of results from Elasticsearch ranked by RRF.
    """

    # Default boost values
    boost_values = boost_values or {
        "title_boost": 1,
        "description_boost": 1,
        "attributes_boost": 1,
        "product_text_string_vector_boost": 1,
    }

    # Collect all our hits and the score column names
    hits_list = []
    hits_score_cols = []

    # Text search
    text_results = query_elasticsearch_hybrid(
        es_client,
        index_name,
        search_text=search_text,
        num_results=num_query_results,
        boost_values=boost_values,
    )
    text_hits = pd.DataFrame(text_results["hits"]["hits"])

    if len(text_hits) > 0:
        text_hits["product_uid"] = [x["product_uid"] for x in text_hits["_source"]]
        text_hits["text_score"] = 1 / (text_hits["_score"].rank(ascending=False) + k)
        hits_list.append(text_hits)
        hits_score_cols.append("text_score")

    # Vector seach
    for key in boost_values.keys():
        if "vector" in key:
            vector_boost_value = {key: boost_values[key]}

            vector_results = query_elasticsearch_hybrid(
                es_client,
                index_name,
                search_vector=search_vector,
                num_results=num_query_results,
                boost_values=vector_boost_value,
            )

            vector_hits = pd.DataFrame(vector_results["hits"]["hits"])
            vector_hits["product_uid"] = [
                x["product_uid"] for x in vector_hits["_source"]
            ]
            vector_hits[key + "_score"] = 1 / (
                vector_hits["_score"].rank(ascending=False) + k
            )
            vector_hits.rename(
                columns={
                    "_index": key + "_index",
                    "_id": key + "_id",
                    "_score": key + "_score",
                    "_source": key + "_source",
                },
                inplace=True,
            )

            hits_list.append(vector_hits)
            hits_score_cols.append(key + "_score")

    # Combine all hits
    all_hits = hits_list[0]
    if len(hits_list) > 1:
        for i in range(1, len(hits_list)):
            all_hits = all_hits.merge(hits_list[i], how="outer", on="product_uid")

    all_hits[hits_score_cols] = all_hits[hits_score_cols].fillna(0)
    all_hits["rrf_score"] = all_hits[hits_score_cols].sum(axis=1)

    all_hits.sort_values("rrf_score", inplace=True, ascending=False)
    all_hits.reset_index(inplace=True, drop=True)

    return all_hits[:num_results]
