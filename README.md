# paper-embedding-public-apis
Collection of public APIs for embedding scientific papers

## Specter embeddings

Currently we support a single public endpoint for creating paper embeddings from papers' titles and abstracts. Future APIs may follow a similar setup. Please note that the URL used here is subject to change once we add a more permanent sub-domain for this effort.

Requirements:

* Send a flat JSON array where objects have at least attributes for paper_id, title, abstract
* Do not send batches of more than 16 papers at a time (you will receive a 422 HTTP response)

Note that "paper_id" can be any string value and is only used to map to the generated embedding in the result.

Using the API (from Python):

```python
from typing import Dict, List
import json

import requests


# Current location of the embedding API (subject to change)
URL = "https://9yqruv40oc.execute-api.us-west-2.amazonaws.com/prod/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


SAMPLE_PAPERS = [
    {
        "paper_id": "A",
        "title": "Angiotensin-converting enzyme 2 is a functional receptor for the SARS coronavirus",
        "abstract": "Spike (S) proteins of coronaviruses ...",
    },
    {
        "paper_id": "B",
        "title": "Hospital outbreak of Middle East respiratory syndrome coronavirus",
        "abstract": "Between April 1 and May 23, 2013, a total of 23 cases of MERS-CoV ...",
    },
]


def embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}

    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(URL, json=chunk)

        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


if __name__ == "__main__":
    all_embeddings = embed(SAMPLE_PAPERS)

    # Prints { 'A': [4.089589595794678, ...], 'B': [-0.15814849734306335, ...] }
    print(all_embeddings)
```
