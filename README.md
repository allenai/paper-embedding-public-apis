# paper-embedding-public-apis

Collection of public APIs for embedding scientific papers.

Currently supported embedding methods:
- [SPECTER](#specter)

## SPECTER

Currently we support a single public endpoint for creating paper embeddings from papers' titles and abstracts. Future APIs may follow a similar setup. Please note that the URL used here is subject to change once we add a more permanent sub-domain for this effort.

Requirements:

* Send a flat JSON array where objects have the required attributes for paper_id, title, abstract
* Any additional attributes sent in JSON are ignored
* Do not send batches of more than 16 papers at a time (you will receive a 422 HTTP response)

Note that "paper_id" can be any string value and is only used to map to the generated embedding in the result.

#### Python example (Python 3)

```python
from typing import Dict, List
import json

import requests


URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
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

#### Citation

If using SPECTER embeddings, please cite the upcoming ACL paper:

```
@inproceedings{specter_cohan_2020,
    title = "{SPECTER: Document-level Representation Learning using Citation-informed Transformers}",
    author = "Cohan, Arman and
      Feldman, Sergey and
      Beltagy, Iz  and
      Downey, Doug and
      Weld, Daniel",
    booktitle = "ACL",
    year = "2020",
}
```

## Support / Questions

Please feel free to submit an issue directly on this repository and please make this your first course of action in case of any issues or errors. Thanks!

## Deploying

Code for managing this deployment on AWS can be found [here](https://github.com/allenai/s2-cdk-sagemaker-apis).
