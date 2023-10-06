# Dataset

The NSF Research Awards Abstracts dataset comprises several paper abstracts, one per file, that were furnished by the NSF (National Science Foundation).

For this especific project we used [a sample from 2020](https://www.nsf.gov/awardsearch/download?DownloadFileName=2020&All=true).

I start by creating a single CSV file with only the title, abstract and identifier.


```python
import xml.etree.ElementTree as Xet 
import pandas as pd
import os

input_directory: str = os.path.join('data', 'inputs')
filepaths = next(os.walk(input_directory), (None, None, []))[2]  # [] if no file

data = []

for filepath in filepaths:
    xmlparse = Xet.parse(os.path.join(input_directory, filepath)) 
    root = xmlparse.getroot() 
    for tag in root:
        data.append({
            "award_id": tag.find("AwardID").text,
            "title": tag.find("AwardTitle").text,
            "abstract": tag.find("AbstractNarration").text
        })

abstracts_df = pd.DataFrame(data) 

# Writing dataframe to csv 
abstracts_df.to_csv(os.path.join('data', 'processed', 'abstracts.csv'), index=False)
```

After trying to get the embeddings from the OpenAI API, I found the [rate limits](https://platform.openai.com/account/rate-limits) didn't allow me to continue working with their API. So, I lost the opportunity of [applying prompts to identify the topics in the clusters](https://cookbook.openai.com/examples/clustering)


```python
'''
# imports
import tiktoken

from openai.embeddings_utils import get_embedding


# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
abstracts_df["n_tokens"] = abstracts_df.abstract.apply(lambda x: len(encoding.encode(str(x))))
abstracts_df = abstracts_df[abstracts_df.n_tokens <= max_tokens].copy()
print(f'# of valid abstracts: {len(abstracts_df)}')

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
# This may take a few hours
import time

openai_embeddings_filepath = os.path.join('data', 'processed', 'openai_embeddings.csv')

for item in abstracts_df.to_dict('records'):
    print(f'Processing award {item["award_id"]}')
    try:
        openai_embeddings = get_embedding(item['abstract'], engine=embedding_model)
    except Exception as e:
        print(f'Retrying after finding an error {str(e)}')
        time.sleep(60)
        openai_embeddings = get_embedding(item['abstract'], engine=embedding_model)
    # Append to file using the write() method
    with open(openai_embeddings_filepath, 'a') as f:
        f.write(f'{item["award_id"]},{openai_embeddings}\n')
    time.sleep(3)
"""
```




    '\n# imports\nimport tiktoken\n\nfrom openai.embeddings_utils import get_embedding\n\n\n# embedding model parameters\nembedding_model = "text-embedding-ada-002"\nembedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002\nmax_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191\n\nencoding = tiktoken.get_encoding(embedding_encoding)\n\n# omit reviews that are too long to embed\nabstracts_df["n_tokens"] = abstracts_df.abstract.apply(lambda x: len(encoding.encode(str(x))))\nabstracts_df = abstracts_df[abstracts_df.n_tokens <= max_tokens].copy()\nprint(f\'# of valid abstracts: {len(abstracts_df)}\')\n\n# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage\n# This may take a few hours\nimport time\n\nopenai_embeddings_filepath = os.path.join(\'data\', \'processed\', \'openai_embeddings.csv\')\n\nfor item in abstracts_df.to_dict(\'records\'):\n    print(f\'Processing award {item["award_id"]}\')\n    try:\n        openai_embeddings = get_embedding(item[\'abstract\'], engine=embedding_model)\n    except Exception as e:\n        print(f\'Retrying after finding an error {str(e)}\')\n        time.sleep(60)\n        openai_embeddings = get_embedding(item[\'abstract\'], engine=embedding_model)\n    # Append to file using the write() method\n    with open(openai_embeddings_filepath, \'a\') as f:\n        f.write(f\'{item["award_id"]},{openai_embeddings}\n\')\n    time.sleep(3)\n'




