# Description
To Analyze customer behavior in buying groceries product.
The data here is from IRI.

#### Requirements

IRI Data is accessible via UChicago portal. You will need to be granted NDA to access this data.
 
#### Extract or have the following files into 'IRIData' folder:

* parsed stub files 2012.zip
* Year12.zip
* ads demos12.csv

#### Install necessary packages

```bash
pip install -r requirments.txt
```

#### Pipeline explanation:

1. Files from raw IRI data are pre-processed in `clean_data.py`, by running this script first.

    Then save into perspective `cleaned_data` folder.
    
    This will make it easier and faster for doing data exploratory and feeding into model.
    
2. The main process is in `process_data.py`. This script calls:

    a. Vectoring data into ready-to-feed format, using `vectorize_data` in `process_data.py`

        New data format will be:
        
        ```
        ------------------------------ 
        | user_id | item_id | rating |
        ------------------------------
        ```
        
      where rating is total number of purchases per user_id, item_id
        
      And
      
        ```
        ----------------------------------------------- 
        | item_id | feature_a | feature_b | feature_c |
        -----------------------------------------------
        ```
    b. Split data into train, test sets. Then using `ContentKNNAlgo` in `content_based.py` to fit and compute similarities.
        
        i. To save computation time, after computing item cosine similarities, it'll be saved as `item_similarities.npy`
            and loaded when being used.
            Thus, the first time model fit data will take a while, but after that, model will process faster.
        
        ii. The algorithm has method to get top n similar items, given item ID. 
            Or top n recommended items, given userID.
    
    c. Model predictions are validated in `evaluation.py` (in process)
        Methods of evaluations:
    
        i. Precision, Recall, F1 at k

        ii. Stimulate users behavior
        
