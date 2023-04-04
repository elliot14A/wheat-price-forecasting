#%%
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime as dt


FILE_PATH = Path(__file__).resolve()
PARENT_FOLDER  = FILE_PATH.parents[0]
RAW_DATA_FOLDER = PARENT_FOLDER / "data" / "raw"
CLEANED_DATA_FOLDER = PARENT_FOLDER / "data" / "cleaned"

CLEANED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
#%%

logger=logging.getLogger(__file__)
#create a stream handler for logging out to console.
stream_handler = logging.StreamHandler()
logger.setLevel(logging.INFO)

#create a file handler for logging to file
file_handler = logging.FileHandler('clean.logs',mode='a')
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


#%%

#%%

# Delete all the row with more than half of the blank values 

# Remove row if centre is Zone or Price or Name in the string
# usually `Centre` column is the first column
# Use `filter` in column to get only wheat
# add year, month , week and date column properly to the dataset

master_df = []
logger.info(">>> "*30)
logger.info(f"Starting to clean the data : {dt.now()}")
logger.info("")

for file_idx, file_path in enumerate(RAW_DATA_FOLDER.glob("**/*.csv")):

    try:
        data : pd.DataFrame = pd.read_csv(file_path)
        centre_column = data.columns[0]
        non_centre_index = data[data[centre_column].str.contains("Zone|Price|Name", case = False)]
        date_timestamp = pd.Timestamp(file_path.parent.name)

        output_df = (data
        .drop(non_centre_index.index)
        .set_index(centre_column)
        .filter(regex = r"^[wheatWheat]*\s?$", axis = "columns")
        .assign(year = date_timestamp.year)
        .assign(month = date_timestamp.month)
        .assign(date = date_timestamp)
        .reset_index(drop = False)
        .set_axis(["centre", "value", "year", "month", "date"], axis = "columns")
        .reindex(["year", "month", "date", "centre", "value"], axis = "columns")
    )
        
    except Exception as e:
        logger.error(f"Error in file : {file_path}")
    else :
        master_df.append(output_df)
        logger.info(f"File Cleaned : {file_idx}")

# %%
# concat all the dataframes in the list
merged_df = pd.concat(master_df, axis = 0, ignore_index = True)

# sort the dataframe by date and centre
merged_df = merged_df.sort_values(["date", "centre"], ignore_index = True)

# convert to numeric column using to_numeric
merged_df["value"] = pd.to_numeric(merged_df["value"], errors = "coerce", downcast="integer")

# save the dataframe to csv
merged_df.to_csv(CLEANED_DATA_FOLDER/ "cleaned_data.csv", index = False)
# %%