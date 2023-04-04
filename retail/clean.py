import pandas as pd
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[1]
DATA_FOLDER = PROJECT_DIR / "raw"
RAW_FILE_PATH = DATA_FOLDER / "Wheat1.csv"

df = pd.read_csv(RAW_FILE_PATH)

# Melting operation to convert wide column to long column
df_melt = df.melt(
    id_vars = list(df.columns[:4]),
    value_vars = list(df.columns[4:]),
    var_name = "date"
)

# convert all column headers to lower case
df_melt.columns = [col.lower().strip() for col in df_melt.columns]

# Split year and month to separate column to make distinction for future use
splitted_date_series = df_melt["date"].str.split(" ", expand= True)
df_melt["month"], df_melt["year"] = splitted_date_series[0], splitted_date_series[1]

# Clean unit column 
df_melt["unit"] = df_melt["unit"].str.strip(".")

#converting data column format to dd/mm/yyy
df_melt.date = df_melt.date.str.title()
df_melt['date'] = pd.to_datetime(df_melt['date'])

# Method 1 : We have chosen the centres with 100% values. 
# No imputation required.
groupby_centre = (df_melt.groupby('centre').count())
filtered_df_100 = groupby_centre[groupby_centre['value'] == 144]
centre_list_100 = filtered_df_100.index.tolist()

df_final_100 = df_melt[df_melt['centre'].isin(centre_list_100)]
df_final_100.to_excel(DATA_FOLDER / "dataset.xlsx")