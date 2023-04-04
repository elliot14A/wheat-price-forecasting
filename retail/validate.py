from pandera.dtypes import  Float32, String
from typing import Union
import pandera as pa
from pandera import typing as typ
import pandas as pd
from pprint import pprint

from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[1]
RAW_FILE_PATH = PROJECT_DIR / "dataset.xlsx"

df = pd.read_excel(RAW_FILE_PATH)
df.drop(columns=["Unnamed: 0"], inplace=True)

class PriceValidation(pa.SchemaModel):
    state  : typ.Series[String] = pa.Field()
    centre : typ.Series[String] = pa.Field()
    variety  : typ.Series[String] = pa.Field()
    unit : typ.Series[String] = pa.Field()
    date : typ.Series[pd.Timestamp] = pa.Field(nullable= False)
    value  : typ.Series[Float32] = pa.Field(nullable= True)
    month : typ.Series[String] = pa.Field()
    year : typ.Series[String] = pa.Field()
    
    class Config():
        strict = True
        coerce = True
        ordered = True
        
PriceValidation.validate(df)
print("Validation Successful")