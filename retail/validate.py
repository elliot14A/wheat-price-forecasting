from pandera.dtypes import Float32, String
import pandera as pa
from pandera import typing as typ
import pandas as pd


class PriceValidation(pa.SchemaModel):
    state: typ.Series[String] = pa.Field()
    centre: typ.Series[String] = pa.Field()
    variety: typ.Series[String] = pa.Field()
    unit: typ.Series[String] = pa.Field()
    date: typ.Series[pd.Timestamp] = pa.Field(nullable=False)
    value: typ.Series[Float32] = pa.Field(nullable=True)
    month: typ.Series[String] = pa.Field()
    year: typ.Series[String] = pa.Field()

    class Config():
        strict = True
        coerce = True
        ordered = True
