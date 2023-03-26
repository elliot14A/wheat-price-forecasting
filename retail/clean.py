import click
import pandas as pd
import time
from validate import PriceValidation


class Cleaner:
    def __init__(self, input_path):
        self.input_path = input_path
        self.output_path = None
        self.df = pd.read_csv(input_path)

    def start(self):
        self.df = self.df.melt(
            id_vars=list(self.df.columns[:4]),
            value_vars=list(self.df.columns[4:]),
            var_name="date"
        )

        self.df.columns = [col.lower().strip() for col in self.df.columns]

        splitted_date_series = self.df["date"].str.split(" ", expand=True)
        self.df["month"], self.df["year"] = splitted_date_series[0], splitted_date_series[1]

        self.df["unit"] = self.df["unit"].str.strip(".")

        self.df.date = self.df.date.str.title()
        self.df['date'] = pd.to_datetime(self.df['date'])

        groupby_centre = (self.df.groupby('centre').count())
        filtered_df_100 = groupby_centre[groupby_centre['value'] == 144]
        self.centre_list_100 = filtered_df_100.index.tolist()

    def to_xlsx(self, output_path):
        self.df = self.df[self.df['centre'].isin(self.centre_list_100)]
        self.df.to_excel(output_path)
        self.output_path = output_path

    def validate(self):
        PriceValidation(self.df)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def cli(input_file, output_file):
    cleaner = Cleaner(input_file)
    click.echo("Reading input file...")
    time.sleep(1)
    cleaner.start()
    click.echo("Processing data...")
    time.sleep(1)
    cleaner.to_xlsx(output_file)
    click.echo("Done!")
    click.echo("Output file: {}".format(cleaner.output_path))
    click.echo("Validating output file...")
    cleaner.validate()
    click.echo("Done!")


if __name__ == "__main__":
    cli()
