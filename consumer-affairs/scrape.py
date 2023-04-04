import scrapy
from scrapy.shell import inspect_response
from scrapy.utils.response import open_in_browser
from scrapy import FormRequest
from scrapy.crawler import CrawlerProcess
import pandas as pd
from pathlib import Path

from datetime import datetime as dt
from datetime import timedelta
from scrapy.utils.project import get_project_settings

FILE_PATH = Path(__file__).resolve()
PARENT_FOLDER = FILE_PATH.parents[0]
RAW_DATA_FOLDER = PARENT_FOLDER / "data" / "raw"
RAW_DATA_FOLDER.mkdir(exist_ok=True, parents=True)

def get_date_ranges(
    from_date: str ,
    to_date: str
):
  """
  Function that provide date generator for the between range for a date
  """
  start = dt.strptime(from_date, '%Y-%m-%d')
  end = dt.strptime(to_date, '%Y-%m-%d')
  step = timedelta(days=1)

  while start <= end:
     yield start.date().strftime("%d/%m/%Y")
     start += step
     

class PricesSpider(scrapy.Spider):
    name = 'prices'
    start_urls = ['https://fcainfoweb.nic.in/reports/report_menu_web.aspx']

    def __init__(self):
      # self.start_time = start_time
      # self.end_time = end_time
      self.start_time, self.end_time = '2018-01-01','2018-12-31'

    def parse(self, response):
        data = {
            'ctl00_MainContent_ToolkitScriptManager1_HiddenField': '',
            'ctl00$MainContent$Ddl_Rpt_type': 'Wholesale',
            'ctl00$MainContent$ddl_Language': 'English',
            'ctl00$MainContent$Rbl_Rpt_type': 'Price report',
        }
        yield FormRequest.from_response(
            response,
            formdata=data, 
            callback=self.step2,
            meta = {
                "dont_filter" : True
            },
            dont_filter=True
        )

    def step2(self, response):
        data = {
            'ctl00$MainContent$Ddl_Rpt_Option0': 'Daily Prices'
        }
        yield FormRequest.from_response(response, formdata=data, callback=self.step3)
    def step3(self, response):
        for search_date in get_date_ranges(self.start_time, self.end_time):
          data = {
              'ctl00$MainContent$Txt_FrmDate': search_date,
              'ctl00$MainContent$btn_getdata1': 'Get Data'
          }
          yield FormRequest.from_response(
              response, 
              formdata=data, 
              callback=self.parse_table,
              meta = {
                "date" : dt.strptime(search_date, '%d/%m/%Y').strftime("%Y-%m-%d"),
                "dont_filter" : True
              },
                dont_filter=True
          )

    def parse_table(self, response):
        dfs = pd.read_html(response.text, attrs = {'align':'Center'})
        # inspect_response(response, self)
        date = response.meta["date"].replace("/","-")
        for i,df in enumerate(dfs):
            date_folder = RAW_DATA_FOLDER.joinpath(date)
            date_folder.mkdir(parents=True, exist_ok=True)
            df.to_csv(f'{date_folder}/data_{i}.csv', index = False)
            self.logger.info(f"Saved file {date}/data_{i}.csv")

def main():
    settings = get_project_settings()
    settings.set("CUSTOM_SETTING", "Super Custom Setting")
    settings.update(
        {
            "CONCURRENT_REQUESTS": 20,
            "ROBOTSTXT_OBEY": False,
            "BOT_NAME": "prices",
        }
    )
    
    crawler = CrawlerProcess(settings)
    crawler.crawl(PricesSpider)
    crawler.start()

if __name__ == "__main__" :
  main()