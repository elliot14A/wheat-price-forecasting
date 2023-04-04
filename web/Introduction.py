from pathlib import Path
from PIL import Image
import streamlit as st
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[0]
IMAGE_PATH = PROJECT_DIR / "introduction_1_image.jpg"

st.set_page_config(page_title="wheat-price-forecasting", layout="wide")


@st.cache_resource
def load_image():
    image = Image.open(str(IMAGE_PATH))
    return image


class Page:
    def __init__(self, title):
        self.title = title

    def header(self):
        st.title(self.title)
        st.markdown('---')

    def section(self, title, content):
        st.markdown(f'**{title}**', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:18px;">{content}</p>', unsafe_allow_html=True)
        st.markdown('---')


page = Page('Wheat Price Forecasting')

with st.container():
    _, image_col, _ = st.columns([1, 4, 1])

    with image_col:
        st.image(load_image(), use_column_width=True)

page.header()

page.section('How many farmers are relying on Food Prices?',
             'India holds the record for second-largest agricultural land in the world, with around 58% of the Indian Population depends on agriculture for their livelihood.')

page.section('What bad price on crops leads to in Farmers life?',
             'Farmers face significant challenges when the crop prices fall too low (below the cost of production) and has a huge impact on their livelihoods. This can be challenging for small-scale farmer who completely rely on agriculture for their livelihoods. They are forced to sell their crops at very low price which leave them in a loss and struggle to meet the necessities like enough food, healthcare.')

page.section('Govt Initiatives in providing better MSP to farmers?',
             'The government has initiated few number schemes aiming to provide better Minimum Support Prices (MSP) to farmers which include providing subsidies for fertilizer and irrigation and investing in rural infrastructure. Government has implemented several measures to ensure that farmers receive better MSPs for their crops. One of the key initiatives is the Pradhan Mantri Annadata Aay SanraksHan Abhiyan (PM-AASHA), which was launched in 2018 to provide a solution for the MSP and procurement of crops. In addition, the government has increased the MSPs for various crops, including wheat, paddy, and pulses, and has also extended the procurement of crops to more states and regions.')

page.section('What is the Role of Retail Markets in Farmers?',
             'In addition to government support, Retail markets play a crucial role in connecting farmers with consumers and providing them an outlet to sell their produce directly to consumers which allows farmers to skip the intermediaries and receive a higher price for their goods. Retail markets can help to support local food systems and promote sustainable agriculture by encouraging the production and consumption of locally grown foods.')
