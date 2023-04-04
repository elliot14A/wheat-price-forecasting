import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import pickle
from typing import Dict, List
import plotly.graph_objects as go

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[2]
MLRUNS = PROJECT_DIR / "mlruns" / "0"
HISTORIC_DATA = PROJECT_DIR / "dataset.xlsx"

st.set_page_config(page_title="wheat-price-prediction", layout="wide")


@st.cache_data
def load_historic_data(centre: str, variety: str):
    data = pd.read_excel(HISTORIC_DATA)
    data_filter = data[(data["centre"] == centre) & (data["variety"] == variety)]
    data_filter = data_filter[["date", "value"]]
    return data_filter


@st.cache_data
def load_historic_data_all_cols(centre: str, variety: str):
    data = pd.read_excel(HISTORIC_DATA)
    data_filter = data[(data["centre"] == centre) & (data["variety"] == variety)]
    return data_filter


@st.cache_resource
def load_model(model_id: str):
    model_path = MLRUNS / model_id / "artifacts" / "model" / "model.pkl"
    if not model_path.is_file():
        st.error(
            "Model File not found for selected Centre and Variety. Please Check the Model ID is configured correctly with Streamlit app."
        )
    model = pickle.load(open(model_path, "rb"))
    return model


@st.cache_data
def load_perc_change(predicted, base):
    value = round(100 * ((predicted - base) / base), 1)
    return value


def layout_predict_wheat_price(model_id: str, centre: str, variety: str):
    centre_variety_model = load_model(model_id)
    predicted_df = centre_variety_model.predict().forecast.reset_index().round(2)
    predicted_df.columns = ["date", "value"]

    df = load_historic_data(centre, variety)
    df = pd.concat([df, predicted_df], ignore_index=True)
    df["pct_change"] = df["value"].pct_change()

    # add chart
    fig = px.line(
        df,
        x="date",
        y="value",
        range_x=["2021-01-01", "2023-04-01"],
        labels={
            "date": "Month Year",
            "value": "Price (INR/kg)",
        },
        markers=True,
    )
    fig.update_traces(
        line_color="silver",
        line_width=3,
        marker=dict(
            color="LightSlateGrey", size=7, line=dict(width=2, color="DarkSlateGrey")
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=predicted_df["date"],
            y=predicted_df["value"],
            mode="lines+markers",
            marker=dict(
                color="limegreen", size=10, line=dict(width=2, color="DarkGreen")
            ),
            hovertemplate=None,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={
            "text": f"Price Forecast for next 3 months for {centre} -  {variety}",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(
                size=25,
            ),
        },
        hovermode="x unified",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with st.container():
        # add layout metric for previous month
        _, month_1, _, month_2, _, month_3, _ = st.columns(7)

        with month_1:
            month_1_val = round(df.iloc[-3, 1])
            st.metric(
                df.iloc[-3, 0].strftime("%b-%Y"),
                month_1_val,
                f"{round(df.iloc[-3,2]*100,2)}%",
            )

        with month_2:
            month_2_val = round(df.iloc[-2, 1], 2)
            st.metric(
                df.iloc[-2, 0].strftime("%b-%Y"),
                month_2_val,
                f"{round(df.iloc[-2,2]*100,2)}%",
            )

        with month_3:
            month_3_val = round(df.iloc[-1, 1], 2)
            st.metric(
                df.iloc[-1, 0].strftime("%b-%Y"),
                month_3_val,
                f"{round(df.iloc[-1,2]*100,2)}%",
            )

    st.info(f"**Note:** Predicted and Historic prices are INR/Kg")


label = "Select a Centre"
centre_model_mapping: Dict[str, List[str]] = {
    "Bhopal": ["564a9065bac341a28cbfedbdcc45716a", "46258876af3c452ea29c576300c22c93"],
    "Udaipur": ["5e033428b96f4a56a6ff051ee28bad08", "c55246ecd5444f68989a3f749ca71e9d"],
    "Ranchi": ["928a1f2602a444af91c7d1e02933908a","34a4472954a1427696b8c5dfa084c622"],
    "Mumbai": ["b6a76880bf0648f4a6e8693bea765f57", "462808cbbfa74a4bb45fdaa65a6a936e"],
    "Hyderabad": ["f697cf3789bc4a0b9059bfcc1c817b5a", "ae0c38ae66264d36943fefe31852c0e9"],
    "Bhubneshwar": ["02dd49cd3b8a416bb7ec36bb9f61d881", "50670fe6333c441292963c94f2bd3ea8"],
    "Patna": ["cefd64ae299e4764b83516fc12bc3520","29cf24a43eb84f13940adbc1f8246b90"]
}

select_centre = st.selectbox(
    label,
    list(centre_model_mapping.keys()),
    index=0,
    key=list(centre_model_mapping.keys())[0],
    help=None,
    on_change=None,
    label_visibility="visible",
)

value_month_tab, explanation_tab = st.tabs(["Predict (Monthly)", "Explanations"])

with value_month_tab:
    st.title("Predict Wheat Prices")

    # add 2 columns here
    desi_wheat, high_yield_wheat = st.columns(2)

    with st.expander("Desi Wheat"):
        layout_predict_wheat_price(
            model_id=centre_model_mapping[select_centre][0],
            centre=select_centre,
            variety="Desi",
        )

    with st.expander("Kalyan HYV"):
        layout_predict_wheat_price(
            model_id=centre_model_mapping[select_centre][1],
            centre=select_centre,
            variety="Kalyan HYV",
        )


@st.cache_resource
def last_6_month_chart(df, centre, variety):
    fig = px.bar(
        df,
        x="month",
        y="pct_change",
        category_orders={"month": df["month"].to_list()},
        labels={
            "pct_change": "Percentage Change",
            "month": "Month",
        },
    )
    fig.update_traces(
        marker_color=df["color"],
        marker=dict(line=dict(width=2, color="DarkSlateGrey")),
    )
    fig.update_layout(
        title={
            "text": f"Percentage Changes in prices for Last 6 Months for {centre} - {variety}",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.9,
            "x": 0.5,
        }
    )
    return fig


@st.cache_resource
def prev_month_fig(df, centre, variety, month):
    fig = px.bar(
        month_wise,
        x="year",
        y="pct_change",
        category_orders={"month": month_wise["month"].to_list()},
        labels={
            "pct_change": "Percentage Change",
            "year": "Year",
        },
    )
    fig.update_traces(
        marker_color=month_wise["color"],
        marker=dict(line=dict(width=2, color="DarkSlateGrey")),
    )
    fig.update_layout(
        title={
            "text": f"Percentage Changes in prices for all {month} for {centre} - {variety}",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.9,
            "x": 0.5,
        }
    )
    return fig


@st.cache_resource
def trendline_over_historic_chart(df):
    fig = px.scatter(
        df,
        x="date",
        y="value",
        trendline="lowess",
        trendline_options=dict(frac=0.6),
        range_y=[15, 40],
        labels={
            "date": "Month Year",
            "value": "Price (INR/kg)",
        },
    )
    fig.update_traces(
        line_color="red",
        line_width=3,
        marker=dict(
            color="silver",
            size=13,
            line=dict(width=1.5, color="DarkSlateGrey"),
            opacity=0.7,
        ),
    )
    fig.update_layout(
        title={
            "text": "Trendline over all Historic Prices",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    return fig


with explanation_tab:
    desi_historic_data = load_historic_data_all_cols(select_centre, "Desi")
    kyv_historic_data = load_historic_data_all_cols(select_centre, "Kalyan HYV")

    with st.container():

        desi_last_6_month_col, hyv_last_6_month_col = st.columns(2)

        with desi_last_6_month_col:
            last_6_month = desi_historic_data.iloc[-6:, :].copy(deep=False)
            last_6_month["pct_change"] = last_6_month["value"].pct_change() * 100
            last_6_month["color"] = last_6_month["pct_change"].apply(
                lambda x: "lightgreen" if x > 0 else "lightpink"
            )
            last_6_month_chart_fig = last_6_month_chart(
                last_6_month, select_centre, "Desi"
            )
            st.plotly_chart(
                last_6_month_chart_fig, theme="streamlit", use_container_width=True
            )

        with hyv_last_6_month_col:
            last_6_month = kyv_historic_data.iloc[-6:, :].copy(deep=False)
            last_6_month["pct_change"] = last_6_month["value"].pct_change() * 100
            last_6_month["color"] = last_6_month["pct_change"].apply(
                lambda x: "lightgreen" if x > 0 else "lightpink"
            )
            last_6_month_chart_fig = last_6_month_chart(
                last_6_month, select_centre, "Kalyan HYV"
            )
            st.plotly_chart(
                last_6_month_chart_fig, theme="streamlit", use_container_width=True
            )

    with st.container():

        desi_month_wise_col, hyv_month_wise_col = st.columns(2)

        with desi_month_wise_col:
            select_month_desi = st.selectbox(
                label="Select a Month", options=["JAN", "FEB", "MAR"]
            )
            month_wise = desi_historic_data[
                desi_historic_data["month"] == select_month_desi
            ].copy(deep=False)
            month_wise["pct_change"] = month_wise["value"].pct_change() * 100
            month_wise["color"] = month_wise["pct_change"].apply(
                lambda x: "lightgreen" if x > 0 else "lightpink"
            )
            month_wise_chart = prev_month_fig(
                month_wise, select_centre, "Desi", select_month_desi
            )
            st.plotly_chart(
                month_wise_chart, theme="streamlit", use_container_width=True
            )

        with hyv_month_wise_col:
            select_month_kyv = st.selectbox(
                label="Select a Month", options=["JAN", "FEB", "MAR"], key="kyv"
            )
            month_wise = kyv_historic_data[
                kyv_historic_data["month"] == select_month_kyv
            ].copy(deep=False)
            month_wise["pct_change"] = month_wise["value"].pct_change() * 100
            month_wise["color"] = month_wise["pct_change"].apply(
                lambda x: "lightgreen" if x > 0 else "lightpink"
            )
            month_wise_chart = prev_month_fig(
                month_wise, select_centre, "Kalyan HYV", select_month_kyv
            )
            st.plotly_chart(
                month_wise_chart, theme="streamlit", use_container_width=True
            )

    with st.container():

        desi_trend_col, hyv_trend_col = st.columns(2)

        with desi_trend_col:
            trend_line_plot_desi = trendline_over_historic_chart(desi_historic_data)
            st.plotly_chart(
                trend_line_plot_desi, theme="streamlit", use_container_width=True
            )

        with hyv_trend_col:
            trend_line_plot_kyv = trendline_over_historic_chart(kyv_historic_data)
            st.plotly_chart(
                trend_line_plot_kyv, theme="streamlit", use_container_width=True
            )
