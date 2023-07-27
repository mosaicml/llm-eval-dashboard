import streamlit as st
from io import StringIO
import pandas as pd
from typing import List, Tuple

def grab_categories_and_values(df: pd.DataFrame, model_name: str = None) -> Tuple[List[str], List[float]]:

    categories =  list(df.loc[model_name].index)
    values = list(df.loc[model_name].values)

    # append the first category to the end to close the loop
    categories.append(categories[0])
    values.append(values[0])

    return categories, values

st.title("LLM evaluator")

# Add a sidebar title
st.sidebar.title('Eval Input')
# Add file uploader for .csv format
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
md_default = """
| model_name               |   average |   world_knowledge |   commonsense_reasoning |   language_understanding |   symbolic_problem_solving |   reading_comprehension |
|:-------------------------|----------:|------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|
| llama-30b    |  0.508013 |          0.570561 |                0.521302 |                 0.549439 |                   0.321474 |                0.577292 |
| huggyllama/llama-13b                |  0.428223 |          0.511058 |                0.464285 |                 0.482423 |                  0.23844   |                0.444907 |
| huggyllama/llama-7b |  0.351241 |          0.354118 |                0.396072 |                 0.428827 |                   0.182015 |                0.395171 |
| togethercomputer/RedPajama-INCITE-7B-Instruct |  0.354936 |          0.368793 |                0.367142 |                 0.395898 |                   0.210048 |                0.432801 |
| mosaicml/mpt-7b-instruct |  0.338077 |          0.338253 |                0.416911 |                 0.371509 |                   0.17265  |                0.391062 |
| mosaicml/mpt-7b          |  0.310326 |          0.310191 |                0.384509 |                 0.380392 |                   0.162957 |                0.31358  |
| tiiuae/falcon-7b         |  0.309822 |          0.272142 |                0.419968 |                 0.369998 |                   0.158363 |                0.328637 |
| togethercomputer/RedPajama-INCITE-7B-Base     |  0.29738  |          0.312032 |                0.363261 |                 0.3733   |                   0.126577 |                0.311731 |
| tiiuae/falcon-7b-instruct                     |  0.28197  |          0.260288 |                0.370308 |                 0.332523 |                   0.107958 |                0.338774 |
| EleutherAI/pythia-12b  |  0.274429 |          0.252255 |                0.344973 |                 0.33249  |                   0.136118 |                0.306308 |
| EleutherAI/gpt-j-6b                 |  0.268168 |          0.260849 |                0.330648 |                 0.311813 |                  0.120669  |                0.31686  |
| facebook/opt-6.7b                   |  0.24994  |          0.236678 |                0.326348 |                 0.322621 |                  0.0930295 |                0.271022 |
| EleutherAI/pythia-6.9b              |  0.248811 |          0.218628 |                0.308817 |                 0.304028 |                  0.120792  |                0.291793 |
| stabilityai/stablelm-tuned-alpha-7b |  0.163522 |          0.129503 |                0.198957 |                 0.20249  |                  0.093985  |                0.192676 |
"""
# Create a text area in the sidebar
md_table_string = st.sidebar.text_area("Paste your markdown text here (no triple quotes)", md_default, height=500)

# st.write(md_table_string)

print(md_table_string)

md_df = pd.read_csv(
    StringIO(md_table_string.replace(' ', '')),  # Get rid of whitespaces
    sep='|',
    index_col=1
).dropna(
    axis=1,
    how='all'
).iloc[1:]
models = list(md_df.index)
md_df = md_df.reset_index()
md_df = md_df.set_index('model_name')
# convert all columns to float type, except for the index
md_df = md_df.astype(float)

# two columns for the model names and the metrics
col1, col2 = st.columns(2)

with col1:
    # show a multiselect widget with the model names. default to just "mosaicml/mpt-7b"
    selected_models = st.multiselect(
        'Select models to compare',
        list(md_df.index),
        default=['mosaicml/mpt-7b']
    )

with col2:
    # show a multiselect widget with the metrics. default to to everything except for "average"
    selected_metrics = st.multiselect(
        'Select metrics to compare',
        list(md_df.columns),
        default=list(md_df.columns)[1:]
    )

# restrict the dataframe to the selected models and metrics
md_df = md_df[selected_metrics].loc[selected_models]

# show the dataframe
st.write(md_df)

import plotly.graph_objects as go

# loop over model names. for each, grab the categories and values and add a trace to the figure
# build a streamlit plotly chart

fig = go.Figure()

for model_name in selected_models:
    categories, values = grab_categories_and_values(md_df, model_name)
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model_name
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0., 1.]
    )),
  showlegend=True
)

# fig.show()
st.plotly_chart(fig)