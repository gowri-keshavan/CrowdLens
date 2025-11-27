# crowdlens_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from crowdlens_backend import run_analysis

st.set_page_config("CrowdLens Dashboard", layout="wide")

st.title("CrowdLens – Crowd Analytics Dashboard")

if st.button("Run Analysis"):
    with st.spinner("Processing video..."):
        result = run_analysis()

    st.success("Analysis finished!")

    total_people = result["total_people"]
    steps = result["steps"]
    density = result["density"]
    aggression = result["aggression"]
    counts = result["counts"]

    df = pd.DataFrame({
        "Frame": steps,
        "Current People": counts,
        "Density": density,
        "Aggression (%)": aggression,
    })

    # KPIs
    col1, col2 = st.columns(2)
    col1.metric("Total Unique People in Video", total_people)
    col2.metric("Peak Aggression (%)", f"{max(aggression):.1f}")

    # Charts
    st.subheader("Crowd Density Over Time")
    fig_density = px.line(df, x="Frame", y="Density",
                          labels={"Density": "People per pixel"})
    st.plotly_chart(fig_density, use_container_width=True)

    st.subheader("Aggression Score Over Time")
    fig_aggr = px.line(df, x="Frame", y="Aggression (%)",
                       labels={"Aggression (%)": "Aggression (%)"},
                       range_y=[0,100])
    st.plotly_chart(fig_aggr, use_container_width=True)

    st.subheader("Current People Count Over Time")
    fig_count = px.line(df, x="Frame", y="Current People")
    st.plotly_chart(fig_count, use_container_width=True)
else:
    st.info("Click **Run Analysis** to process the video and see crowd statistics and aggression graph.")
