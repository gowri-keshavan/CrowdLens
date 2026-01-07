# crowdlens_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from crowdlens_backend import run_analysis
from crowdlens_backend_rak import run_analysis

st.set_page_config("CrowdLens Dashboard", layout="wide")

# Sidebar controls and explanation
st.sidebar.title("Settings & Controls")
rolling_window = st.sidebar.slider("Rolling window for smoothing (frames)", 1, 200, 10)
aggr_threshold = st.sidebar.slider("Aggression threshold (%)", 0, 100, 60)
preset = st.sidebar.selectbox("Sensitivity preset", ["Balanced", "Demonstration (High)", "Strict (Sensitive)"])
st.sidebar.markdown("---")
st.sidebar.markdown("Use the controls to adjust smoothing and the threshold used for highlighting high-aggression regions.\n\nThis does not change the detection backend, only the visualization and interpretation shown here.")

st.title("CrowdLens – Crowd Analytics Dashboard")

# Short interpretation box for teachers
with st.expander("How to interpret the results", expanded=False):
    st.write("- Aggression (%) is a scene-level score derived from per-person speeds (pixels/frame), smoothed for stability.\n- Density is people per pixel (low-level proxy).\n- Use the rolling window slider to reduce noise; higher window = smoother curve.\n- Shaded regions indicate where aggression exceeds the chosen threshold.\n- Watch out for camera motion or background activity — these can produce false positives.")

if st.button("Run Analysis"):
    with st.spinner("Processing video... this may take a while depending on model and video length"):
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

    # compute rolling mean for plotting
    df["Aggression_Roll"] = df["Aggression (%)"].rolling(window=rolling_window, min_periods=1).mean()

    # KPIs (more informative)
    peak_aggr = float(max(aggression)) if aggression else 0.0
    avg_aggr = float(sum(aggression)/len(aggression)) if aggression else 0.0
    peak_density = float(max(density)) if density else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique People", total_people, help="Unique track IDs seen in the video")
    col2.metric("Peak Aggression (%)", f"{peak_aggr:.1f}", delta=None)
    # color hint with emoji
    status = "🟢 Normal" if peak_aggr < aggr_threshold * 0.8 else ("🟡 Caution" if peak_aggr < aggr_threshold else "🔴 Alert")
    col3.metric("Status", status)

    # Charts area
    st.subheader("Aggression Score Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Frame"], y=df["Aggression (%)"], mode='lines', name='Raw', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(x=df["Frame"], y=df["Aggression_Roll"], mode='lines', name=f'Rolling (w={rolling_window})', line=dict(color='firebrick', width=3)))
    # horizontal threshold
    fig.add_hline(y=aggr_threshold, line=dict(color='orange', dash='dash'), annotation_text=f"Threshold ({aggr_threshold}%)", annotation_position="top left")

    # shaded regions where rolling mean exceeds threshold
    high = df["Aggression_Roll"] > aggr_threshold
    if high.any():
        # find contiguous regions
        regions = []
        start = None
        for i, val in enumerate(high):
            if val and start is None:
                start = df.at[i, "Frame"]
            if not val and start is not None:
                end = df.at[i-1, "Frame"]
                regions.append((start, end))
                start = None
        if start is not None:
            regions.append((start, df.at[len(df)-1, "Frame"]))
        for s, e in regions:
            fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.08, layer="below", line_width=0)

    fig.update_layout(yaxis_title='Aggression (%)', xaxis_title='Frame', yaxis=dict(range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # Density and count charts side by side
    st.subheader("Density and People Count")
    c1, c2 = st.columns(2)
    fig_d = px.line(df, x="Frame", y="Density", labels={"Density":"People per pixel"})
    c1.plotly_chart(fig_d, use_container_width=True)
    fig_c = px.line(df, x="Frame", y="Current People", labels={"Current People":"People"})
    c2.plotly_chart(fig_c, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(label="Download results as CSV", data=csv, file_name='crowd_stats.csv', mime='text/csv')

else:
    st.info("Click **Run Analysis** to process the video and see crowd statistics and aggression graph.")
