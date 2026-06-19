import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import fetch_nasa_data, engineer_features
from model_trainer import train_exoplanet_model, predict_custom_planet

st.set_page_config(page_title="Cosmic Classifier", page_icon="🪐")
st.title("🪐 Cosmic Habitable Zone Classifier")

# Load and process data cleanly using Streamlit's caching
@st.cache_data
def get_ready_data():
    raw_df = fetch_nasa_data()
    return engineer_features(raw_df)

df = get_ready_data()

if not df.empty:
    # Train model
    feature_cols = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff']
    model = train_exoplanet_model(df)

    # Sidebar UI
    st.sidebar.header("Customize Your Exoplanet")
    mass = st.sidebar.slider("Planet Mass (Earth Masses)", 0.1, 50.0, 1.0)
    radius = st.sidebar.slider("Planet Radius (Earth Radii)", 0.1, 20.0, 1.0)
    period = st.sidebar.slider("Orbital Period (Days)", 1.0, 1000.0, 365.0)
    star_temp = st.sidebar.slider("Host Star Temperature (Kelvin)", 2000, 10000, 5778)

    # Run Prediction
    prediction, probabilities = predict_custom_planet(model, feature_cols, mass, radius, period, star_temp)

    col1, col2 = st.columns([1, 2])

    # Output Display
with col1:
    st.subheader("Prediction Result")
    if prediction == "Habitable Candidate":
        st.success(f"🎉 **{prediction}**")
    elif prediction == "Gas Giant":
        st.warning(f"💨 **{prediction}**")
    else:
        st.info(f"🪨 **{prediction}**")

    st.write("### Model Confidence Breakdown")
    for cls, prob in zip(model.classes_, probabilities):
        st.write(f"- {cls}: {prob*100:.1f}%")

with col2:
    st.subheader("Cosmic Mapping")

    #temp dataframe for plotting
    user_df = pd.DataFrame([{
        'pl_bmasse': mass,
        'pl_rade': radius,
        'planet_type': 'YOUR PLANET ✨',
        'pl_name': 'Your Custom Creation',
    }])

    # Combine NASA data with user-defined planet for visualization
    plot_df = pd.concat([df, user_df], ignore_index=True)

    #scatter plot of planet mass vs radius, colored by planet type
    fig = px.scatter(
        plot_df,
        x = 'pl_bmasse',
        y = 'pl_rade',
        color = 'planet_type',
        hover_name = 'pl_name',
        log_x=True, #helps handle massive ranges in plot
        labels={"pl_bmasse": "Planet Mass (Earth Masses)", "pl_rade": "Planet Radius (Earth Radius)"},
        title="Where your planet fits in the Cosmos",
        color_discrete_map={
            "Habitable Candidate": "#00cc96",
            "Gas Giant": "#ef553b",
            "Rocky / Non-Habitable": "#636efa",
            "YOUR PLANET ✨": "#FFFFFF"  # White for user-defined planet
        }
    )
    # Update layout for better aesthetics
    fig.update_layout(template="plotly_dark", legend_title_text='Classification')

    #render
    st.plotly_chart(fig, use_container_width=True)