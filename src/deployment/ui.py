import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# API endpoint
API_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="EV Charging Optimization Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ EV Charging Optimization Dashboard")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("Model Information")
        try:
            model_info = requests.get(f"{API_URL}/model_info").json()
            st.json(model_info)
        except Exception as e:
            st.error(f"Failed to fetch model info: {str(e)}")
    
    # Main content
    tab1, tab2 = st.tabs(["Make Decision", "View History"])
    
    with tab1:
        st.header("Make Charging Decision")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            battery_level = st.slider("Battery Level", 0.0, 100.0, 50.0)
            grid_load = st.slider("Grid Load", 0.0, 100.0, 50.0)
            price = st.number_input("Price", 0.0, 100.0, 50.0)
        
        with col2:
            queue_length = st.number_input("Queue Length", 0, 100, 0)
            time_of_day = st.slider("Time of Day", 0.0, 24.0, 12.0)
            day_of_week = st.selectbox("Day of Week", range(7), format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
        
        # Constraints
        st.subheader("Constraints")
        col3, col4 = st.columns(2)
        
        with col3:
            max_queue_length = st.number_input("Max Queue Length", 0, 100, 10)
            min_charging_rate = st.number_input("Min Charging Rate", 0.0, 100.0, 0.0)
        
        with col4:
            max_charging_rate = st.number_input("Max Charging Rate", 0.0, 100.0, 100.0)
            price_threshold = st.number_input("Price Threshold", 0.0, 100.0, 80.0)
        
        if st.button("Make Decision"):
            try:
                # Prepare request data
                state = {
                    "battery_level": battery_level,
                    "grid_load": grid_load,
                    "price": price,
                    "queue_length": queue_length,
                    "time_of_day": time_of_day,
                    "day_of_week": day_of_week
                }
                
                constraints = {
                    "max_queue_length": max_queue_length,
                    "min_charging_rate": min_charging_rate,
                    "max_charging_rate": max_charging_rate,
                    "price_threshold": price_threshold
                }
                
                # Make API request
                response = requests.post(
                    f"{API_URL}/make_decision",
                    json={"state": state, "constraints": constraints}
                )
                decision = response.json()
                
                # Display results
                st.success("Decision made successfully!")
                
                # Show decision details
                col5, col6 = st.columns(2)
                
                with col5:
                    st.metric("Charging Rate", f"{decision['action'][0]:.2f}%")
                    st.metric("Queue Management", f"{decision['action'][1]:.2f}")
                
                with col6:
                    st.metric("Constraints Applied", "Yes" if decision["constraints_applied"] else "No")
                    st.metric("Timestamp", decision["timestamp"])
                
                # Save decision history
                requests.post(f"{API_URL}/save_history")
                
            except Exception as e:
                st.error(f"Failed to make decision: {str(e)}")
    
    with tab2:
        st.header("Decision History")
        
        try:
            # Fetch decision history
            history = requests.get(f"{API_URL}/decision_history").json()
            
            if history:
                # Convert to DataFrame
                df = pd.DataFrame(history)
                
                # Create visualizations
                col7, col8 = st.columns(2)
                
                with col7:
                    fig1 = px.line(df, x="timestamp", y="action", title="Charging Actions Over Time")
                    st.plotly_chart(fig1)
                
                with col8:
                    fig2 = px.scatter(df, x="state.battery_level", y="action", 
                                   color="state.grid_load", title="Charging Rate vs Battery Level")
                    st.plotly_chart(fig2)
                
                # Show raw data
                st.subheader("Raw Data")
                st.dataframe(df)
            else:
                st.info("No decision history available.")
                
        except Exception as e:
            st.error(f"Failed to fetch decision history: {str(e)}")

if __name__ == "__main__":
    main() 