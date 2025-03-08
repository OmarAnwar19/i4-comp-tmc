import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="i4 Case Competition Dashboard", page_icon="🚗")

st.title("🚗 i4 Case Competition Dashboard")

# Loading data
@st.cache_data
def load_data():
    file_id = "12l2p5Hgbrn4qViwkY6SHm-4WDhPqQdoD"
    url = f"https://drive.google.com/uc?id={file_id}"
    df_tmc = pd.read_csv(url)

    file_id = "1CLZFscFiMc2hFclbLIgQEACqSkAIpWYp"
    url = f"https://drive.google.com/uc?id={file_id}"
    df_ghg = pd.read_csv(url, index_col=0)

    # Data processing for GHG data
    df_ghg = df_ghg.transpose()
    df_ghg.dropna(axis=1, how='all', inplace=True)
    df_ghg = df_ghg.iloc[:-1, :]
    df_ghg.index.name = 'year'
    df_ghg.index = (df_ghg.index.astype(int) + 2).astype(str)
    
    df_ghg['Buses/Heavy'] = df_ghg['Buses'] + df_ghg[' Light Trucks'] + df_ghg[' Medium Trucks'] + df_ghg[' Heavy Trucks']
    df_ghg = df_ghg[['Total GHG Emissions Including Electricity (Mt of CO2e)a,b,d,e,f', 'Cars', 'Buses/Heavy']]
    df_ghg.rename(columns={'Total GHG Emissions Including Electricity (Mt of CO2e)a,b,d,e,f': 'Emissions (Mt of CO2e)'}, inplace=True)
    df_ghg.columns = [x.lower() for x in df_ghg.columns]
    
    # Data processing for TMC data
    df_tmc['start_time'] = pd.to_datetime(df_tmc['start_time'])
    df_tmc['end_time'] = pd.to_datetime(df_tmc['end_time'])
    
    df_tmc['hour'] = df_tmc['start_time'].dt.hour
    df_tmc['date'] = df_tmc['start_time'].dt.date
    
    df_tmc['total_cars'] = df_tmc.filter(like='_appr_cars_').sum(axis=1)
    df_tmc['total_trucks'] = df_tmc.filter(like='_appr_truck_').sum(axis=1)
    df_tmc['total_buses'] = df_tmc.filter(like='_appr_bus_').sum(axis=1)
    df_tmc['total_peds'] = df_tmc.filter(like='_appr_peds').sum(axis=1)
    df_tmc['total_bikes'] = df_tmc.filter(like='_appr_bike').sum(axis=1)
    
    traffic_overview = df_tmc.groupby(['location_name', 'date', 'hour'])[
        ['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes']].sum().reset_index()
    
    df_tmc_cleaned = df_tmc.drop(columns=['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes'])
    
    df_tmc = df_tmc_cleaned.merge(traffic_overview,
                                  on=['location_name', 'date', 'hour'],
                                  how='left')
    
    df_tmc['total_traffic'] = df_tmc[['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes']].sum(axis=1)
    df_tmc['year'] = pd.to_datetime(df_tmc['start_time']).dt.year
    
    yearly_totals = df_tmc.groupby('year').agg({
        'total_cars': 'sum',
        'total_trucks': 'sum',
        'total_buses': 'sum',
        'total_peds': 'sum',
        'total_bikes': 'sum'
    }).reset_index()
    
    df_ghg = df_ghg.reset_index()
    df_ghg['year'] = df_ghg['year'].astype(int)
    df_ghg = pd.merge(
        df_ghg,
        yearly_totals,
        on='year',
        how='left'
    )
    
    df_ghg['total_car_emissions'] = df_tmc['total_cars'] * df_ghg['cars']
    df_ghg['total_bus_emissions'] = df_tmc['total_buses'] * df_ghg['buses/heavy']
    df_ghg['total_truck_emissions'] = df_tmc['total_trucks'] * df_ghg['buses/heavy']
    df_ghg['total_ped_emissions'] = df_tmc['total_peds'] * 0
    df_ghg['total_bike_emissions'] = df_tmc['total_bikes'] * 0
    
    df_ghg['total_emissions'] = (
        df_ghg['total_car_emissions'] + df_ghg['total_bus_emissions'] +
        df_ghg['total_truck_emissions'] + df_ghg['total_ped_emissions'] +
        df_ghg['total_bike_emissions']
    )
    
    df_ghg.set_index('year', inplace=True)
    df_ghg = df_ghg[df_ghg.index >= 2020]
    
    # Create intersections data
    df_unique = df_tmc.drop_duplicates(subset=["centreline_id"], keep='first').reset_index(drop=True)
    coordinates = np.radians(df_unique[['latitude', 'longitude']].values)
    tree = BallTree(coordinates, metric='haversine')
    distances, indices = tree.query(coordinates, k=2)
    
    earth_radius = 6371000
    df_unique['distance_to_closest'] = distances[:, 1] * earth_radius
    df_unique['closest_intersection'] = [df_unique.loc[idx[1], 'location_name'] for idx in indices]
    
    df_unique = df_unique[['centreline_id', 'location_name', 'closest_intersection', 'distance_to_closest']]
    df_intersections = df_unique.sort_values(by='distance_to_closest')
    
    location_traffic = df_tmc.groupby(['location_name']).agg({
        'total_traffic': 'sum',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    location_traffic.rename(columns={
        'location_name': 'closest_intersection',
        'total_traffic': 'total_traffic_intersection',
        'latitude': 'latitude_intersection',
        'longitude': 'longitude_intersection'
    }, inplace=True)
    
    df_intersections = pd.merge(df_intersections, location_traffic, on='closest_intersection', how='left')
    location_traffic.rename(columns={
        'closest_intersection': 'location_name',
        'total_traffic_intersection': 'total_traffic_location',
        'latitude_intersection': 'latitude_location',
        'longitude_intersection': 'longitude_location'
    }, inplace=True)
    
    df_intersections = pd.merge(df_intersections, location_traffic, on='location_name', how='left')
    
    # Apply KMeans clustering
    traffic_features = ['total_cars', 'total_trucks', 'total_buses', 'total_bikes', 'total_peds']
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_tmc_subset = df_tmc[traffic_features].fillna(0)  # Handle NaN values
    clusters = kmeans.fit_predict(df_tmc_subset)
    df_tmc['congestion_cluster'] = clusters
    
    return df_tmc, df_ghg, df_intersections

with st.spinner('Loading data... This may take a moment.'):
    try:
        df_tmc, df_ghg, df_intersections = load_data()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.warning("For this dashboard to work, please make sure you've updated the file paths to your actual data files.")
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Traffic Analysis", "Emissions Analysis", "Location Analysis", "Correlations"])
with tab1:
    st.header("Traffic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Patterns")
        traffic_hourly = df_tmc.groupby('hour')[['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes']].sum().reset_index()
        fig = px.line(traffic_hourly, x='hour', y=['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes'],
                    markers=True, title='Traffic Volume Trends by Hour',
                    labels={'value': 'Total Count', 'hour': 'Hour of Day'},
                    template='plotly_dark')
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Daily Traffic Trends")
        daily_traffic = df_tmc.groupby('date')[
            ['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes']
        ].sum().reset_index()
        fig = px.line(daily_traffic, x='date', y=['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes'],
                    title='Daily Traffic Trends by Vehicle Type',
                    labels={'value': 'Total Count', 'date': 'Date'},
                    template='plotly_dark')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Most Congested Locations")
        location_traffic = df_tmc.groupby('location_name')[['total_cars', 'total_trucks', 'total_buses']].sum().reset_index()
        top_locations = location_traffic.sort_values(by='total_cars', ascending=False).head(10)
        fig = px.bar(top_locations, x='location_name', y=['total_cars', 'total_trucks', 'total_buses'],
                    title='Top 10 Most Congested Locations',
                    labels={'value': 'Total Vehicles', 'location_name': 'Intersection'},
                    barmode='stack', template='plotly_dark')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Least Congested Locations")
        bottom_cars = location_traffic.sort_values(by='total_cars', ascending=True).head(10)
        fig = px.bar(bottom_cars, x='location_name', y=['total_cars', 'total_trucks', 'total_buses'],
                    title='Bottom 10 Least Congested Locations',
                    labels={'value': 'Total Vehicles', 'location_name': 'Intersection'},
                    barmode='stack', template='plotly_dark')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Emissions Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Emissions Over Time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_ghg.index, y=df_ghg['total_emissions'], mode='lines+markers', name='Total Emissions', line=dict(color='crimson')))
        fig1.update_layout(title='Total Emissions Over Time (All Vehicles)', xaxis_title='Year', yaxis_title='Total Emissions (kg CO2)', template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Emissions by Vehicle Type")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_ghg.index, y=df_ghg['total_car_emissions'], mode='lines', name='Car Emissions', line=dict(color='steelblue')))
        fig4.add_trace(go.Scatter(x=df_ghg.index, y=df_ghg['total_bus_emissions'], mode='lines', name='Bus Emissions', line=dict(color='darkorange')))
        fig4.add_trace(go.Scatter(x=df_ghg.index, y=df_ghg['total_truck_emissions'], mode='lines', name='Truck Emissions', line=dict(color='firebrick')))
        fig4.update_layout(title='Emissions Over Time by Vehicle Type', xaxis_title='Year', yaxis_title='Emissions (kg CO2)', template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)
    
    
    st.subheader("Emissions Breakdown")
    col3 = st.container()
    with col3:
        years = sorted(df_ghg.index.astype(str).tolist())
        selected_year = st.selectbox("Select Year for Breakdown", years, index=len(years)-1)
        fig = px.scatter(df_tmc, x='total_cars', y='total_trucks', size='total_buses',
                        color='location_name', hover_name='location_name',
                        title='Traffic Composition: Cars vs. Trucks vs. Buses',
                        labels={'total_cars': 'Total Cars', 'total_trucks': 'Total Trucks'},
                        template='plotly_dark')
        fig.update_layout(legend_title_text='Location')
        st.plotly_chart(fig, use_container_width=True)

    col4 = st.container()
    with col4:
        latest_year = df_ghg.index.max()
        latest_data = df_ghg.loc[latest_year]
        emissions_data = [latest_data['total_car_emissions'], latest_data['total_bus_emissions'], latest_data['total_truck_emissions']]
        labels = ['Car Emissions', 'Bus Emissions', 'Truck Emissions']
        fig5 = go.Figure(data=[go.Pie(labels=labels, values=emissions_data, hole=0.3, textinfo='percent+label',
                                    marker=dict(colors=['#4C72B0', '#DD8452', '#C44E52']))])
        fig5.update_layout(title=f'Emissions Breakdown by Vehicle Type ({latest_year})', template='plotly_dark')
        st.plotly_chart(fig5, use_container_width=True)


with tab3:
    st.header("Location Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic Congestion Clusters")
        fig = px.scatter(
            df_tmc,
            x='longitude',
            y='latitude',
            color='congestion_cluster',
            color_continuous_scale='viridis',
            title='Traffic Congestion Clusters',
            labels={'color': 'Cluster'},
            template='plotly_dark',
            opacity=0.6
        )
        
        fig.update_layout(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            coloraxis_colorbar=dict(title='Cluster')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Location Traffic Data")
        fig = px.scatter_mapbox(
            df_intersections,
            lat='latitude_location',
            lon='longitude_location',
            hover_name='location_name',
            hover_data={
                'closest_intersection': True,
                'distance_to_closest': True,
                'total_traffic_location': True,
                'total_traffic_intersection': True
            },
            size='total_traffic_location',
            color='total_traffic_intersection',
            title='Traffic Visualization by Location and Closest Intersection',
            template='plotly_dark',
            size_max=15,
            zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Correlations and Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Types Correlation")
        correlation = df_tmc[['total_cars', 'total_trucks', 'total_buses', 'total_peds', 'total_bikes']].corr()
        
        fig = px.imshow(
            correlation, 
            text_auto=True, 
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Heatmap: Vehicle Types'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Emissions Correlation")
        emissions_corr = df_ghg[['total_car_emissions', 'total_bus_emissions', 'total_truck_emissions']].corr()
        fig3 = px.imshow(
            emissions_corr, 
            text_auto=True, 
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Heatmap: Vehicle Emissions'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Relationship scatter plots
    st.subheader("Relationship Analysis")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        fig = px.scatter(df_tmc, x='total_cars', y='total_bikes',
                         title='Relationship Between Cars and Bikes',
                         labels={'total_cars': 'Total Cars', 'total_bikes': 'Total Bikes'},
                         template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = px.scatter(df_tmc, x='total_cars', y='total_peds',
                         title='Relationship Between Cars and Pedestrians',
                         labels={'total_cars': 'Total Cars', 'total_peds': 'Total Pedestrians'},
                         template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col5:
        fig = px.scatter(df_tmc, x='total_cars', y='total_trucks',
                         title='Relationship Between Cars and Trucks',
                         labels={'total_cars': 'Total Cars', 'total_trucks': 'Total Trucks'},
                         template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)