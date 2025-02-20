import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mplsoccer import Pitch
import base64


# 🎨 Custom CSS for White Filter Text
st.markdown("""
    <style>
        /* Sidebar background */
        .stSidebar {
            background-color: #1E1E1E !important;
            padding: 10px;
        }

        /* Sidebar header, labels */
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label {
            color: white !important;
            font-weight: bold;
        }

        /* Dropdown text */
        div.stSelectbox div[data-baseweb="select"] > div,
        div.stMultiSelect div[data-baseweb="select"] > div {
            color: white !important;
        }

        /* Sidebar borders */
        .stSidebar selectbox, .stSidebar multiselect {
            border: 1px solid #FFFFFF;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the logo image and encode it to base64
with open("/Users/pegra441/Desktop/DeepTactix.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# 🎯 Sidebar Logo & Company Name
st.sidebar.markdown("""
    <style>
        .sidebar-logo {
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-logo img {
            width: 150px;  /* Adjust the width */
            border-radius: 10px;
        }
        .company-name {
            font-size: 22px;
            font-weight: bold;
            color: #FFFFFF;  /* White text color */
            text-align: center;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Display Logo and Company Name
st.sidebar.markdown(f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{encoded_image}" alt="DeepTactix Logo" />
        <div class="company-name">DeepTactix</div>
    </div>
""", unsafe_allow_html=True)


# Load the dataset
file_path = '/Users/pegra441/Desktop/stories_with_cluster.xlsx'
df = pd.read_excel(file_path)

# Mapping team IDs to names
team_names = {
    '240': 'Manchester City',
    '303': 'West Ham United'
}
df['team_name'] = df['team_id_x'].astype(str).map(team_names)

# Cluster to Playing Style Mapping
cluster_names = {
    0: 'Pressured Deep Build-Ups',
    1: 'Midfield Build-Ups',
    2: 'High-Pressure Penetration',
    3: 'Direct Long Ball Transition',
    4: 'Prolonged Tiki-Taka (Possession-Based)'
}

# App Title
st.title("⚽ Football Data Analysis: Manchester City vs West Ham United")
st.markdown("---")

match_ids = df['match_id_x'].unique()
selected_match_id = st.sidebar.selectbox("Select Match ID", options=match_ids)

# Sidebar for Team Selection
selected_team = st.sidebar.selectbox("Select Team", ['Manchester City', 'West Ham United'])
# Filter data for the selected team
team_data = df[df['team_name'] == selected_team]

# Define Tabs
tabs = st.tabs(["📊 Match Metrics", "🏃‍♂️ Player Performance", "🎯 Playing Style Analysis", "📍 Pass Maps", "📊 Match Summary"])

# 1️⃣ Key Metrics Tab
with tabs[0]:
    st.header(f"📊 Key Metrics for {selected_team}")

    # Metrics Calculation
    total_possessions = team_data['possession_id'].nunique()
    average_xT = round(team_data['total_xT'].mean(), 3)
    total_goals = team_data['goal'].sum()
    possession_efficiency = round(team_data['possession_efficiency'].mean(), 2)
    pass_success_rate = round(team_data['pass_success_rate'].mean(), 2)

    # Display Metrics in Columns
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    # Display KPIs
    col1.metric("⚽ Total Possessions", total_possessions)
    col2.metric("📈 Avg xT", average_xT)
    col3.metric("🥅 Total Goals", total_goals)
    col4.metric("🔥 Possession Efficiency", f"{possession_efficiency * 100}%")
    col5.metric("🎯 Pass Success Rate", f"{pass_success_rate * 100}%")

    # Radar Plot for Playing Style Distribution
    st.subheader("📊 Playing Style Distribution")

    # Count possessions per cluster
    style_distribution = team_data['Cluster'].value_counts(normalize=True).sort_index() * 100

    # Map cluster numbers to style names
    labels = [cluster_names.get(i, f"Cluster {i}") for i in style_distribution.index]
    values = style_distribution.values

    # Radar Chart Setup
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))  # Closing the loop
    angles += angles[:1]  # Closing the loop

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='b', linewidth=2)
    ax.fill(angles, values, color='b', alpha=0.25)

    # Set labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"Playing Style Distribution - {selected_team}", size=14, weight='bold')

    st.pyplot(fig)

# Placeholder for Future Tabs
# 2️⃣ Top Players Tab
# 2️⃣ Top Players Tab
with tabs[1]:
    st.header(f"🏃‍♂️ Top Players for {selected_team}")

    # 🎯 Filter for Specific Playing Styles (Clusters)
    style_filter = st.multiselect("Filter by Playing Style", list(cluster_names.values()), default=list(cluster_names.values()))
    selected_clusters = [key for key, value in cluster_names.items() if value in style_filter]
    filtered_data = team_data[team_data['Cluster'].isin(selected_clusters)]

    # 🎯 Filter by Player Role (Position)
    st.subheader("🎯 Filter by Player Role")
    player_roles = filtered_data['player_position'].dropna().unique()
    selected_roles = st.multiselect("Select Player Role(s)", options=player_roles, default=list(player_roles))

    # Apply Player Role Filter
    if selected_roles:
        filtered_data = filtered_data[filtered_data['player_position'].isin(selected_roles)]

    # Group by Player Name and Cluster
    player_performance = filtered_data.groupby(['name', 'Cluster']).agg({
        'total_xT': ['sum', 'mean'],                # Total and Average xT
        'goal': 'sum',                              # Total Goals
        'possession_efficiency': 'mean',            # Average Possession Efficiency
        'pass_success_rate': 'mean',                # Pass Success Rate
        'BYPASSED_OPPONENTS': 'sum',                # Total Bypassed Opponents
        'ASSISTS': 'sum'                            # Total Assists
    }).reset_index()

    # Flatten column names
    player_performance.columns = [
        'Player', 'Cluster', 'Total xT', 'Avg xT', 
        'Goals', 'Possession Efficiency', 
        'Pass Success Rate', 'Bypassed Opponents', 
        'Assists'
    ]

    # Map cluster numbers to names
    player_performance['Playing Style'] = player_performance['Cluster'].map(cluster_names)

    # 🎯 Select Metric to View
    st.subheader("📊 Select Player Metric to Visualize")
    metric_options = ['Total xT', 'Avg xT', 'Goals', 'Possession Efficiency', 'Pass Success Rate', 'Bypassed Opponents', 'Assists']
    selected_metric = st.selectbox("Choose Metric", metric_options)

    # Display Top 10 Players Based on Selected Metric
    st.subheader(f"🔥 Top 10 Players by {selected_metric}")
    top_players = player_performance.sort_values(by=selected_metric, ascending=False).head(10)

    # Bar Plot for Selected Metric
    fig, ax = plt.subplots(figsize=(10, 6))
    top_players_sorted = top_players.sort_values(by=selected_metric)
    ax.barh(top_players_sorted['Player'], top_players_sorted[selected_metric], color='dodgerblue')
    ax.set_xlabel(selected_metric)
    ax.set_title(f"Top 10 Players by {selected_metric} - {selected_team}")
    st.pyplot(fig)

    st.markdown("---")

    # 🎯 Display Raw Data (Optional)
    if st.checkbox("Show Player Performance Data"):
        st.dataframe(player_performance)



# 🎯 Tab 3: Playing Style
with tabs[2]:
    st.header("🎯 Playing Style Analysis")

    # Ensure the user selects a team
    if selected_team:
        if isinstance(selected_team, str):
            selected_team = [selected_team]  # Ensure it's a list

        # Filter data for the selected team(s)
        team_data = df[df['team_name'].isin(selected_team)]

        # 1️⃣ Possession Style Distribution
        st.subheader("⚽ Possession Style Distribution")
        style_distribution = team_data['Cluster'].map(cluster_names).value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=style_distribution.values, y=style_distribution.index, palette="coolwarm")
        plt.xlabel("Number of Possessions")
        plt.ylabel("Playing Style")
        plt.title(f"Possession Style Distribution for {', '.join(selected_team)}")
        st.pyplot(plt)

        # 2️⃣ Style Efficiency Metrics
        st.subheader("📊 Style Efficiency Metrics")
        style_metrics = team_data.groupby('Cluster').agg({
            'total_xT': 'mean',
            'pass_success_rate': 'mean',
            'possession_duration': 'mean',
            'goal': 'sum'
        }).reset_index()

        style_metrics['Cluster'] = style_metrics['Cluster'].map(cluster_names)
        st.dataframe(style_metrics)

        # 3️⃣ Improved Passing Network Visualization with Pitch
        st.subheader("🔗 Passing Network")
        selected_style = st.selectbox("Select Playing Style", style_distribution.index)

        # Filter passes based on the selected style
        style_passes = team_data[team_data['Cluster'] ==
                                list(cluster_names.keys())[list(cluster_names.values()).index(selected_style)]]
        
        # 🎯 Player Filter
        players_in_style = style_passes['name'].dropna().unique()
        selected_player = st.selectbox("Select Player", options=players_in_style)

        # Filter passes for the selected player
        player_passes = style_passes[style_passes['name'] == selected_player]

        # Plotting Passing Network on the Pitch
        pitch = Pitch(pitch_type='custom', pitch_length=104, pitch_width=68,
                    pitch_color='white', line_color='black', linewidth=2)
        fig, ax = pitch.draw(figsize=(12, 8))

        # Filter valid passes (with start and end coordinates)
        valid_passes = player_passes.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])

        # ⚽ Correcting the Coordinates for Accurate Scaling
        for _, row in valid_passes.iterrows():
            # Adjusting coordinates to fit the pitch
            start_x = row['start_x'] + 52
            start_y = row['start_y'] + 34
            end_x = row['end_x'] + 52
            end_y = row['end_y'] + 34

            # Draw arrows for passes
            pitch.arrows(start_x, start_y, end_x, end_y,
                        width=2, headwidth=8, headlength=6, color='blue', alpha=0.7, ax=ax)

        plt.title(f"Passing Network for {selected_style} ({', '.join(selected_team)})", fontsize=16)
        st.pyplot(fig)

        # 4️⃣ Possession Sequence Visualization
        st.subheader("🎥 Possession Sequence Visualization")

        # Possession ID Selection for the Chosen Style
        possession_ids = style_passes['possession_id'].dropna().unique()
        selected_possession = st.selectbox("Select Possession ID", possession_ids)

        # Possession Visualization Function
        def visualize_possession(df, possession_id):
            df_possession = df[df['possession_id'] == possession_id]

            # Ensure coordinates are finite
            df_possession = df_possession[
                df_possession[['start.adjCoordinates.x', 'start.adjCoordinates.y',
                                'end.adjCoordinates.x', 'end.adjCoordinates.y']].apply(np.isfinite).all(axis=1)
            ]

            pitch = Pitch(pitch_type='custom', pitch_length=104, pitch_width=68,
                          pitch_color='green', line_color='white', linewidth=2)
            fig, ax = pitch.draw()

            event_list = []

            for i in range(len(df_possession)):
                start_x = df_possession.iloc[i]['start.adjCoordinates.x'] + 52
                start_y = df_possession.iloc[i]['start.adjCoordinates.y'] + 34
                end_x = df_possession.iloc[i]['end.adjCoordinates.x'] + 52
                end_y = df_possession.iloc[i]['end.adjCoordinates.y'] + 34

                event_type = df_possession.iloc[i]['event_type']
                match_time = df_possession.iloc[i]['time']
                outcome = df_possession.iloc[i]['outcome']

                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2))
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2 + 4
                ax.text(mid_x, mid_y, str(i + 1), fontsize=10, color='black', ha='center')

                event_list.append({'No.': i + 1, 'Event': event_type, 'Outcome': outcome, 'Time': match_time})

            cluster = df_possession['Cluster'].iloc[0]
            xT = df_possession['total_xT'].iloc[0]

            plt.title(f"Possession ID: {possession_id}, Style: {cluster_names[cluster]}, xT: {xT}", color='black')
            plt.gca().set_facecolor('green')

            event_df = pd.DataFrame(event_list)
            table = plt.table(cellText=event_df.values,
                              colLabels=event_df.columns,
                              cellLoc='center',
                              loc='right',
                              bbox=[1.0, 0.0, 0.5, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(6)

            return fig

        # 🚀 Display Possession Visualization
        if selected_possession:
            fig = visualize_possession(style_passes, selected_possession)
            st.pyplot(fig)

    else:
        st.warning("Please select at least one team from the sidebar filters to see the playing style analysis.")

with tabs[3]:
    
    st.header("📊 Match Summary & Heatmaps")

    # Ensure the user selects a team
    if selected_team:
        # Ensure selected_team is always a list
        if isinstance(selected_team, str):
            selected_team = [selected_team]

        # Filter data based on the selected team
        team_data = df[df['team_name'].isin(selected_team)]

        # 1️⃣ Team Possession Heatmap
        st.subheader("🔥 Team Possession Heatmap")

        # Select which team to visualize
        selected_heatmap_team = st.selectbox("Select Team for Heatmap", options=selected_team)

        team_heatmap_data = team_data[team_data['team_name'] == selected_heatmap_team]

        # Initialize the pitch
        pitch = Pitch(pitch_type='custom', pitch_length=104, pitch_width=68, 
                      pitch_color='white', line_color='black')

        fig, ax = pitch.draw(figsize=(12, 8))

        # Adjust the coordinates
        team_heatmap_data['adjusted_x'] = team_heatmap_data['start.adjCoordinates.x'] + 52
        team_heatmap_data['adjusted_y'] = team_heatmap_data['start.adjCoordinates.y'] + 34

        # Create the heatmap
        pitch.kdeplot(
            x=team_heatmap_data['adjusted_x'],
            y=team_heatmap_data['adjusted_y'],
            ax=ax,
            cmap='coolwarm',
            fill=True,
            levels=50,
            shade=True,
            alpha=0.7,
            thresh=0.1
        )

        # Add the title
        plt.title(f"🔥 Possession Heatmap - {selected_heatmap_team}", fontsize=16)

        # Display the plot
        st.pyplot(fig)

    else:
        st.warning("Please select at least one team from the sidebar filters to see the match summary.")

    # 2️⃣ Player Heatmap
    st.subheader("🎯 Player Heatmap")

    # Select a player to visualize
    selected_player_heatmap = st.selectbox("Select Player for Heatmap", options=team_data['name'].dropna().unique())

    # Filter data for the selected player
    player_heatmap_data = team_data[team_data['name'] == selected_player_heatmap]

    # Initialize the pitch
    pitch = Pitch(pitch_type='custom', pitch_length=104, pitch_width=68,
                pitch_color='white', line_color='black')

    fig, ax = pitch.draw(figsize=(12, 8))

    # Adjust the coordinates
    player_heatmap_data['adjusted_x'] = player_heatmap_data['start.adjCoordinates.x'] + 52
    player_heatmap_data['adjusted_y'] = player_heatmap_data['start.adjCoordinates.y'] + 34

    # Create the heatmap
    pitch.kdeplot(
        x=player_heatmap_data['adjusted_x'],
        y=player_heatmap_data['adjusted_y'],
        ax=ax,
        cmap='plasma',  # Different color for player heatmap
        fill=True,
        levels=50,
        shade=True,
        alpha=0.7,
        thresh=0.1
    )

    # Add the title
    plt.title(f"🔥 Heatmap - {selected_player_heatmap}", fontsize=16)

    # Display the plot
    st.pyplot(fig)

    

with tabs[4]:
    st.header("📊 Match Summary")

    # Ensure both teams are selected
    selected_teams = df['team_name'].dropna().unique()
    team1, team2 = st.selectbox("Select Team 1", selected_teams, index=0), st.selectbox("Select Team 2", selected_teams, index=1)

    # Filter data for the selected teams
    team1_data = df[df['team_name'] == team1]
    team2_data = df[df['team_name'] == team2]

    # Calculate Match Statistics
    def calculate_stats(team_data):
        return {
            "Total Possessions": team_data['possession_id'].nunique(),
            "Total Goals": team_data['goal'].sum(),
            "Total Shots": team_data[team_data['event_type'] == 'Shot'].shape[0],
            "Total xG": team_data['total_xG'].sum(),
            "Pass Accuracy (%)": (team_data['pass_success_rate'].mean() * 100).round(2),
            "Possession Efficiency": team_data['possession_efficiency'].mean().round(2)
        }

    team1_stats = calculate_stats(team1_data)
    team2_stats = calculate_stats(team2_data)

    # Display Team Comparison
    st.subheader("🔍 Team Comparison")

    comparison_df = pd.DataFrame([team1_stats, team2_stats], index=[team1, team2])
    st.dataframe(comparison_df)

    # 2️⃣ Team Comparison Bar Chart
    st.subheader("📊 Team Stats Comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.T.plot(kind='bar', ax=ax)
    plt.title("Team Stats Comparison")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    st.pyplot(fig)

