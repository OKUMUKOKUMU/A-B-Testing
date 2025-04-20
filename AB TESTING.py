import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import datetime
from datetime import timedelta

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
    }
    .significance {
        font-weight: bold;
        font-size: 18px;
    }
    .significant {
        color: #28a745;
    }
    .not-significant {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate sample data
@st.cache_data
def generate_sample_data(n_users=5000, seed=42):
    np.random.seed(seed)
    # Create date range for the last 30 days
    end_date = datetime.datetime.now().date()
    start_date = end_date - timedelta(days=30)
    date_range = [start_date + timedelta(days=x) for x in range(31)]
    
    # User IDs
    user_ids = np.arange(1, n_users + 1)
    
    # Create base dataframe
    data = {
        'user_id': np.random.choice(user_ids, n_users),
        'date': np.random.choice(date_range, n_users),
        'variant': np.random.choice(['A', 'B'], n_users),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_users, p=[0.6, 0.3, 0.1]),
        'country': np.random.choice(['US', 'UK', 'Canada', 'Germany', 'France'], n_users, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_users),
    }
    
    df = pd.DataFrame(data)
    
    # Add conversion metrics with different probabilities for A and B
    # Variant B has higher conversion rate
    df['viewed_page'] = np.random.rand(n_users) < 0.95
    
    conversion_prob_a = 0.12
    conversion_prob_b = 0.15
    
    df['converted'] = np.where(
        df['variant'] == 'A',
        np.random.rand(n_users) < conversion_prob_a,
        np.random.rand(n_users) < conversion_prob_b
    )
    
    # Add time spent metrics - B has slightly longer engagement
    df['time_spent'] = np.where(
        df['variant'] == 'A',
        np.random.normal(120, 30, n_users),
        np.random.normal(135, 30, n_users)
    )
    
    # Add revenue data for those who converted
    df['revenue'] = np.where(
        df['converted'],
        np.random.normal(50, 15, n_users),
        0
    )
    
    # Ensure revenue is 0 for non-conversions
    df.loc[~df['converted'], 'revenue'] = 0
    
    # Round numeric columns for cleaner display
    df['time_spent'] = df['time_spent'].round(2)
    df['revenue'] = df['revenue'].round(2)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# Function to calculate statistical significance
def calculate_significance(df, metric, alpha=0.05):
    a_data = df[df['variant'] == 'A'][metric]
    b_data = df[df['variant'] == 'B'][metric]
    
    # For binary metrics (converted, clicked, etc.)
    if metric in ['converted', 'viewed_page']:
        # Perform z-test for proportions
        n_a = len(a_data)
        n_b = len(b_data)
        
        p_a = a_data.mean()
        p_b = b_data.mean()
        
        # Pooled proportion
        p_pooled = (a_data.sum() + b_data.sum()) / (n_a + n_b)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
        
        # Z-score
        z = (p_b - p_a) / se
        
        # P-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        result = {
            'metric': metric,
            'value_a': f"{p_a:.2%}",
            'value_b': f"{p_b:.2%}",
            'absolute_diff': f"{(p_b - p_a):.2%}",
            'relative_diff': f"{((p_b - p_a) / p_a):.2%}" if p_a > 0 else "N/A",
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence': f"{(1-alpha)*100:.0f}%"
        }
    
    # For continuous metrics (time_spent, revenue, etc.)
    else:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(a_data, b_data, equal_var=False)
        
        result = {
            'metric': metric,
            'value_a': f"{a_data.mean():.2f}",
            'value_b': f"{b_data.mean():.2f}",
            'absolute_diff': f"{(b_data.mean() - a_data.mean()):.2f}",
            'relative_diff': f"{((b_data.mean() - a_data.mean()) / a_data.mean()):.2%}" if a_data.mean() > 0 else "N/A",
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence': f"{(1-alpha)*100:.0f}%"
        }
    
    return result

# Generate sample data
df = generate_sample_data()

# App Header
st.title("📊 A/B Testing Dashboard")
st.markdown("### Analyze and visualize your experiment results with ease")

# Sidebar for filtering
st.sidebar.markdown("## 🔍 Filters")

# Date range filter
start_date = st.sidebar.date_input("Start Date", min(df['date']).date())
end_date = st.sidebar.date_input("End Date", max(df['date']).date())

# Convert to datetime for filtering
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply date filter
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Segment filters
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=df['country'].unique(),
    default=df['country'].unique()
)

selected_devices = st.sidebar.multiselect(
    "Devices",
    options=df['device'].unique(),
    default=df['device'].unique()
)

selected_age_groups = st.sidebar.multiselect(
    "Age Groups",
    options=df['age_group'].unique(),
    default=df['age_group'].unique()
)

# Apply segment filters
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
if selected_devices:
    filtered_df = filtered_df[filtered_df['device'].isin(selected_devices)]
if selected_age_groups:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age_groups)]

# Significance level
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

# Main metrics selection
st.sidebar.markdown("## 📈 Metrics")
primary_metric = st.sidebar.selectbox(
    "Primary Metric",
    options=['converted', 'revenue', 'time_spent'],
    index=0
)

# Experiment summary
st.markdown("## 🔬 Experiment Summary")

# Display basic experiment metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Total Users",
        f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    variant_counts = filtered_df['variant'].value_counts()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Variant A Users", f"{variant_counts.get('A', 0):,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Variant B Users", f"{variant_counts.get('B', 0):,}")
    st.markdown("</div>", unsafe_allow_html=True)

# Show sample of the dataframe
with st.expander("View Sample Data"):
    st.dataframe(filtered_df.head(10))

# Statistical analysis
st.markdown("## 📊 Results Analysis")

# Calculate statistical significance for key metrics
metrics_to_analyze = ['converted', 'revenue', 'time_spent']
results = []

for metric in metrics_to_analyze:
    result = calculate_significance(filtered_df, metric, alpha)
    results.append(result)

# Display statistical analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### Conversion Funnel")
    
    # Funnel chart data
    funnel_data = {
        "Stage": ["Viewed", "Converted"],
        "Variant A": [
            filtered_df[(filtered_df['variant'] == 'A') & (filtered_df['viewed_page'])].shape[0],
            filtered_df[(filtered_df['variant'] == 'A') & (filtered_df['converted'])].shape[0]
        ],
        "Variant B": [
            filtered_df[(filtered_df['variant'] == 'B') & (filtered_df['viewed_page'])].shape[0],
            filtered_df[(filtered_df['variant'] == 'B') & (filtered_df['converted'])].shape[0]
        ]
    }
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # Create a funnel chart
    fig = go.Figure()
    
    fig.add_trace(go.Funnel(
        name='Variant A',
        y=funnel_df['Stage'],
        x=funnel_df['Variant A'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": "#1E88E5"},
    ))
    
    fig.add_trace(go.Funnel(
        name='Variant B',
        y=funnel_df['Stage'],
        x=funnel_df['Variant B'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": "#FFC107"},
    ))
    
    fig.update_layout(
        funnelmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### Primary Metric Over Time")
    
    # Group by date and variant, calculate mean of primary metric
    time_series = filtered_df.groupby([pd.Grouper(key='date', freq='D'), 'variant'])[primary_metric].mean().reset_index()
    
    # Create time series chart
    fig = px.line(
        time_series,
        x='date',
        y=primary_metric,
        color='variant',
        color_discrete_map={'A': '#1E88E5', 'B': '#FFC107'},
        title=f"{primary_metric.replace('_', ' ').title()} over Time"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=primary_metric.replace('_', ' ').title(),
        legend_title="Variant",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Display statistical test results
st.markdown("## 📈 Statistical Significance")

# Create a table to display statistical test results
results_df = pd.DataFrame(results)

for i, result in enumerate(results):
    metric_name = result['metric'].replace('_', ' ').title()
    
    st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f"### {metric_name}")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown(f"**Variant A:** {result['value_a']}")
        st.markdown(f"**Variant B:** {result['value_b']}")
    
    with col2:
        st.markdown(f"**Absolute Diff:** {result['absolute_diff']}")
        st.markdown(f"**Relative Diff:** {result['relative_diff']}")
    
    with col3:
        st.markdown(f"**P-value:** {result['p_value']:.4f}")
        st.markdown(f"**Confidence Level:** {result['confidence']}")
        
        if result['significant']:
            st.markdown(f'<p class="significance significant">✅ Statistically Significant</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="significance not-significant">❌ Not Statistically Significant</p>', unsafe_allow_html=True)
    
    # Add bar chart comparison
    if result['metric'] in ['converted']:
        # For binary metrics, show percentage
        data = {
            'Variant': ['A', 'B'],
            'Value': [float(result['value_a'].strip('%'))/100, float(result['value_b'].strip('%'))/100]
        }
    else:
        # For continuous metrics, show raw values
        data = {
            'Variant': ['A', 'B'],
            'Value': [float(result['value_a']), float(result['value_b'])]
        }
    
    comparison_df = pd.DataFrame(data)
    
    fig = px.bar(
        comparison_df,
        x='Variant',
        y='Value',
        color='Variant',
        color_discrete_map={'A': '#1E88E5', 'B': '#FFC107'},
        text_auto=True
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Additional segments analysis (if needed)
with st.expander("View Segment Analysis"):
    segment_options = ['device', 'country', 'age_group']
    selected_segment = st.selectbox("Select Segment", segment_options)
    
    # Create segment analysis
    segment_results = filtered_df.groupby([selected_segment, 'variant'])[primary_metric].mean().reset_index()
    
    # Pivot for easier visualization
    segment_pivot = segment_results.pivot(index=selected_segment, columns='variant', values=primary_metric)
    segment_pivot['difference'] = segment_pivot['B'] - segment_pivot['A']
    segment_pivot['percent_difference'] = (segment_pivot['difference'] / segment_pivot['A']) * 100
    
    # Display segment analysis
    st.dataframe(segment_pivot.style.format({
        'A': '{:.4f}',
        'B': '{:.4f}',
        'difference': '{:.4f}',
        'percent_difference': '{:.2f}%'
    }))
    
    # Visualize segment analysis
    fig = px.bar(
        segment_results,
        x=selected_segment,
        y=primary_metric,
        color='variant',
        barmode='group',
        color_discrete_map={'A': '#1E88E5', 'B': '#FFC107'},
        title=f"{primary_metric.replace('_', ' ').title()} by {selected_segment.replace('_', ' ').title()}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Recommendations
st.markdown("## 🚀 Recommendations")

# Generate automated recommendations based on results
primary_result = next((r for r in results if r['metric'] == primary_metric), None)

if primary_result:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if primary_result['significant']:
        better_variant = 'B' if float(primary_result['value_b'].replace('%', '')) > float(primary_result['value_a'].replace('%', '')) else 'A'
        
        st.markdown(f"""
        ### Main Recommendation
        
        Based on the analysis, Variant {better_variant} shows a statistically significant improvement in {primary_metric.replace('_', ' ')} 
        with a {primary_result['confidence']} confidence level. We recommend implementing Variant {better_variant}.
        
        ### Next Steps
        
        1. Implement Variant {better_variant} for all users
        2. Continue monitoring key metrics to ensure sustained improvement
        3. Consider further optimization opportunities based on segment analysis
        """)
    else:
        st.markdown("""
        ### Main Recommendation
        
        The test did not show statistically significant results for the primary metric. We recommend:
        
        1. Extend the test duration to collect more data
        2. Consider a more impactful variation for testing
        3. Analyze segments to identify areas where one variant might perform better
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# App footer
st.markdown("---")
st.markdown("A/B Testing Dashboard | Created with Streamlit")

# Download options
st.sidebar.markdown("## 📥 Download")
csv_download = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download Data as CSV",
    csv_download,
    "ab_testing_data.csv",
    "text/csv",
    key='download-csv'
)
