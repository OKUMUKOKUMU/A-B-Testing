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
import io

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
    .upload-section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
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

# Function to validate uploaded data
def validate_data(df):
    required_columns = ['user_id', 'date', 'variant']
    metric_columns = ['converted', 'viewed_page', 'time_spent', 'revenue']
    
    # Check for required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        return False, f"Missing required columns: {', '.join(missing_required)}"
    
    # Check if at least one metric column exists
    available_metrics = [col for col in metric_columns if col in df.columns]
    if not available_metrics:
        return False, f"At least one metric column is required: {', '.join(metric_columns)}"
    
    # Check if variant column has at least two values (A/B or similar)
    unique_variants = df['variant'].nunique()
    if unique_variants < 2:
        return False, "Variant column must have at least two different values"
    
    # Check if date column can be converted to datetime
    try:
        pd.to_datetime(df['date'])
    except:
        return False, "Date column could not be parsed. Please ensure it's in a valid date format."
    
    return True, "Data validation successful"

# Function to download sample template
def get_sample_template():
    sample_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'variant': ['A', 'A', 'B', 'B'],
        'device': ['Mobile', 'Desktop', 'Mobile', 'Tablet'],
        'country': ['US', 'UK', 'US', 'Canada'],
        'converted': [True, False, True, True],
        'viewed_page': [True, True, True, False],
        'time_spent': [120.5, 85.2, 150.3, 95.7],
        'revenue': [45.0, 0.0, 50.5, 35.2]
    })
    return sample_df

# App Header
st.title("📊 A/B Testing Dashboard")
st.markdown("### Analyze and visualize your experiment results with ease")

# File Upload Section
st.markdown("## 📤 Data Source")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

data_source = st.radio(
    "Choose data source:",
    ("Upload your own data", "Use sample data")
)

df = None

if data_source == "Upload your own data":
    st.markdown("### Upload your data")
    st.markdown("""
    Please upload a CSV file with the following required columns:
    - `user_id`: Unique identifier for users
    - `date`: Date of the event
    - `variant`: Test variant (A, B, etc.)
    
    And at least one of these metric columns:
    - `converted`: Boolean (True/False) for conversion
    - `viewed_page`: Boolean (True/False) for page views
    - `time_spent`: Numeric value for time spent on page
    - `revenue`: Numeric value for revenue generated
    
    Optional columns for segmentation:
    - `device`: User device type
    - `country`: User country
    - `age_group`: User age group
    """)
    
    # Option to download template
    st.markdown("#### Need a template?")
    sample_template = get_sample_template()
    template_csv = sample_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV Template",
        template_csv,
        "ab_testing_template.csv",
        "text/csv",
        key='download-template'
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate the data
            is_valid, validation_message = validate_data(df)
            
            if is_valid:
                st.success(validation_message)
                
                # Convert columns to appropriate data types
                df['date'] = pd.to_datetime(df['date'])
                
                # Convert boolean columns if they exist as strings
                for col in ['converted', 'viewed_page']:
                    if col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].map({'True': True, 'true': True, '1': True, 1: True, 
                                                  'False': False, 'false': False, '0': False, 0: False})
                
                # Show data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head())
                
                # Column mapping
                st.markdown("#### Column Mapping")
                st.info("Confirm that your columns are correctly mapped for analysis")
                
                # Let user select which columns to use for analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    variant_col = st.selectbox("Variant Column", 
                                              [col for col in df.columns if df[col].nunique() <= 10],
                                              index=df.columns.get_loc('variant') if 'variant' in df.columns else 0)
                    
                    date_col = st.selectbox("Date Column",
                                          [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower()],
                                          index=df.columns.get_loc('date') if 'date' in df.columns else 0)
                
                with col2:
                    # Let user select available metric columns
                    metric_options = [col for col in df.columns if col not in [variant_col, date_col, 'user_id'] and df[col].dtype != 'object']
                    if metric_options:
                        metrics_to_analyze = st.multiselect(
                            "Metrics to Analyze",
                            options=metric_options,
                            default=[col for col in ['converted', 'time_spent', 'revenue'] if col in metric_options]
                        )
                    else:
                        st.error("No numeric metrics found in the uploaded data")
                        metrics_to_analyze = []
                
                # Rename columns if needed
                if variant_col != 'variant':
                    df = df.rename(columns={variant_col: 'variant'})
                
                if date_col != 'date':
                    df = df.rename(columns={date_col: 'date'})
            else:
                st.error(validation_message)
                df = None
                st.stop()
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            df = None
else:
    # Generate sample data
    df = generate_sample_data()
    st.info("Using sample data with simulated A/B test results")
    
    # Show data preview
    st.markdown("#### Sample Data Preview")
    st.dataframe(df.head())
    
    # Default metrics for sample data
    metrics_to_analyze = ['converted', 'revenue', 'time_spent']

st.markdown("</div>", unsafe_allow_html=True)

# If no data is available, stop rendering
if df is None:
    st.warning("Please upload a valid data file or select 'Use sample data' to continue")
    st.stop()

# Sidebar for filtering
st.sidebar.markdown("## 🔍 Filters")

# Date range filter
min_date = min(df['date']).date()
max_date = max(df['date']).date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Convert to datetime for filtering
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply date filter
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Segment filters - only show if columns exist
segment_columns = [col for col in ['country', 'device', 'age_group'] if col in df.columns]

for segment in segment_columns:
    segment_values = df[segment].unique()
    selected_values = st.sidebar.multiselect(
        f"{segment.title()}",
        options=segment_values,
        default=segment_values
    )
    
    if selected_values:
        filtered_df = filtered_df[filtered_df[segment].isin(selected_values)]

# Significance level
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

# Main metrics selection
st.sidebar.markdown("## 📈 Metrics")
if metrics_to_analyze:
    primary_metric = st.sidebar.selectbox(
        "Primary Metric",
        options=metrics_to_analyze,
        index=0
    )
else:
    st.error("No metrics available for analysis")
    st.stop()

# Experiment summary
st.markdown("## 🔬 Experiment Summary")

# Display basic experiment metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Total Users",
        f"{filtered_df['user_id'].nunique():,}",
        delta=f"{filtered_df['user_id'].nunique() - df['user_id'].nunique()}" if filtered_df['user_id'].nunique() != df['user_id'].nunique() else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    variant_counts = filtered_df['variant'].value_counts()
    variant_a = next((x for x in variant_counts.index if x in ['A', 'a', 'Control', 'control']), variant_counts.index[0])
    variant_b = next((x for x in variant_counts.index if x in ['B', 'b', 'Treatment', 'treatment', 'test']), variant_counts.index[1 if len(variant_counts.index) > 1 else 0])
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(f"Variant {variant_a} Users", f"{variant_counts.get(variant_a, 0):,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(f"Variant {variant_b} Users", f"{variant_counts.get(variant_b, 0):,}")
    st.markdown("</div>", unsafe_allow_html=True)

# Show sample of the dataframe
with st.expander("View Data Sample"):
    st.dataframe(filtered_df.head(10))

# Statistical analysis
st.markdown("## 📊 Results Analysis")

# Calculate statistical significance for selected metrics
results = []

for metric in metrics_to_analyze:
    if metric in filtered_df.columns:
        result = calculate_significance(filtered_df, metric, alpha)
        results.append(result)

# Display statistical analysis
col1, col2 = st.columns(2)

with col1:
    # Only show funnel chart if both 'viewed_page' and 'converted' exist
    if 'viewed_page' in filtered_df.columns and 'converted' in filtered_df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Conversion Funnel")
        
        # Get variant names
        variants = filtered_df['variant'].unique()
        variant_a = variants[0]
        variant_b = variants[1] if len(variants) > 1 else variants[0]
        
        # Funnel chart data
        funnel_data = {
            "Stage": ["Viewed", "Converted"],
            f"Variant {variant_a}": [
                filtered_df[(filtered_df['variant'] == variant_a) & (filtered_df['viewed_page'])].shape[0],
                filtered_df[(filtered_df['variant'] == variant_a) & (filtered_df['converted'])].shape[0]
            ],
            f"Variant {variant_b}": [
                filtered_df[(filtered_df['variant'] == variant_b) & (filtered_df['viewed_page'])].shape[0],
                filtered_df[(filtered_df['variant'] == variant_b) & (filtered_df['converted'])].shape[0]
            ]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # Create a funnel chart
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            name=f'Variant {variant_a}',
            y=funnel_df['Stage'],
            x=funnel_df[f'Variant {variant_a}'],
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.8,
            marker={"color": "#1E88E5"},
        ))
        
        fig.add_trace(go.Funnel(
            name=f'Variant {variant_b}',
            y=funnel_df['Stage'],
            x=funnel_df[f'Variant {variant_b}'],
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
    else:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Primary Metric Distribution")
        
        # Create histogram of primary metric by variant
        if primary_metric in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x=primary_metric,
                color='variant',
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
                title=f"{primary_metric.replace('_', ' ').title()} Distribution by Variant"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Primary metric '{primary_metric}' not found in data")
        
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if primary_metric in filtered_df.columns:
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
            color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
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

if results:
    # Create a table to display statistical test results
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
        if result['metric'] in ['converted', 'viewed_page']:
            # For binary metrics, show percentage
            data = {
                'Variant': [variant_a, variant_b],
                'Value': [float(result['value_a'].strip('%'))/100, float(result['value_b'].strip('%'))/100]
            }
        else:
            # For continuous metrics, show raw values
            data = {
                'Variant': [variant_a, variant_b],
                'Value': [float(result['value_a']), float(result['value_b'])]
            }
        
        comparison_df = pd.DataFrame(data)
        
        fig = px.bar(
            comparison_df,
            x='Variant',
            y='Value',
            color='Variant',
            color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
            text_auto=True
        )
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("No metric results available for analysis")

# Additional segments analysis (if needed)
segment_options = [col for col in ['device', 'country', 'age_group'] if col in filtered_df.columns]

if segment_options:
    with st.expander("View Segment Analysis"):
        selected_segment = st.selectbox("Select Segment", segment_options)
        
        # Create segment analysis
        segment_results = filtered_df.groupby([selected_segment, 'variant'])[primary_metric].mean().reset_index()
        
        # Pivot for easier visualization
        segment_pivot = segment_results.pivot(index=selected_segment, columns='variant', values=primary_metric)
        
        # Add difference columns if we have both variants
        if len(segment_pivot.columns) >= 2:
            segment_pivot['difference'] = segment_pivot[variant_b] - segment_pivot[variant_a]
            segment_pivot['percent_difference'] = (segment_pivot['difference'] / segment_pivot[variant_a]) * 100
        
        # Display segment analysis
        st.dataframe(segment_pivot.style.format({
            variant_a: '{:.4f}',
            variant_b: '{:.4f}',
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
            color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
            title=f"{primary_metric.replace('_', ' ').title()} by {selected_segment.replace('_', ' ').title()}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Recommendations
primary_result = next((r for r in results if r['metric'] == primary_metric), None)

if primary_result:
    st.markdown("## 🚀 Recommendations")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if primary_result['significant']:
        # Determine which variant performed better
        if primary_metric in ['converted', 'viewed_page']:
            value_a = float(primary_result['value_a'].strip('%')) 
            value_b = float(primary_result['value_b'].strip('%'))
        else:
            value_a = float(primary_result['value_a'])
            value_b = float(primary_result['value_b'])
            
        better_variant = variant_b if value_b > value_a else variant_a
        
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
    "Download Filtered Data as CSV",
    csv_download,
    "ab_testing_filtered_data.csv",
    "text/csv",
    key='download-csv'
)

# Generate sample data button
if data_source == "Upload your own data":
    st.sidebar.markdown("## 🔄 Sample Data")
    if st.sidebar.button("Generate Sample Data"):
        st.session_state['use_sample_data'] = True
        st.experimental_rerun()
