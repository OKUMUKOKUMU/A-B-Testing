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
import base64

# Set page configuration
st.set_page_config(
    page_title="Universal A/B Testing Dashboard",
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
        max-width: 1400px;
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
    .file-uploader {
        border: 2px dashed #1E88E5;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    .data-preview {
        max-height: 300px;
        overflow-y: auto;
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
    df['viewed_page'] = np.random.rand(n_users) < 0.95
    
    conversion_prob_a = 0.12
    conversion_prob_b = 0.15
    
    df['converted'] = np.where(
        df['variant'] == 'A',
        np.random.rand(n_users) < conversion_prob_a,
        np.random.rand(n_users) < conversion_prob_b
    )
    
    # Add time spent metrics
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
@st.cache_data
def calculate_significance(df, metric, variant_col='variant', alpha=0.05):
    variants = df[variant_col].unique()
    if len(variants) < 2:
        return None
    
    variant_a, variant_b = variants[0], variants[1]
    a_data = df[df[variant_col] == variant_a][metric]
    b_data = df[df[variant_col] == variant_b][metric]
    
    # For binary metrics (converted, clicked, etc.)
    if df[metric].dtype == 'bool' or (df[metric].nunique() == 2 and set(df[metric].unique()).issubset({0, 1})):
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
            'variant_a': variant_a,
            'variant_b': variant_b,
            'value_a': f"{p_a:.2%}",
            'value_b': f"{p_b:.2%}",
            'absolute_diff': f"{(p_b - p_a):.2%}",
            'relative_diff': f"{((p_b - p_a) / p_a):.2%}" if p_a > 0 else "N/A",
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence': f"{(1-alpha)*100:.0f}%",
            'test_type': 'proportions_ztest'
        }
    
    # For continuous metrics (time_spent, revenue, etc.)
    else:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(a_data, b_data, equal_var=False)
        
        result = {
            'metric': metric,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'value_a': f"{a_data.mean():.2f}",
            'value_b': f"{b_data.mean():.2f}",
            'absolute_diff': f"{(b_data.mean() - a_data.mean()):.2f}",
            'relative_diff': f"{((b_data.mean() - a_data.mean()) / a_data.mean()):.2%}" if a_data.mean() > 0 else "N/A",
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence': f"{(1-alpha)*100:.0f}%",
            'test_type': 't-test_ind'
        }
    
    return result

# Function to validate uploaded data
def validate_data(df):
    # Check if dataframe is empty
    if df.empty:
        return False, "The uploaded file is empty"
    
    # Check if there are enough columns
    if len(df.columns) < 3:
        return False, "The file must contain at least user_id, date, and variant columns"
    
    # Check if we have at least some numeric or boolean columns for metrics
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    if len(numeric_cols) < 1:
        return False, "The file must contain at least one numeric or boolean column for metrics"
    
    return True, "Data validation passed"

# Function to get sample template for download
def get_sample_template():
    sample_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'],
        'variant': ['A', 'A', 'B', 'B', 'A', 'B'],
        'device': ['Mobile', 'Desktop', 'Mobile', 'Tablet', 'Desktop', 'Mobile'],
        'country': ['US', 'UK', 'US', 'Canada', 'Germany', 'France'],
        'converted': [True, False, True, True, False, True],
        'viewed_page': [True, True, True, False, True, True],
        'time_spent': [120.5, 85.2, 150.3, 95.7, 110.0, 130.4],
        'revenue': [45.0, 0.0, 50.5, 35.2, 0.0, 60.0],
        'clicks': [5, 3, 7, 2, 4, 6],
        'engagement_score': [0.75, 0.45, 0.85, 0.35, 0.60, 0.90]
    })
    return sample_df

# Function to convert df to csv for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to analyze metric distributions
def analyze_distributions(df, metric, variant_col='variant'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    sns.histplot(
        data=df,
        x=metric,
        hue=variant_col,
        element='step',
        stat='density',
        common_norm=False,
        kde=True,
        ax=ax1
    )
    ax1.set_title(f'Distribution of {metric}')
    
    # Boxplot
    sns.boxplot(
        data=df,
        x=variant_col,
        y=metric,
        ax=ax2
    )
    ax2.set_title(f'Boxplot by Variant')
    
    plt.tight_layout()
    return fig

# Main App
def main():
    # App Header
    st.title("📊 Universal A/B Testing Dashboard")
    st.markdown("### Analyze and visualize your experiment results for any type of A/B test")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis_df' not in st.session_state:
        st.session_state.analysis_df = None
    if 'primary_metric' not in st.session_state:
        st.session_state.primary_metric = None
    
    # File Upload Section
    st.markdown("## 📤 Data Source")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source:",
        ("Upload your own data", "Use sample data"),
        horizontal=True
    )
    
    if data_source == "Upload your own data":
        st.markdown("### Upload your A/B test data")
        st.markdown("""
        Upload a CSV, Excel, or JSON file containing your A/B test results. The data should include:
        - A column identifying the test variants (A/B groups)
        - One or more metric columns to compare
        - Optional: User identifiers, dates, and segmentation columns
        """)
        
        # File format selection
        file_format = st.radio(
            "Select file format:",
            ("CSV", "Excel", "JSON"),
            horizontal=True
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            f"Choose a {file_format} file",
            type=['csv', 'xlsx', 'json'],
            key='file_uploader'
        )
        
        # Download template
        st.markdown("#### Need a template?")
        sample_template = get_sample_template()
        template_csv = convert_df_to_csv(sample_template)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download CSV Template",
                template_csv,
                "ab_testing_template.csv",
                "text/csv",
                key='download-csv-template'
            )
        with col2:
            # Excel template download
            excel_buffer = io.BytesIO()
            sample_template.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            b64 = base64.b64encode(excel_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="ab_testing_template.xlsx">Download Excel Template</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                # Read file based on format
                if file_format == "CSV":
                    df = pd.read_csv(uploaded_file)
                elif file_format == "Excel":
                    df = pd.read_excel(uploaded_file)
                elif file_format == "JSON":
                    df = pd.read_json(uploaded_file)
                
                # Validate the data
                is_valid, validation_message = validate_data(df)
                
                if is_valid:
                    st.success(validation_message)
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head())
                    
                    # Column mapping
                    st.markdown("#### Configure Your Analysis")
                    st.info("Map your columns to the required fields for analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Let user select variant column
                        variant_col = st.selectbox(
                            "Select Variant Column (A/B groups)", 
                            options=df.columns.tolist(),
                            index=0,
                            help="Column that identifies which test group each user is in (typically 'A' and 'B')"
                        )
                        
                        # Let user select date column (optional)
                        date_col_options = ['None'] + df.columns.tolist()
                        date_col = st.selectbox(
                            "Select Date Column (optional)",
                            options=date_col_options,
                            index=0,
                            help="Optional column with dates for time series analysis"
                        )
                        
                        if date_col != 'None':
                            try:
                                df[date_col] = pd.to_datetime(df[date_col])
                                st.success(f"Successfully parsed {date_col} as date")
                            except:
                                st.warning(f"Could not parse {date_col} as date. Will not use for time analysis.")
                                date_col = 'None'
                    
                    with col2:
                        # Get numeric and boolean columns as potential metric columns
                        metric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
                        # Remove variant column if it's numeric
                        if variant_col in metric_cols:
                            metric_cols.remove(variant_col)
                        
                        if metric_cols:
                            metrics_to_analyze = st.multiselect(
                                "Select Metrics to Analyze",
                                options=metric_cols,
                                default=metric_cols[:3],  # Select first 3 metrics by default
                                help="Choose which metrics you want to compare between variants"
                            )
                        else:
                            st.error("No numeric or boolean metrics found in the uploaded data")
                            metrics_to_analyze = []
                    
                    # Check that we have at least two variants
                    if df[variant_col].nunique() < 2:
                        st.error(f"The variant column '{variant_col}' must have at least two different values")
                        st.session_state.df = None
                    else:
                        # Prepare analysis dataframe
                        analysis_df = df.copy()
                        
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.analysis_df = analysis_df
                        st.session_state.variant_col = variant_col
                        st.session_state.date_col = date_col if date_col != 'None' else None
                        st.session_state.metrics_to_analyze = metrics_to_analyze
                        
                        # Set primary metric
                        if metrics_to_analyze:
                            st.session_state.primary_metric = metrics_to_analyze[0]
                else:
                    st.error(validation_message)
                    st.session_state.df = None
            except Exception as e:
                st.error(f"Error reading the file: {str(e)}")
                st.session_state.df = None
    else:
        # Generate sample data
        df = generate_sample_data()
        st.info("Using sample data with simulated A/B test results")
        
        # Show data preview
        st.markdown("#### Sample Data Preview")
        st.dataframe(df.head())
        
        # Store in session state
        st.session_state.df = df
        st.session_state.analysis_df = df.copy()
        st.session_state.variant_col = 'variant'
        st.session_state.date_col = 'date'
        st.session_state.metrics_to_analyze = ['converted', 'revenue', 'time_spent']
        st.session_state.primary_metric = 'converted'
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If no data is available, stop rendering
    if st.session_state.df is None:
        st.warning("Please upload a valid data file or select 'Use sample data' to continue")
        st.stop()
    
    # Sidebar for filtering and configuration
    st.sidebar.markdown("## 🔍 Filters & Configuration")
    
    # Get analysis parameters from session state
    analysis_df = st.session_state.analysis_df
    variant_col = st.session_state.variant_col
    date_col = st.session_state.date_col
    metrics_to_analyze = st.session_state.metrics_to_analyze
    
    # Date range filter (if date column exists)
    if date_col:
        try:
            min_date = analysis_df[date_col].min().date()
            max_date = analysis_df[date_col].max().date()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date)
            with col2:
                end_date = st.date_input("End Date", max_date)
            
            # Convert to datetime for filtering
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Apply date filter
            filtered_df = analysis_df[(analysis_df[date_col] >= start_date) & (analysis_df[date_col] <= end_date)]
        except:
            st.sidebar.warning("Could not filter by date")
            filtered_df = analysis_df
    else:
        filtered_df = analysis_df
    
    # Segment filters - only show if columns exist
    segment_columns = [col for col in ['country', 'device', 'age_group'] if col in filtered_df.columns]
    
    for segment in segment_columns:
        segment_values = filtered_df[segment].unique().tolist()
        selected_values = st.sidebar.multiselect(
            f"Filter by {segment}",
            options=segment_values,
            default=segment_values
        )
        
        if selected_values:
            filtered_df = filtered_df[filtered_df[segment].isin(selected_values)]
    
    # Significance level
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    # Primary metric selection
    st.sidebar.markdown("## 📈 Primary Metric")
    if metrics_to_analyze:
        primary_metric = st.sidebar.selectbox(
            "Select Primary Metric",
            options=metrics_to_analyze,
            index=0
        )
        st.session_state.primary_metric = primary_metric
    
    # Get variant values
    variants = filtered_df[variant_col].unique().tolist()
    if len(variants) < 2:
        st.error("Need at least two variants for comparison")
        st.stop()
    
    variant_a = variants[0]
    variant_b = variants[1] if len(variants) > 1 else variants[0]
    
    # Experiment summary
    st.markdown("## 🔬 Experiment Summary")
    
    # Display basic experiment metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'user_id' in filtered_df.columns:
            total_users = filtered_df['user_id'].nunique()
        else:
            total_users = len(filtered_df)
            
        st.metric(
            "Total Users/Rows",
            f"{total_users:,}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        variant_counts = filtered_df[variant_col].value_counts()
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(f"Variant {variant_a} Count", f"{variant_counts.get(variant_a, 0):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(f"Variant {variant_b} Count", f"{variant_counts.get(variant_b, 0):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show sample of the dataframe
    with st.expander("View Filtered Data Sample"):
        st.dataframe(filtered_df.head(10))
    
    # Statistical analysis
    st.markdown("## 📊 Results Analysis")
    
    # Calculate statistical significance for selected metrics
    results = []
    
    for metric in metrics_to_analyze:
        if metric in filtered_df.columns:
            result = calculate_significance(filtered_df, metric, variant_col, alpha)
            if result:
                results.append(result)
    
    # Display statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution analysis
        if st.session_state.primary_metric in filtered_df.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"### {st.session_state.primary_metric.replace('_', ' ').title()} Distribution")
            
            # Create distribution plots
            try:
                fig = analyze_distributions(filtered_df, st.session_state.primary_metric, variant_col)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not create distribution plots: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Time series analysis (if date column exists)
        if date_col and st.session_state.primary_metric in filtered_df.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"### {st.session_state.primary_metric.replace('_', ' ').title()} Over Time")
            
            try:
                # Group by date and variant, calculate mean of primary metric
                time_series = filtered_df.groupby([pd.Grouper(key=date_col, freq='D'), variant_col])[st.session_state.primary_metric].mean().reset_index()
                
                # Create time series chart
                fig = px.line(
                    time_series,
                    x=date_col,
                    y=st.session_state.primary_metric,
                    color=variant_col,
                    color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
                    title=f"{st.session_state.primary_metric.replace('_', ' ').title()} over Time"
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=st.session_state.primary_metric.replace('_', ' ').title(),
                    legend_title="Variant",
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create time series chart: {e}")
            
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
                st.markdown(f"**Variant {result['variant_a']}:** {result['value_a']}")
                st.markdown(f"**Variant {result['variant_b']}:** {result['value_b']}")
            
            with col2:
                st.markdown(f"**Absolute Diff:** {result['absolute_diff']}")
                st.markdown(f"**Relative Diff:** {result['relative_diff']}")
            
            with col3:
                st.markdown(f"**P-value:** {result['p_value']:.4f}")
                st.markdown(f"**Confidence Level:** {result['confidence']}")
                st.markdown(f"**Test Type:** {result['test_type']}")
                
                if result['significant']:
                    st.markdown(f'<p class="significance significant">✅ Statistically Significant</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="significance not-significant">❌ Not Statistically Significant</p>', unsafe_allow_html=True)
            
            # Add bar chart comparison
            try:
                if result['metric'] in filtered_df.columns:
                    # For binary metrics, show percentage
                    if filtered_df[result['metric']].dtype == 'bool' or (filtered_df[result['metric']].nunique() == 2 and set(filtered_df[result['metric']].unique()).issubset({0, 1})):
                        # For binary metrics, show percentage
                        data = {
                            'Variant': [result['variant_a'], result['variant_b']],
                            'Value': [float(result['value_a'].strip('%'))/100, float(result['value_b'].strip('%'))/100]
                        }
                    else:
                        # For continuous metrics, show raw values
                        data = {
                            'Variant': [result['variant_a'], result['variant_b']],
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
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_title=metric_name
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create comparison chart: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No metric results available for analysis")
    
    # Additional segments analysis (if needed)
    segment_options = [col for col in ['device', 'country', 'age_group'] if col in filtered_df.columns]
    
    if segment_options:
        with st.expander("🔍 View Segment Analysis"):
            selected_segment = st.selectbox("Select Segment for Analysis", segment_options)
            
            # Create segment analysis
            try:
                segment_results = filtered_df.groupby([selected_segment, variant_col])[st.session_state.primary_metric].mean().reset_index()
                
                # Pivot for easier visualization
                segment_pivot = segment_results.pivot(index=selected_segment, columns=variant_col, values=st.session_state.primary_metric)
                
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
                    y=st.session_state.primary_metric,
                    color=variant_col,
                    barmode='group',
                    color_discrete_map={variant_a: '#1E88E5', variant_b: '#FFC107'},
                    title=f"{st.session_state.primary_metric.replace('_', ' ').title()} by {selected_segment.replace('_', ' ').title()}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create segment analysis: {e}")
    
    # Recommendations
    primary_result = next((r for r in results if r['metric'] == st.session_state.primary_metric), None)
    
    if primary_result:
        st.markdown("## 🚀 Recommendations")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if primary_result['significant']:
            # Determine which variant performed better
            try:
                if st.session_state.primary_metric in filtered_df.columns:
                    if filtered_df[st.session_state.primary_metric].dtype == 'bool' or (filtered_df[st.session_state.primary_metric].nunique() == 2 and set(filtered_df[st.session_state.primary_metric].unique()).issubset({0, 1})):
                        value_a = float(primary_result['value_a'].strip('%')) 
                        value_b = float(primary_result['value_b'].strip('%'))
                    else:
                        value_a = float(primary_result['value_a'])
                        value_b = float(primary_result['value_b'])
                        
                    better_variant = variant_b if value_b > value_a else variant_a
                    
                    st.markdown(f"""
                    ### Main Recommendation
                    
                    Based on the analysis, **Variant {better_variant}** shows a statistically significant improvement in **{st.session_state.primary_metric.replace('_', ' ')}** 
                    with a **{primary_result['confidence']}** confidence level (p-value = {primary_result['p_value']:.4f}). 
                    
                    **Recommendation:** Implement Variant {better_variant} as it performed better with a **{primary_result['absolute_diff']}** absolute difference.
                    
                    ### Next Steps
                    
                    1. **Implement** Variant {better_variant} for all users
                    2. **Monitor** key metrics to ensure sustained improvement
                    3. **Analyze** segment performance for optimization opportunities
                    4. **Document** the experiment results and learnings
                    """)
            except Exception as e:
                st.error(f"Error generating recommendation: {e}")
        else:
            st.markdown(f"""
            ### No Clear Winner
            
            The test did **not** show statistically significant differences for the primary metric **{st.session_state.primary_metric.replace('_', ' ')}** 
            (p-value = {primary_result['p_value']:.4f}).
            
            **Recommendations:**
            
            1. **Continue** the test with more data if possible
            2. **Check** if segments show different patterns
            3. **Review** your test design and implementation
            4. **Consider** other metrics that might show significance
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download results
    st.markdown("## 📥 Download Results")
    if results:
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Prepare download options
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Results as CSV",
                data=convert_df_to_csv(results_df),
                file_name="ab_test_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            excel_buffer = io.BytesIO()
            results_df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            b64 = base64.b64encode(excel_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="ab_test_results.xlsx">Download Results as Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
