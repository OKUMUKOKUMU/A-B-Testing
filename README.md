# A/B Testing Dashboard

![A/B Testing Dashboard](https://img.shields.io/badge/Streamlit-A/B%20Testing-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful and beautiful Streamlit dashboard for analyzing A/B test results with statistical significance testing and interactive visualizations.

## 📋 Features

- **Interactive Data Filtering**: Filter your test data by date range, user segments, and more
- **Statistical Analysis**: Automatically calculate p-values and statistical significance
- **Beautiful Visualizations**: View conversion funnels, time series charts, and comparison graphs
- **Segmentation Analysis**: Dive deeper into how different user segments respond to variants
- **Automated Recommendations**: Get actionable insights based on test results
- **Responsive Design**: Beautiful UI that works on different screen sizes

## 🖼️ Screenshots

![Dashboard Overview](https://via.placeholder.com/800x400?text=A/B+Testing+Dashboard)
![Statistical Analysis](https://via.placeholder.com/800x400?text=Statistical+Analysis)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ab-testing-dashboard.git
cd ab-testing-dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit app:
```bash
streamlit run ab_testing_app.py
```

2. Open your browser and navigate to the URL displayed in your terminal (usually `http://localhost:8501`)

3. Use the sidebar filters to adjust your analysis parameters and explore the data

## 📊 Sample Data

The app comes with a built-in sample data generator that creates realistic A/B test data with the following structure:

| Field | Description |
|-------|-------------|
| user_id | Unique identifier for each user |
| date | Date of the interaction |
| variant | Test variant (A or B) |
| device | User's device type (Mobile, Desktop, Tablet) |
| country | User's country |
| age_group | User's age group |
| viewed_page | Whether the user viewed the page (boolean) |
| converted | Whether the user converted (boolean) |
| time_spent | Time spent on the page (seconds) |
| revenue | Revenue generated (if converted) |

## 🔧 Customizing for Your Data

To use your own A/B test data:

1. Replace the `generate_sample_data()` function with code to load your own data
2. Ensure your data has similar columns or adjust the app code accordingly
3. Update the metrics and segments to match your data structure

Example for loading your own CSV data:

```python
@st.cache_data
def load_data():
    df = pd.read_csv("your_ab_test_data.csv")
    # Perform any necessary preprocessing
    return df

# Replace generate_sample_data() with load_data()
df = load_data()
```

## 📈 Understanding Statistical Significance

The app calculates statistical significance using:

- **Binary metrics** (converted, viewed_page): Z-test for proportions
- **Continuous metrics** (time_spent, revenue): T-test for independent samples

The significance level (alpha) can be adjusted in the sidebar. A lower alpha value (e.g., 0.01) makes the test more conservative, requiring stronger evidence to declare significance.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Fordrane Albert Okumu - [fordraneokumu74@gmail.com](mailto:fordraneokumu74@gmail.com)

Project Link: [https://github.com/okumuokumu/ab-testing-dashboard](https://github.com/okumuokumu/ab-testing-dashboard)
