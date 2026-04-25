# Correlation Calculator & Price Fetcher

A professional-grade quantitative analysis tool built with Streamlit for fetching financial price data and calculating advanced benchmark-relative metrics.

## 🚀 Features

- **Multi-Source Price Fetching**: Supports yfinance, Polygon.io, and Alpha Vantage with intelligent provider routing.
- **Equity & Options Support**: Fetch historical daily OHLCV data for stocks and Polygon-format option tickers.
- **Advanced Quantitative Analysis**:
  - Benchmark-relative performance metrics (Alpha, Beta, Sharpe Ratio, Information Ratio).
  - Correlation matrices (Log-return and Simple-return).
  - Rolling correlation analysis.
  - **Winsorization**: Handle outliers by clipping returns at specified percentiles.
  - **Decay Weighting**: Apply exponential decay to emphasize recent observations.
  - **Regime Filtering**: Analyze correlations specifically in Bull or Bear market regimes.
  - **Earnings Impact Removal**: Automatically filter out price action around earnings dates to isolate pure correlation.
- **Interactive Dashboard**: Modern UI with real-time charts, metrics, and data export (CSV).

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **API Clients**: `yfinance`, `requests`
- **Environment Management**: `python-dotenv`

## 📋 Prerequisites

- Python 3.8+
- API Keys for extended data:
  - [Polygon.io](https://polygon.io/)
  - [Alpha Vantage](https://www.alphavantage.co/)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Correlation-Calculator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   POLYGON_API_KEY=your_polygon_api_key_here
   ALPHAVANTAGE_API_KEY=your_alpha_vantage_key_here
   ```

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to start using the tool.

## 🚀 Deployment (Streamlit Community Cloud)

1. **Push to GitHub**: Ensure your code (including `requirements.txt` and `.gitignore`) is pushed to a public or private GitHub repository.
2. **Sign in to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. **Deploy App**:
   - Click **"New app"**.
   - Select your repository, branch, and main file path (`app.py`).
4. **Configure Secrets**:
   Since `.env` is ignored by Git, you must add your API keys manually:
   - In the Streamlit Cloud dashboard, go to your app settings.
   - Select **"Secrets"**.
   - Add your keys in TOML format:
     ```toml
     POLYGON_API_KEY = "your_key_here"
     ALPHAVANTAGE_API_KEY = "your_key_here"
     ```
5. **Launch**: Click **"Deploy!"**. Streamlit will install dependencies from `requirements.txt` and launch your app.

## 📂 Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `fetch_prices.py`: Backend logic for multi-provider price fetching.
- `quant_analysis.py`: Quantitative metrics and correlation engine.
- `regression_v3.py`: Advanced regression and statistical analysis.
- `requirements.txt`: Project dependencies.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
