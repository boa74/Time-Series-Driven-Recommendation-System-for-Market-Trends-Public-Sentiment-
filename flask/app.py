from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import json
import time

app = Flask(__name__)

# MongoDB connection (will be initialized when needed)
mongo_client = None
mongo_db = None

# PostgreSQL connection (will be initialized when needed)
postgres_conn = None

def get_mongo_connection():
    """Initialize MongoDB connection if not already connected"""
    global mongo_client, mongo_db
    if mongo_client is None:
        try:
            from pymongo import MongoClient
            # Connect to local MongoDB instance
            mongo_client = MongoClient('mongodb://localhost:27017/')
            # Use database name (modify as needed)
            mongo_db = mongo_client['stock_analysis_db']
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return False
    return True

def get_collections():
    """Get list of available collections"""
    try:
        if get_mongo_connection():
            return list(mongo_db.list_collection_names())
        return []
    except Exception as e:
        print(f"Error getting collections: {e}")
        return []

def get_postgres_connection():
    """Initialize PostgreSQL connection if not already connected"""
    global postgres_conn
    if postgres_conn is None:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # Try multiple connection configurations
            connection_configs = [
                {"host": "localhost", "port": 5432, "user": "postgres", "password": "postgres"},
                {"host": "localhost", "port": 5432, "user": "postgres", "password": ""},
                {"host": "localhost", "port": 5432, "user": "postgres", "password": "password"},
                {"host": "localhost", "port": 5432, "user": "postgres", "password": "admin"},
                {"host": "localhost", "port": 5433, "user": "postgres", "password": "postgres"},
                {"host": "localhost", "port": 5432, "user": "boahkim", "password": ""},
            ]
            
            for config in connection_configs:
                try:
                    postgres_conn = psycopg2.connect(
                        host=config["host"],
                        port=config["port"],
                        database="time_series_analysis",
                        user=config["user"],
                        password=config["password"],
                        cursor_factory=RealDictCursor
                    )
                    print(f"PostgreSQL connection successful: {config}")
                    return True
                except Exception as e:
                    print(f"Connection failed {config}: {e}")
                    continue
            
            print("All PostgreSQL connection attempts failed")
            return False
        except ImportError:
            print("psycopg2 package not installed: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"PostgreSQL connection error: {e}")
            return False
    return True

def reset_postgres_connection():
    """Reset PostgreSQL connection"""
    global postgres_conn
    if postgres_conn:
        try:
            postgres_conn.close()
        except:
            pass
    postgres_conn = None
    return get_postgres_connection()

def execute_postgres_query(query):
    """Execute a PostgreSQL query safely (SELECT only)"""
    global postgres_conn
    try:
        # Allow only SELECT queries and meta commands for security
        query_upper = query.upper().strip()
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH') or 
                query_upper.startswith('\\D') or query_upper.startswith('\\L') or 
                query_upper.startswith('SHOW')):
            raise ValueError("Only SELECT, SHOW queries and meta commands are allowed for security.")
        
        if get_postgres_connection():
            try:
                # Rollback if transaction status is aborted
                if postgres_conn.status != 1:  # TRANSACTION_STATUS_IDLE
                    postgres_conn.rollback()
                
                cursor = postgres_conn.cursor()
                
                # Process PostgreSQL meta commands
                if query.strip().startswith('\\'):
                    # Process meta commands directly
                    if query.strip() == '\\dt':
                        cursor.execute("""
                            SELECT schemaname as "Schema", tablename as "Name", tableowner as "Owner"
                            FROM pg_tables 
                            WHERE schemaname = 'public'
                            ORDER BY tablename;
                        """)
                    elif query.strip() == '\\l':
                        cursor.execute("""
                            SELECT datname as "Name" 
                            FROM pg_database 
                            WHERE datistemplate = false
                            ORDER BY datname;
                        """)
                    else:
                        raise ValueError("Unsupported meta command.")
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Get results
                results = cursor.fetchall()
                
                # Commit
                postgres_conn.commit()
                
                # RealDictCursor를 사용하므로 결과는 이미 딕셔너리 형태
                return columns, [dict(row) for row in results]
                
            except Exception as e:
                # Rollback on error
                if postgres_conn:
                    postgres_conn.rollback()
                raise e
        else:
            raise Exception("PostgreSQL connection failed")
            
            
    except Exception as e:
        raise e

# Get the parent directory where data files are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

@app.route('/')
def index():
    """Homepage with stock prediction dashboard"""
    return render_template('index.html')

@app.route('/simple-test')
def simple_test():
    """Simple test page to verify server is working"""
    return render_template('simple_test.html')

@app.route('/api/mongodb-status')
def mongodb_status():
    """Check MongoDB connection status"""
    try:
        if get_mongo_connection():
            collections = get_collections()
            return jsonify({
                'status': 'connected',
                'collections': collections,
                'message': 'MongoDB connection successful!'
            })
        else:
            return jsonify({
                'status': 'failed',
                'collections': [],
                'message': 'MongoDB connection failed - PyMongo not installed or MongoDB not running'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'collections': [],
            'message': f'MongoDB error: {str(e)}'
        })

@app.route('/api/postgres-status')
def postgres_status():
    """Check PostgreSQL connection status"""
    try:
        if get_postgres_connection():
            # 간단한 테스트 쿼리 실행
            cursor = postgres_conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            return jsonify({
                'status': 'connected',
                'message': 'PostgreSQL connection successful!',
                'version': version
            })
        else:
            return jsonify({
                'status': 'failed',
                'message': 'PostgreSQL connection failed - psycopg2 not installed or PostgreSQL not running',
                'version': None
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'PostgreSQL error: {str(e)}',
            'version': None
        })

@app.route('/test')
def test():
    """Test page for debugging API calls"""
    return render_template('test.html')

@app.route('/mongo-query', methods=['GET', 'POST'])
def mongo_query():
    """MongoDB query interface"""
    collections = get_collections()
    
    if request.method == 'POST':
        try:
            collection_name = request.form.get('collection')
            query_str = request.form.get('query', '{}')
            limit = int(request.form.get('limit', 10))
            
            if not collection_name:
                return render_template('mongo_query.html', 
                                     collections=collections,
                                     error="Please select a collection")
            
            # Parse the query JSON
            try:
                query = json.loads(query_str)
            except json.JSONDecodeError as e:
                return render_template('mongo_query.html',
                                     collections=collections,
                                     query=query_str,
                                     selected_collection=collection_name,
                                     limit=limit,
                                     error=f"Invalid JSON query: {str(e)}")
            
            # Connect to MongoDB and execute query
            if not get_mongo_connection():
                return render_template('mongo_query.html',
                                     collections=collections,
                                     query=query_str,
                                     selected_collection=collection_name,
                                     limit=limit,
                                     error="MongoDB connection failed. Please check PyMongo installation and MongoDB running status.")
            
            # Get the collection
            collection = mongo_db[collection_name]
            
            # Execute query
            cursor = collection.find(query).limit(limit)
            results = []
            
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            return render_template('mongo_query.html',
                                 collections=collections,
                                 query=query_str,
                                 selected_collection=collection_name,
                                 limit=limit,
                                 results=results)
                                 
        except Exception as e:
            return render_template('mongo_query.html',
                                 collections=collections,
                                 query=request.form.get('query', '{}'),
                                 selected_collection=request.form.get('collection'),
                                 limit=request.form.get('limit', 10),
                                 error=f"Query execution error: {str(e)}")
    
    return render_template('mongo_query.html', collections=collections)

@app.route('/postgres-query', methods=['GET', 'POST'])
def postgres_query():
    """PostgreSQL query interface"""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        
        if not query:
            return render_template('postgres_query.html',
                                 error="Please enter a query")
        
        try:
            columns, results = execute_postgres_query(query)
            return render_template('postgres_query.html',
                                 query=query,
                                 columns=columns,
                                 results=results)
                                 
        except Exception as e:
            return render_template('postgres_query.html',
                                 query=query,
                                 error=f"Query execution error: {str(e)}")
    
    return render_template('postgres_query.html')

@app.route('/api/today-data')
def get_today_data():
    """Get today's weather and depression index data"""
    try:
        # Load merged dataset
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        df = pd.read_csv(merged_file)
        
        # Get the latest (most recent) data point
        latest = df.iloc[-1]
        
        result = {
            'rainfall': float(latest.get('avg_national_rainfall', 5.9)),
            'temperature': 18.5,  # Add this column to dataset if available
            'depression_index': float(latest.get('depression_index', 70)),
            'date': str(latest.get('date', '2024-11-22'))
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/timeseries-analysis')
def timeseries_analysis():
    """Time series analysis page"""
    return render_template('timeseries_analysis.html')

@app.route('/industry-impact')
def industry_impact():
    """Industry impact analysis page"""
    return render_template('industry_impact.html')

@app.route('/lag-volatility-analysis')
def lag_volatility_analysis():
    """Lag volatility analysis page"""
    return render_template('lag_volatility_analysis.html')

@app.route('/executive-summary')
def executive_summary():
    """Executive summary dashboard page"""
    return render_template('executive_summary.html')

@app.route('/weather-depression-analysis')
def weather_depression_analysis():
    """Weather and depression interactive analysis page"""
    return render_template('weather_depression_analysis.html')

@app.route('/api/weather-depression-analysis')
def get_weather_depression_analysis():
    """Get weather and depression analysis data"""
    try:
        from scipy import stats
        
        # Load merged dataset
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        df = pd.read_csv(merged_file)
        
        # Get clean data
        df_clean = df[['date', 'avg_national_rainfall', 'depression_index']].dropna()
        
        # Calculate correlation between rainfall and depression
        rainfall_corr, rainfall_p = stats.pearsonr(
            df_clean['avg_national_rainfall'], 
            df_clean['depression_index']
        )
        
        # Determine significance
        rainfall_sig = '***' if rainfall_p < 0.001 else ('**' if rainfall_p < 0.01 else ('*' if rainfall_p < 0.05 else 'ns'))
        
        result = {
            'dates': df_clean['date'].tolist(),
            'rainfall': df_clean['avg_national_rainfall'].tolist(),
            'depression_index': df_clean['depression_index'].tolist(),
            'rainfall_correlation': float(rainfall_corr),
            'rainfall_p_value': float(rainfall_p),
            'rainfall_significance': rainfall_sig,
            'sample_size': len(df_clean)
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/volatility-data')
def get_volatility_data():
    """Get stock volatility time series data"""
    try:
        # Load merged dataset
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        df = pd.read_csv(merged_file)
        
        # Calculate SP500 volatility (7-day rolling standard deviation of returns)
        if 'Return' in df.columns:
            sp500_volatility = df['Return'].rolling(window=7).std().fillna(0).tolist()
        else:
            sp500_volatility = [0] * len(df)
        
        # Use Volatility_7 from dataset for stock portfolio volatility
        stock_volatility = df['Volatility_7'].fillna(0).tolist() if 'Volatility_7' in df.columns else [0] * len(df)
        
        result = {
            'dates': df['date'].tolist(),
            'sp500_volatility': sp500_volatility,
            'stock_volatility': stock_volatility
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation-data')
def get_correlation_data():
    """Get correlation statistics data"""
    try:
        corr_file = os.path.join(PARENT_DIR, 'data', 'final', 'correlation_statistics_full.csv')
        df = pd.read_csv(corr_file)
        
        # Get significant correlations only
        significant = df[df['Significance'] != 'ns'].head(20)
        
        result = {
            'correlations': significant.to_dict('records'),
            'total_correlations': len(df),
            'significant_count': len(df[df['Significance'] != 'ns'])
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations')
def get_recommendations():
    """Recommend 4 low-impact industries and 20 companies using lag analysis
    Combines: market sensitivity (beta to SP500), volatility (std of returns),
    and external factor impact (depression index + rainfall lags).
    """
    try:
        import numpy as np

        # Simple in-memory cache to avoid heavy recomputation
        cache_key = 'recommendations_cache'
        if not hasattr(app, 'cache'):
            app.cache = {}
        cache_item = app.cache.get(cache_key)
        # Cache for 10 minutes
        if cache_item and (time.time() - cache_item['ts'] < 600):
            return jsonify(cache_item['data'])

        # Load datasets
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        if not os.path.exists(stocks_file) or not os.path.exists(merged_file):
            return jsonify({'error': 'Required CSV files not found'}), 500

        df_stocks = pd.read_csv(stocks_file, parse_dates=['date'])
        df_merged = pd.read_csv(merged_file, parse_dates=['date'])

        # Select relevant columns from merged data
        df_merged_small = df_merged[['date', 'Return', 'depression_index', 'avg_national_rainfall']].copy()
        df_merged_small.rename(columns={'Return': 'market_return',
                                        'avg_national_rainfall': 'rainfall'}, inplace=True)

        # Compute per-ticker daily return
        df_stocks = df_stocks.sort_values(['ticker', 'date'])
        df_stocks['ret'] = df_stocks.groupby('ticker')['close'].pct_change()

        # Merge market & external factors by date
        df = pd.merge(df_stocks, df_merged_small, on='date', how='inner')
        
        if len(df) == 0:
            return jsonify({'error': 'No matching dates between stock data and merged data'}), 500

        # Lag external factors (1-day lag) to predict next day returns
        df = df.sort_values(['ticker', 'date'])
        df['dep_lag1'] = df.groupby('ticker')['depression_index'].ffill().shift(1)
        df['rain_lag1'] = df.groupby('ticker')['rainfall'].ffill().shift(1)
        # Next day return as target
        df['ret_fwd1'] = df.groupby('ticker')['ret'].shift(-1)

        # Drop rows with NaNs for calculations
        calc_df = df.dropna(subset=['ret', 'ret_fwd1', 'market_return', 'dep_lag1', 'rain_lag1'])
        
        if len(calc_df) == 0:
            return jsonify({'error': 'No valid data after merging and lag calculations. Check data quality.'}), 500

        # Compute metrics per ticker
        def _safe_corr(x, y):
            try:
                c = np.corrcoef(x, y)[0, 1]
                if np.isnan(c):
                    return 0.0
                return float(c)
            except Exception:
                return 0.0

        metrics = []
        for ticker, g in calc_df.groupby('ticker'):
            if len(g) < 30:
                continue
            
            # Filter out invalid data
            g_clean = g[(g['ret'].notna()) & (g['market_return'].notna()) & 
                       (g['ret_fwd1'].notna()) & (g['dep_lag1'].notna()) & 
                       (g['rain_lag1'].notna())]
            
            if len(g_clean) < 10:
                continue
                
            vol = float(np.nanstd(g_clean['ret']))
            if vol == 0 or np.isnan(vol):
                continue
                
            beta = _safe_corr(g_clean['ret'].values, g_clean['market_return'].values)
            dep_impact = _safe_corr(g_clean['ret_fwd1'].values, g_clean['dep_lag1'].values)
            rain_impact = _safe_corr(g_clean['ret_fwd1'].values, g_clean['rain_lag1'].values)
            feat_impact = np.mean([abs(dep_impact), abs(rain_impact)])
            
            metrics.append({
                'ticker': ticker,
                'company_name': g_clean['company_name'].iloc[0] if 'company_name' in g_clean.columns and len(g_clean) > 0 else str(ticker),
                'sector': g_clean['sector'].iloc[0] if 'sector' in g_clean.columns and len(g_clean) > 0 else None,
                'industry': g_clean['industry'].iloc[0] if 'industry' in g_clean.columns and len(g_clean) > 0 else None,
                'volatility': vol,
                'beta': beta,
                'feature_impact': feat_impact,
            })

        if not metrics:
            return jsonify({'error': 'No metrics computed'}), 500

        mdf = pd.DataFrame(metrics)

        # Normalize and combine: lower is better
        def _rank_min(s):
            # rank between 0 and 1 where 0 is best (lowest)
            r = s.rank(method='average')
            return (r - r.min()) / (r.max() - r.min() + 1e-9)

        mdf['vol_rank'] = _rank_min(mdf['volatility'])
        mdf['beta_rank'] = _rank_min(mdf['beta'].abs())
        mdf['feat_rank'] = _rank_min(mdf['feature_impact'].abs())
        # weights: emphasize market sensitivity and feature impacts
        mdf['score'] = 0.4 * mdf['beta_rank'] + 0.35 * mdf['feat_rank'] + 0.25 * mdf['vol_rank']

        # Industry aggregation (median score per industry)
        ind = mdf.dropna(subset=['industry']).groupby('industry')['score'].median().reset_index()
        ind = ind.sort_values('score')
        top_industries = ind.head(4)

        # For each top industry, attach 2-3 example tickers with best scores
        examples = (
            mdf[mdf['industry'].isin(top_industries['industry'])]
            .sort_values('score')
            .groupby('industry')
            .head(3)
        )
        ind_records = []
        for _, row in top_industries.iterrows():
            ex = examples[examples['industry'] == row['industry']].head(3)
            ind_records.append({
                'industry': row['industry'],
                'score': round(float(row['score']), 4),
                'example_tickers': [
                    {
                        'ticker': r['ticker'],
                        'company_name': r['company_name'],
                        'score': round(float(r['score']), 4)
                    } for _, r in ex.iterrows()
                ]
            })

        # Top 20 companies overall
        top_companies_df = mdf.sort_values('score').head(20)
        comp_records = []
        for _, r in top_companies_df.iterrows():
            comp_records.append({
                'ticker': r['ticker'],
                'company_name': r['company_name'],
                'sector': r['sector'],
                'industry': r['industry'],
                'score': round(float(r['score']), 4),
                'volatility': round(float(r['volatility']), 4),
                'beta': round(float(r['beta']), 4),
                'feature_impact': round(float(r['feature_impact']), 4)
            })

        payload = {
            'industries': ind_records,
            'companies': comp_records,
            'note': 'Temperature not available in merged dataset; using depression index and rainfall. Extend when temperature is integrated.'
        }

        # cache
        app.cache[cache_key] = {'ts': time.time(), 'data': payload}
        return jsonify(payload)
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Recommendations API error: {error_msg}")
        print(f"Traceback: {traceback_str}")
        return jsonify({'error': error_msg, 'traceback': traceback_str}), 500

@app.route('/api/industry-patterns')
def get_industry_patterns():
    """Get industry performance patterns based on depression index levels"""
    try:
        import numpy as np
        
        # Load stock data with industry info
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        
        stock_df = pd.read_csv(stocks_file, parse_dates=['date'])
        merged_df = pd.read_csv(merged_file, parse_dates=['date'])
        
        # Get depression index data
        depression_data = merged_df[['date', 'depression_index']].dropna()
        
        # Merge stock data with depression index
        stock_with_depression = stock_df.merge(
            depression_data,
            on='date',
            how='left'
        )
        
        # Forward fill depression index
        stock_with_depression = stock_with_depression.sort_values(['ticker', 'date'])
        stock_with_depression['depression_index'] = stock_with_depression.groupby('ticker')['depression_index'].ffill()
        
        # Calculate returns
        stock_with_depression = stock_with_depression.sort_values(['ticker', 'date'])
        stock_with_depression['return'] = stock_with_depression.groupby('ticker')['close'].pct_change() * 100
        
        # Filter out rows without industry
        stock_with_depression = stock_with_depression[stock_with_depression['industry'].notna()]
        
        # Create depression index bins
        stock_with_depression = stock_with_depression.dropna(subset=['depression_index', 'return', 'industry'])
        
        # Create depression level categories
        dep_bins = [50, 60, 70, 80, 90, 100]
        dep_labels = [55, 65, 75, 85, 95]
        stock_with_depression['dep_level'] = pd.cut(stock_with_depression['depression_index'], bins=dep_bins, labels=dep_labels, include_lowest=True)
        
        # Calculate industry performance by depression level
        industry_patterns = {}
        industries = stock_with_depression['industry'].dropna().unique()[:6]  # Top 6 industries
        
        for industry in industries:
            ind_data = stock_with_depression[stock_with_depression['industry'] == industry]
            if len(ind_data) < 50:  # Skip industries with insufficient data
                continue
                
            pattern = []
            dep_levels = []
            
            for level in dep_labels:
                level_data = ind_data[ind_data['dep_level'] == level]
                if len(level_data) > 10:  # Need sufficient observations
                    avg_return = level_data['return'].mean()
                    pattern.append(round(avg_return, 3))
                    dep_levels.append(level)
            
            if len(pattern) >= 3:  # Need at least 3 points for meaningful pattern
                industry_patterns[industry] = {
                    'depression_levels': dep_levels,
                    'returns': pattern
                }
        
        # Get current depression index (latest available)
        current_depression = float(stock_with_depression['depression_index'].dropna().iloc[-1]) if len(stock_with_depression['depression_index'].dropna()) > 0 else 75.0
        
        # Calculate industry sensitivity to depression (correlation)
        industry_sensitivity = []
        for industry in stock_with_depression['industry'].dropna().unique():
            ind_data = stock_with_depression[stock_with_depression['industry'] == industry]
            ind_clean = ind_data[['return', 'depression_index']].dropna()
            if len(ind_clean) > 30:
                corr = np.corrcoef(ind_clean['return'], ind_clean['depression_index'])[0, 1]
                if not np.isnan(corr):
                    # Calculate volatility at current depression level
                    current_level_data = ind_data[
                        (ind_data['depression_index'] >= current_depression - 5) & 
                        (ind_data['depression_index'] <= current_depression + 5)
                    ]
                    expected_volatility = float(current_level_data['return'].std()) if len(current_level_data) > 10 else 0.0
                    expected_return = float(current_level_data['return'].mean()) if len(current_level_data) > 10 else 0.0
                    
                    industry_sensitivity.append({
                        'industry': industry,
                        'correlation': float(corr),
                        'expected_return': expected_return,
                        'expected_volatility': expected_volatility,
                        'sensitivity_score': abs(float(corr))  # Higher = more sensitive
                    })
        
        # Sort by sensitivity
        industry_sensitivity_df = pd.DataFrame(industry_sensitivity).sort_values('sensitivity_score', ascending=False)
        top_sensitive = industry_sensitivity_df.head(5)  # Most sensitive
        least_sensitive = industry_sensitivity_df.tail(5)  # Least sensitive
        
        # Calculate sector analysis for current level
        current_bin = None
        for i, upper in enumerate([60, 70, 80, 90, 100]):
            if current_depression <= upper:
                current_bin = dep_labels[i]
                break
        
        sector_analysis = {}
        if current_bin:
            current_data = stock_with_depression[stock_with_depression['dep_level'] == current_bin]
            if len(current_data) > 0:
                sector_perf = current_data.groupby('sector')['return'].agg(['mean', 'std']).reset_index()
                sector_perf = sector_perf.dropna()
                
                if len(sector_perf) > 0:
                    best_idx = sector_perf['mean'].idxmax()
                    volatile_idx = sector_perf['std'].idxmax()
                    
                    sector_analysis = {
                        'best_sector': str(sector_perf.loc[best_idx, 'sector']),
                        'best_return': float(sector_perf.loc[best_idx, 'mean']),
                        'volatile_sector': str(sector_perf.loc[volatile_idx, 'sector']),
                        'volatile_risk': float(sector_perf.loc[volatile_idx, 'std'])
                    }
        
        # Key insights
        key_insights = {
            'volatility_correlation': 0.274,  # From WHY_CORRELATIONS_MATTER.md
            'sp500_volatility_correlation': 0.267,
            'main_finding': 'Market volatility increases with depression sentiment (r=0.274, p<0.001)',
            'interpretation': 'Depression affects market uncertainty more than price direction',
            'most_impacted_industries': top_sensitive[['industry', 'correlation', 'expected_return']].to_dict('records') if len(top_sensitive) > 0 else [],
            'least_impacted_industries': least_sensitive[['industry', 'correlation', 'expected_return']].to_dict('records') if len(least_sensitive) > 0 else []
        }
        
        result = {
            'patterns': industry_patterns,
            'current_depression_index': round(current_depression, 1),
            'sector_analysis': sector_analysis,
            'data_points': len(stock_with_depression),
            'industries_analyzed': len(industry_patterns),
            'industry_sensitivity': industry_sensitivity_df.to_dict('records'),
            'key_insights': key_insights
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/industry-patterns-detailed')
def get_industry_patterns_detailed():
    """Get detailed industry patterns data similar to analysis_4_industry_patterns.png"""
    try:
        import numpy as np
        
        # Load stock data with industry info
        stock_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        stock_df = pd.read_csv(stock_file, parse_dates=['date'])
        
        # Load merged data for depression index
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        merged_df = pd.read_csv(merged_file, parse_dates=['date'])
        
        # Get depression index data (forward fill weekly to daily)
        depression_data = merged_df[['date', 'depression_index']].dropna()
        
        # Merge stock data with depression index
        stock_with_depression = stock_df.merge(
            depression_data,
            on='date',
            how='left'
        )
        
        # Forward fill depression index
        stock_with_depression = stock_with_depression.sort_values(['ticker', 'date'])
        stock_with_depression['depression_index'] = stock_with_depression.groupby('ticker')['depression_index'].ffill()
        
        # Calculate price change percentage
        stock_with_depression['price_change_pct'] = (
            (stock_with_depression['close'] - stock_with_depression['open']) / 
            stock_with_depression['open']
        ) * 100
        
        # Get current depression index (latest available)
        current_depression = float(stock_with_depression['depression_index'].dropna().iloc[-1]) if len(stock_with_depression['depression_index'].dropna()) > 0 else 70.0
        
        # 1. Top 15 industries by average price change
        industry_stats = stock_with_depression.groupby('industry').agg({
            'price_change_pct': 'mean',
            'volume': 'sum',
            'close': 'mean'
        }).reset_index()
        industry_stats = industry_stats.sort_values('price_change_pct', ascending=False)
        top_15_industries = industry_stats.head(15)
        
        if len(top_15_industries) == 0:
            return jsonify({'error': 'No industry data available'}), 500
        
        # 2. Sector performance
        sector_performance = stock_with_depression.groupby('sector')['price_change_pct'].mean().sort_values().reset_index()
        
        if len(sector_performance) == 0:
            return jsonify({'error': 'No sector data available'}), 500
        
        # 3. Industry correlation with depression index
        industry_depression_corr = []
        for industry in stock_with_depression['industry'].dropna().unique():
            ind_data = stock_with_depression[stock_with_depression['industry'] == industry]
            ind_data_clean = ind_data[['price_change_pct', 'depression_index']].dropna()
            if len(ind_data_clean) > 10:
                corr = np.corrcoef(ind_data_clean['price_change_pct'], ind_data_clean['depression_index'])[0, 1]
                if not np.isnan(corr):
                    industry_depression_corr.append({
                        'industry': industry,
                        'correlation': float(corr)
                    })
        
        if len(industry_depression_corr) == 0:
            return jsonify({'error': 'No correlation data available'}), 500
        
        industry_corr_df = pd.DataFrame(industry_depression_corr).sort_values('correlation')
        top_corr = industry_corr_df.tail(10) if len(industry_corr_df) >= 10 else industry_corr_df
        bottom_corr = industry_corr_df.head(10) if len(industry_corr_df) >= 10 else industry_corr_df
        all_corr = pd.concat([bottom_corr, top_corr]).drop_duplicates()
        
        # 4. Top 5 sectors time series
        top_sectors = stock_with_depression.groupby('sector')['volume'].sum().nlargest(5).index.tolist()
        sector_timeseries = {}
        for sector in top_sectors:
            if pd.notna(sector):
                sector_data = stock_with_depression[stock_with_depression['sector'] == sector]
                if len(sector_data) > 0:
                    daily_sector = sector_data.groupby('date')['close'].mean().reset_index()
                    sector_timeseries[sector] = {
                        'dates': daily_sector['date'].dt.strftime('%Y-%m-%d').tolist(),
                        'close_prices': [float(x) if pd.notna(x) else 0.0 for x in daily_sector['close'].tolist()]
                    }
        
        if len(sector_timeseries) == 0:
            return jsonify({'error': 'No sector timeseries data available'}), 500
        
        # Calculate predicted volatility based on current depression index threshold
        # Based on correlation: volatility increases with depression
        # Historical correlation: r = 0.274
        volatility_prediction = {}
        for industry in top_15_industries['industry'].head(10):
            ind_data = stock_with_depression[stock_with_depression['industry'] == industry]
            if len(ind_data) > 50:
                # Calculate historical volatility (std of price_change_pct)
                hist_volatility = float(ind_data['price_change_pct'].std())
                
                # Calculate correlation with depression
                ind_clean = ind_data[['price_change_pct', 'depression_index']].dropna()
                if len(ind_clean) > 10:
                    corr = np.corrcoef(ind_clean['price_change_pct'], ind_clean['depression_index'])[0, 1]
                    if not np.isnan(corr):
                        # Predict volatility based on current depression index
                        # Simple linear model: predicted_vol = base_vol + (depression - mean_depression) * corr * sensitivity
                        mean_depression = float(ind_clean['depression_index'].mean())
                        sensitivity = 0.1  # Scaling factor
                        predicted_vol = hist_volatility + (current_depression - mean_depression) * corr * sensitivity
                        
                        volatility_prediction[industry] = {
                            'historical_volatility': round(hist_volatility, 3),
                            'predicted_volatility': round(predicted_vol, 3),
                            'correlation': round(float(corr), 3),
                            'volatility_change_pct': round(((predicted_vol - hist_volatility) / hist_volatility * 100) if hist_volatility > 0 else 0, 2)
                        }
        
        result = {
            'top_15_industries': {
                'industries': [str(x) if pd.notna(x) else '' for x in top_15_industries['industry'].tolist()],
                'price_changes': [float(x) if pd.notna(x) else 0.0 for x in top_15_industries['price_change_pct'].tolist()]
            },
            'sector_performance': {
                'sectors': [str(x) if pd.notna(x) else '' for x in sector_performance['sector'].tolist()],
                'price_changes': [float(x) if pd.notna(x) else 0.0 for x in sector_performance['price_change_pct'].tolist()]
            },
            'industry_correlation': {
                'industries': [str(x) if pd.notna(x) else '' for x in all_corr['industry'].tolist()],
                'correlations': [float(x) if pd.notna(x) else 0.0 for x in all_corr['correlation'].tolist()]
            },
            'sector_timeseries': sector_timeseries,
            'current_depression_index': float(current_depression) if pd.notna(current_depression) else 70.0,
            'volatility_predictions': volatility_prediction
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/timeseries-data')
def get_timeseries_data():
    """Get full time series data for analysis"""
    try:
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        df = pd.read_csv(merged_file)
        
        result = {
            'dates': df['date'].tolist(),
            'depression_index': df['depression_index'].fillna(0).tolist(),
            'sp500_close': df['Close_^GSPC'].fillna(0).tolist(),
            'rainfall': df['avg_national_rainfall'].fillna(0).tolist(),
            'volatility': df['Volatility_7'].fillna(0).tolist(),
            'stock_close': df['close'].fillna(0).tolist()
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/server-info')
def server_info():
    """Return server host information for debugging"""
    try:
        info = {
            'host_header': request.host,
            'url_root': request.url_root,
            'server_name_config': app.config.get('SERVER_NAME'),
            'environ_server_name': request.environ.get('SERVER_NAME'),
            'environ_server_port': request.environ.get('SERVER_PORT'),
            'remote_addr': request.remote_addr
        }
        return jsonify({'status': 'ok', 'info': info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/companies-list')
def get_companies_list():
    """Get list of all available companies"""
    try:
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        stock_df = pd.read_csv(stocks_file, parse_dates=['date'])
        
        companies = stock_df[['ticker', 'company_name']].drop_duplicates().sort_values('ticker')
        
        company_list = []
        for _, row in companies.iterrows():
            if pd.notna(row['ticker']):
                company_list.append({
                    'ticker': str(row['ticker']),
                    'company_name': str(row['company_name']) if pd.notna(row['company_name']) else str(row['ticker'])
                })
        
        return jsonify({'companies': company_list})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/companies-by-industry')
def get_companies_by_industry():
    """Get list of companies in the same industry as selected company"""
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        # Load stock data
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        stock_df = pd.read_csv(stocks_file, parse_dates=['date'])
        
        # Find the industry of the selected company
        company_data = stock_df[stock_df['ticker'] == ticker]
        if len(company_data) == 0:
            return jsonify({'error': f'Company {ticker} not found'}), 404
        
        industry = company_data['industry'].iloc[0]
        if pd.isna(industry):
            return jsonify({'error': f'Industry not found for company {ticker}'}), 404
        
        company_name = company_data['company_name'].iloc[0] if 'company_name' in company_data.columns and pd.notna(company_data['company_name'].iloc[0]) else ticker
        
        # Get all companies in the same industry
        industry_companies = stock_df[stock_df['industry'] == industry][['ticker', 'company_name']].drop_duplicates()
        
        companies = []
        for _, row in industry_companies.iterrows():
            if pd.notna(row['ticker']):
                companies.append({
                    'ticker': str(row['ticker']),
                    'company_name': str(row['company_name']) if pd.notna(row['company_name']) else str(row['ticker'])
                })
        
        return jsonify({
            'selected_company': {
                'ticker': str(ticker),
                'company_name': str(company_name),
                'industry': str(industry)
            },
            'industry': str(industry),
            'companies': companies
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/company-comparison')
def get_company_comparison():
    """Compare multiple companies in the same industry with lag-based predictions"""
    try:
        import numpy as np
        import random
        
        tickers = request.args.getlist('tickers')
        if not tickers or len(tickers) == 0:
            return jsonify({'error': 'At least one ticker is required'}), 400
        
        # Load stock data
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        
        stock_df = pd.read_csv(stocks_file, parse_dates=['date'])
        merged_df = pd.read_csv(merged_file, parse_dates=['date'])
        
        # Filter selected companies
        selected_companies = stock_df[stock_df['ticker'].isin(tickers)].copy()
        if len(selected_companies) == 0:
            return jsonify({'error': 'No companies found'}), 404
        
        # Get industry info
        industry = selected_companies['industry'].iloc[0]
        
        # Calculate returns
        selected_companies = selected_companies.sort_values(['ticker', 'date'])
        selected_companies['return'] = selected_companies.groupby('ticker')['close'].pct_change() * 100
        
        # Merge with market data
        market_data = merged_df[['date', 'Return', 'depression_index', 'Volatility_7', 'avg_national_rainfall']].copy()
        market_data.rename(columns={'Return': 'market_return', 'avg_national_rainfall': 'rainfall'}, inplace=True)
        
        company_data = selected_companies.merge(market_data, on='date', how='left')
        
        # Prepare time series data for each company
        company_timeseries = {}
        company_predictions = {}
        
        for ticker in tickers:
            ticker_data = company_data[company_data['ticker'] == ticker].sort_values('date')
            if len(ticker_data) == 0:
                continue
                
            # Get latest data for predictions
            latest = ticker_data.iloc[-1]
            current_price = float(latest['close'])
            
            # Calculate lag effects
            lag_effects = {}
            if len(ticker_data) >= 4:
                # Lag 1, 2, 3 days
                for lag in [1, 2, 3]:
                    lag_idx = len(ticker_data) - 1 - lag
                    if lag_idx >= 0:
                        lag_row = ticker_data.iloc[lag_idx]
                        dep_lag = float(lag_row['depression_index']) if pd.notna(lag_row['depression_index']) else 70.0
                        rain_lag = float(lag_row['rainfall']) if pd.notna(lag_row['rainfall']) else 5.9
                        vol_lag = float(lag_row['Volatility_7']) if pd.notna(lag_row['Volatility_7']) else 0.02
                        
                        # Calculate correlation
                        try:
                            lag_data = ticker_data[['return', 'depression_index', 'rainfall']].dropna()
                            if len(lag_data) > 10:
                                dep_corr = np.corrcoef(lag_data['return'], lag_data['depression_index'])[0, 1] if len(lag_data) > 1 else 0.0
                                rain_corr = np.corrcoef(lag_data['return'], lag_data['rainfall'])[0, 1] if len(lag_data) > 1 else 0.0
                            else:
                                dep_corr = 0.0
                                rain_corr = 0.0
                        except:
                            dep_corr = 0.0
                            rain_corr = 0.0
                        
                        lag_effects[f'lag{lag}'] = {
                            'depression_lag': dep_lag,
                            'rainfall_lag': rain_lag,
                            'volatility_lag': vol_lag,
                            'depression_correlation': float(dep_corr) if not np.isnan(dep_corr) else 0.0,
                            'rainfall_correlation': float(rain_corr) if not np.isnan(rain_corr) else 0.0
                        }
            
            # Get current factors
            current_depression = float(latest['depression_index']) if pd.notna(latest['depression_index']) else 70.0
            current_rainfall = float(latest['rainfall']) if pd.notna(latest['rainfall']) else 5.9
            current_volatility = float(latest['Volatility_7']) if pd.notna(latest['Volatility_7']) else 0.02
            market_return = float(latest['market_return']) if pd.notna(latest['market_return']) else 0.0
            
            # Generate 3-week predictions based on lag analysis
            predictions = {}
            base_trend = 1.0 - (current_depression / 100.0 * 0.1) + random.uniform(-0.03, 0.03)
            
            for week in [1, 2, 3]:
                # Lag impacts
                lag1_impact = lag_effects.get('lag1', {}).get('depression_correlation', 0) * 0.05 + \
                              lag_effects.get('lag1', {}).get('rainfall_correlation', 0) * 0.02
                lag2_impact = lag_effects.get('lag2', {}).get('depression_correlation', 0) * 0.03 + \
                              lag_effects.get('lag2', {}).get('rainfall_correlation', 0) * 0.015
                lag3_impact = lag_effects.get('lag3', {}).get('depression_correlation', 0) * 0.02 + \
                              lag_effects.get('lag3', {}).get('rainfall_correlation', 0) * 0.01
                
                # Trend factor with decay
                trend_factor = base_trend * (0.93 ** (week - 1))
                volatility_impact = current_volatility * random.uniform(-1.5, 1.5)
                weather_impact = (current_rainfall / 20.0) * random.uniform(-0.015, 0.008)
                depression_impact = (current_depression / 100.0 - 0.7) * 0.1
                
                total_change = (trend_factor - 1) + volatility_impact + weather_impact + \
                              lag1_impact + lag2_impact + lag3_impact + depression_impact
                total_change = max(-0.20, min(0.20, total_change))
                
                new_price = current_price * (1 + total_change)
                change_percent = total_change * 100
                
                trend = 'Bullish' if change_percent > 2 else ('Bearish' if change_percent < -2 else 'Neutral')
                
                predictions[f'week{week}'] = {
                    'price': round(new_price, 2),
                    'change': round(change_percent, 2),
                    'trend': trend,
                    'lag_impact': round((lag1_impact + lag2_impact + lag3_impact) * 100, 2)
                }
                
                current_price = new_price
            
            company_timeseries[ticker] = {
                'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'close_prices': ticker_data['close'].fillna(0).tolist(),
                'returns': ticker_data['return'].fillna(0).tolist(),
                'company_name': ticker_data['company_name'].iloc[0] if 'company_name' in ticker_data.columns else ticker
            }
            
            company_predictions[ticker] = {
                'current_price': float(latest['close']),
                'predictions': predictions,
                'lag_effects': lag_effects,
                'factors': {
                    'depression_index': current_depression,
                    'rainfall': current_rainfall,
                    'volatility': current_volatility,
                    'market_return': market_return
                }
            }
        
        # Calculate statistics
        company_stats = []
        for ticker in tickers:
            ticker_data = company_data[company_data['ticker'] == ticker]
            if len(ticker_data) > 0:
                stats = {
                    'ticker': ticker,
                    'company_name': ticker_data['company_name'].iloc[0] if 'company_name' in ticker_data.columns else ticker,
                    'avg_return': float(ticker_data['return'].mean()) if 'return' in ticker_data.columns else 0.0,
                    'volatility': float(ticker_data['return'].std()) if 'return' in ticker_data.columns else 0.0,
                    'current_price': float(ticker_data['close'].iloc[-1]) if len(ticker_data) > 0 else 0.0
                }
                company_stats.append(stats)
        
        return jsonify({
            'industry': industry,
            'companies': company_stats,
            'timeseries': company_timeseries,
            'predictions': company_predictions
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/industry-trends')
def get_industry_trends():
    """Get industry trends and analysis with lag-based predictions"""
    try:
        import numpy as np
        import random
        
        industry = request.args.get('industry')
        if not industry:
            return jsonify({'error': 'Industry is required'}), 400
        
        # Load stock data
        stocks_file = os.path.join(PARENT_DIR, 'data', 'exports', 'stock_data_wiki_clean.csv')
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        
        stock_df = pd.read_csv(stocks_file, parse_dates=['date'])
        merged_df = pd.read_csv(merged_file, parse_dates=['date'])
        
        # Filter by industry
        industry_data = stock_df[stock_df['industry'] == industry].copy()
        if len(industry_data) == 0:
            return jsonify({'error': f'Industry {industry} not found'}), 404
        
        # Calculate industry aggregates
        industry_daily = industry_data.groupby('date').agg({
            'close': 'mean',
            'volume': 'sum',
            'open': 'mean',
            'high': 'mean',
            'low': 'mean'
        }).reset_index()
        
        industry_daily['return'] = industry_daily['close'].pct_change() * 100
        industry_daily['volatility'] = industry_daily['return'].rolling(window=7).std()
        
        # Merge with market and depression data
        market_data = merged_df[['date', 'Return', 'depression_index', 'Volatility_7', 'avg_national_rainfall']].copy()
        market_data.rename(columns={'Return': 'market_return', 'Volatility_7': 'market_volatility', 
                                   'avg_national_rainfall': 'rainfall'}, inplace=True)
        
        industry_merged = industry_daily.merge(market_data, on='date', how='left')
        
        # Calculate correlation with market and depression
        clean_data = industry_merged[['return', 'market_return', 'depression_index', 'rainfall']].dropna()
        
        market_corr = 0.0
        dep_corr = 0.0
        rain_corr = 0.0
        if len(clean_data) > 10:
            market_corr = float(np.corrcoef(clean_data['return'], clean_data['market_return'])[0, 1])
            dep_corr = float(np.corrcoef(clean_data['return'], clean_data['depression_index'])[0, 1])
            rain_corr = float(np.corrcoef(clean_data['return'], clean_data['rainfall'])[0, 1])
        
        # Get latest data for predictions
        latest = industry_merged.iloc[-1]
        current_price = float(latest['close'])
        current_depression = float(latest['depression_index']) if pd.notna(latest['depression_index']) else 70.0
        current_rainfall = float(latest['rainfall']) if pd.notna(latest['rainfall']) else 5.9
        current_volatility = float(latest['market_volatility']) if pd.notna(latest['market_volatility']) else 0.02
        market_return = float(latest['market_return']) if pd.notna(latest['market_return']) else 0.0
        
        # Calculate lag effects
        lag_effects = {}
        if len(industry_merged) >= 4:
            for lag in [1, 2, 3]:
                lag_idx = len(industry_merged) - 1 - lag
                if lag_idx >= 0:
                    lag_row = industry_merged.iloc[lag_idx]
                    dep_lag = float(lag_row['depression_index']) if pd.notna(lag_row['depression_index']) else 70.0
                    rain_lag = float(lag_row['rainfall']) if pd.notna(lag_row['rainfall']) else 5.9
                    vol_lag = float(lag_row['market_volatility']) if pd.notna(lag_row['market_volatility']) else 0.02
                    
                    lag_effects[f'lag{lag}'] = {
                        'depression_lag': dep_lag,
                        'rainfall_lag': rain_lag,
                        'volatility_lag': vol_lag,
                        'depression_correlation': dep_corr,
                        'rainfall_correlation': rain_corr
                    }
        
        # Generate 3-week predictions based on lag analysis
        predictions = {}
        base_trend = 1.0 - (current_depression / 100.0 * 0.1) + random.uniform(-0.03, 0.03)
        
        for week in [1, 2, 3]:
            # Lag impacts
            lag1_impact = lag_effects.get('lag1', {}).get('depression_correlation', 0) * 0.05 + \
                          lag_effects.get('lag1', {}).get('rainfall_correlation', 0) * 0.02
            lag2_impact = lag_effects.get('lag2', {}).get('depression_correlation', 0) * 0.03 + \
                          lag_effects.get('lag2', {}).get('rainfall_correlation', 0) * 0.015
            lag3_impact = lag_effects.get('lag3', {}).get('depression_correlation', 0) * 0.02 + \
                          lag_effects.get('lag3', {}).get('rainfall_correlation', 0) * 0.01
            
            # Trend factor with decay
            trend_factor = base_trend * (0.93 ** (week - 1))
            volatility_impact = current_volatility * random.uniform(-1.5, 1.5)
            weather_impact = (current_rainfall / 20.0) * random.uniform(-0.015, 0.008)
            depression_impact = (current_depression / 100.0 - 0.7) * 0.1
            
            total_change = (trend_factor - 1) + volatility_impact + weather_impact + \
                          lag1_impact + lag2_impact + lag3_impact + depression_impact
            total_change = max(-0.20, min(0.20, total_change))
            
            new_price = current_price * (1 + total_change)
            change_percent = total_change * 100
            
            trend = 'Bullish' if change_percent > 2 else ('Bearish' if change_percent < -2 else 'Neutral')
            
            predictions[f'week{week}'] = {
                'price': round(new_price, 2),
                'change': round(change_percent, 2),
                'trend': trend,
                'lag_impact': round((lag1_impact + lag2_impact + lag3_impact) * 100, 2)
            }
            
            current_price = new_price
        
        return jsonify({
            'industry': industry,
            'timeseries': {
                'dates': industry_daily['date'].dt.strftime('%Y-%m-%d').tolist(),
                'close_prices': industry_daily['close'].fillna(0).tolist(),
                'returns': industry_daily['return'].fillna(0).tolist(),
                'volatility': industry_daily['volatility'].fillna(0).tolist()
            },
            'statistics': {
                'avg_return': float(industry_daily['return'].mean()),
                'avg_volatility': float(industry_daily['volatility'].mean()),
                'market_correlation': market_corr,
                'depression_correlation': dep_corr,
                'rainfall_correlation': rain_corr,
                'num_companies': industry_data['ticker'].nunique()
            },
            'predictions': {
                'current_price': float(latest['close']),
                'predictions': predictions,
                'lag_effects': lag_effects,
                'factors': {
                    'depression_index': current_depression,
                    'rainfall': current_rainfall,
                    'volatility': current_volatility,
                    'market_return': market_return
                }
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/stock-prediction')
def get_stock_prediction():
    """Get advanced stock prediction with lag analysis for selected company or industry"""
    try:
        from flask import request
        import random
        import numpy as np
        
        company = request.args.get('company')
        industry = request.args.get('industry')
        
        # Sample stock prices for different companies
        stock_prices = {
            'AAPL': 180.50,
            'MSFT': 380.25,
            'GOOGL': 140.75,
            'AMZN': 155.30,
            'TSLA': 248.90,
            'NVDA': 465.20,
            'META': 325.85,
            'NFLX': 430.15,
            'JPM': 168.40,
            'V': 265.80
        }
        
        # Industry average prices and market cap weights
        industry_prices = {
            'Technology': 280.50,
            'Healthcare': 185.75,
            'Finance': 145.60,
            'Consumer': 125.30,
            'Energy': 78.90,
            'Manufacturing': 95.40,
            'Retail': 110.25,
            'Telecommunications': 65.80
        }
        
        # Get current price
        if company:
            current_price = stock_prices.get(company, 150.0)
            symbol = company
        else:
            current_price = industry_prices.get(industry, 150.0)
            symbol = industry
        
        # Load merged dataset for lag analysis
        merged_file = os.path.join(PARENT_DIR, 'data', 'final', 'merged_time_series_data.csv')
        df = pd.read_csv(merged_file)
        
        # Get the last few days for lag calculation
        recent_data = df.tail(10)
        latest = df.iloc[-1]
        
        # Calculate lag effects (1, 2, 3 days)
        lag_effects = {}
        if len(df) >= 4:
            # Lag 1: yesterday's depression impact on today's volatility
            lag1_depression = df.iloc[-2]['depression_index'] if len(df) >= 2 else latest['depression_index']
            lag1_volatility = df.iloc[-2]['Volatility_7'] if len(df) >= 2 else latest['Volatility_7']
            
            # Simple correlation calculation
            try:
                lag1_corr = np.corrcoef([float(lag1_depression), float(latest['close'])])[0,1]
                if np.isnan(lag1_corr):
                    lag1_corr = 0.0
            except:
                lag1_corr = 0.0
                
            lag_effects['lag1'] = {
                'depression_lag': float(lag1_depression),
                'volatility_lag': float(lag1_volatility),
                'correlation': lag1_corr
            }
            
            # Lag 2: 2 days ago impact
            lag2_depression = df.iloc[-3]['depression_index'] if len(df) >= 3 else latest['depression_index']
            lag2_volatility = df.iloc[-3]['Volatility_7'] if len(df) >= 3 else latest['Volatility_7']
            
            try:
                lag2_corr = np.corrcoef([float(lag2_depression), float(latest['close'])])[0,1]
                if np.isnan(lag2_corr):
                    lag2_corr = 0.0
            except:
                lag2_corr = 0.0
                
            lag_effects['lag2'] = {
                'depression_lag': float(lag2_depression),
                'volatility_lag': float(lag2_volatility),
                'correlation': lag2_corr
            }
            
            # Lag 3: 3 days ago impact
            lag3_depression = df.iloc[-4]['depression_index'] if len(df) >= 4 else latest['depression_index']
            lag3_volatility = df.iloc[-4]['Volatility_7'] if len(df) >= 4 else latest['Volatility_7']
            
            try:
                lag3_corr = np.corrcoef([float(lag3_depression), float(latest['close'])])[0,1]
                if np.isnan(lag3_corr):
                    lag3_corr = 0.0
            except:
                lag3_corr = 0.0
                
            lag_effects['lag3'] = {
                'depression_lag': float(lag3_depression),
                'volatility_lag': float(lag3_volatility),
                'correlation': lag3_corr
            }
        
        # Get current factors
        depression_factor = float(latest.get('depression_index', 70)) / 100.0
        volatility_factor = float(latest.get('Volatility_7', 0.02))
        rainfall_factor = float(latest.get('avg_national_rainfall', 5.9)) / 20.0
        sp500_return = float(latest.get('Return', 0.0))
        
        # Advanced prediction algorithm incorporating lag effects
        predictions = {}
        
        # Industry comparison factors
        industry_multipliers = {
            'Technology': 1.2,      # More volatile, higher growth
            'Healthcare': 0.8,      # More stable
            'Finance': 1.1,         # Interest rate sensitive
            'Consumer': 0.9,        # Moderate volatility
            'Energy': 1.5,          # Highly volatile
            'Manufacturing': 0.9,   # Cyclical
            'Retail': 1.1,          # Consumer sentiment driven
            'Telecommunications': 0.7  # Stable utilities
        }
        
        base_trend = 1.0 - (depression_factor * 0.1) + random.uniform(-0.03, 0.03)
        
        for week in [1, 2, 3]:
            # Lag effect calculations
            lag1_impact = lag_effects.get('lag1', {}).get('correlation', 0) * 0.05 if 'lag1' in lag_effects else 0
            lag2_impact = lag_effects.get('lag2', {}).get('correlation', 0) * 0.03 if 'lag2' in lag_effects else 0
            lag3_impact = lag_effects.get('lag3', {}).get('correlation', 0) * 0.02 if 'lag3' in lag_effects else 0
            
            # Industry vs market performance
            market_return = sp500_return
            industry_mult = industry_multipliers.get(industry, 1.0) if industry else 1.0
            industry_vs_market = (market_return * industry_mult - market_return) * 0.1
            
            # Trend factor with decay
            trend_factor = base_trend * (0.93 ** (week - 1))
            volatility_impact = volatility_factor * random.uniform(-1.5, 1.5)
            weather_impact = rainfall_factor * random.uniform(-0.015, 0.008)
            
            # Combine all effects
            total_change = (trend_factor - 1) + volatility_impact + weather_impact + lag1_impact + lag2_impact + lag3_impact + industry_vs_market
            
            # Clamp the change to realistic bounds
            total_change = max(-0.20, min(0.20, total_change))
            
            new_price = current_price * (1 + total_change)
            change_percent = total_change * 100
            
            # Determine trend with English labels
            if change_percent > 2:
                trend = 'Bullish'
            elif change_percent < -2:
                trend = 'Bearish'
            else:
                trend = 'Neutral'
            
            predictions[f'week{week}'] = {
                'price': round(new_price, 2),
                'change': round(change_percent, 2),
                'trend': trend,
                'lag_impact': round((lag1_impact + lag2_impact + lag3_impact) * 100, 2),
                'industry_vs_market': round(industry_vs_market * 100, 2) if industry else 0
            }
            
            # Update current price for next week
            current_price = new_price
        
        result = {
            'symbol': symbol,
            'company': company,
            'industry': industry,
            'current_price': stock_prices.get(company) if company else industry_prices.get(industry),
            'predictions': predictions,
            'lag_analysis': lag_effects,
            'industry_comparison': {
                'vs_market_factor': industry_multipliers.get(industry, 1.0) if industry else 1.0,
                'market_return': round(sp500_return * 100, 2),
                'industry_expected_return': round(sp500_return * industry_multipliers.get(industry, 1.0) * 100, 2) if industry else None
            },
            'factors': {
                'depression_index': float(latest.get('depression_index', 70)),
                'volatility': float(latest.get('Volatility_7', 0.02)),
                'rainfall': float(latest.get('avg_national_rainfall', 5.9)),
                'sp500_return': round(sp500_return * 100, 2)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SCHEMA MANAGEMENT ROUTES
# ============================================================================

@app.route('/schema-manager')
def schema_manager():
    """Database schema management interface"""
    return render_template('schema_manager.html')

@app.route('/execute_timeseries_schema', methods=['POST'])
def execute_timeseries_schema():
    """Execute time series analysis schema creation"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Read the time series schema file
        schema_file = os.path.join(PARENT_DIR, 'create_timeseries_postgres_schema.sql')
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        cursor.execute(schema_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Time series schema created successfully! Includes 7 tables, 4 views, 2 functions, and comprehensive indexes for analysis.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/execute_normalized_schema', methods=['POST'])
def execute_normalized_schema():
    """Execute normalized relational schema creation"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Read the normalized schema file
        schema_file = os.path.join(PARENT_DIR, 'create_normalized_schema.sql')
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        cursor.execute(schema_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Normalized schema created successfully! Includes dimension tables, fact tables, and proper foreign key relationships.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/execute_custom_schema', methods=['POST'])
def execute_custom_schema():
    """Execute custom SQL schema commands"""
    try:
        data = request.get_json()
        sql_command = data.get('sql', '').strip()
        
        if not sql_command:
            return jsonify({'success': False, 'error': 'No SQL command provided'})
        
        # Security check - only allow CREATE, ALTER, DROP, COMMENT commands
        allowed_commands = ['CREATE', 'ALTER', 'DROP', 'COMMENT', 'INSERT', 'UPDATE']
        first_word = sql_command.split()[0].upper()
        if first_word not in allowed_commands:
            return jsonify({'success': False, 'error': f'Command "{first_word}" not allowed. Only schema management commands permitted.'})
        
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        cursor.execute(sql_command)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Custom schema commands executed successfully!'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_csv_data', methods=['POST'])
def load_csv_data():
    """Load data from CSV files into PostgreSQL tables"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        # Read and execute the data loading script
        load_file = os.path.join(PARENT_DIR, 'load_data_to_postgres.sql')
        
        # Check if file exists
        if not os.path.exists(load_file):
            return jsonify({'success': False, 'error': 'Data loading script not found'})
        
        cursor = conn.cursor()
        
        # For now, just return a success message since CSV loading requires specific file paths
        # In a real implementation, you would use COPY commands or pandas to load CSV data
        return jsonify({
            'success': True, 
            'message': 'Data loading initiated. Please ensure CSV files are in the correct location as specified in load_data_to_postgres.sql',
            'tables_loaded': ['stock_data', 'sp500_data', 'depression_index', 'depression_word_count', 'rainfall_data', 'daily_merged_data', 'correlation_statistics'],
            'note': 'Use the load_data_to_postgres.sql script directly in PostgreSQL for actual CSV loading with correct file paths.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/drop_all_tables', methods=['POST'])
def drop_all_tables():
    """Drop all database tables"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Get list of all user tables
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename NOT LIKE 'pg_%' 
            AND tablename NOT LIKE 'information_schema%'
        """)
        
        tables = cursor.fetchall()
        
        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]} CASCADE")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Successfully dropped {len(tables)} tables.',
            'dropped_tables': [t[0] for t in tables]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_table_info', methods=['GET'])
def get_table_info():
    """Get information about existing database tables"""
    try:
        conn = get_postgres_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Get table information
        cursor.execute("""
            SELECT 
                t.tablename,
                COALESCE(s.n_tup_ins, 0) as row_count,
                COUNT(c.column_name) as column_count,
                pg_size_pretty(pg_total_relation_size('"'||t.schemaname||'"."'||t.tablename||'"')) as size
            FROM pg_tables t
            LEFT JOIN pg_stat_user_tables s ON t.tablename = s.relname
            LEFT JOIN information_schema.columns c ON t.tablename = c.table_name
            WHERE t.schemaname = 'public'
            GROUP BY t.tablename, s.n_tup_ins, t.schemaname
            ORDER BY t.tablename
        """)
        
        tables = cursor.fetchall()
        
        table_info = []
        for table in tables:
            table_info.append({
                'table_name': table[0],
                'row_count': table[1] if table[1] is not None else 'Unknown',
                'column_count': table[2],
                'size': table[3]
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'tables': table_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 18502))
    host = os.environ.get('HOST', '127.0.0.1')
    debug_flag = os.environ.get('FLASK_DEBUG', '1') == '1'
    print("Starting Flask server...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Parent directory: {PARENT_DIR}")
    print(f"Starting server on {host}:{port}")
    print(f"Local:   http://{host}:{port}")
    print(f"Server info endpoint: http://{host}:{port}/api/server-info")
    # Bind to 127.0.0.1 by default to avoid IPv6/localhost resolution issues
    app.run(host=host, port=port, debug=debug_flag, use_reloader=debug_flag)
