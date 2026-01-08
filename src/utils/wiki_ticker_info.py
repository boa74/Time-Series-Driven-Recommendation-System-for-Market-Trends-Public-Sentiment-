import pandas as pd
import urllib.request
# Need to do mapping to merge the stock data - ticker with real company name and the industry.


# 1) read the data from wikipedia with proper headers
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Create a request with headers to avoid 403 error
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

# Read the HTML content
with urllib.request.urlopen(req) as response:
    html_content = response.read()

tables = pd.read_html(html_content)
sp500_const = tables[1]  # Table 1 contains the S&P 500 constituents

# 2) extract the columns which are needed for the analysis
dim_company = sp500_const.rename(columns={
    "Symbol": "ticker",
    "Security": "company_name",
    "GICS Sector": "sector",
    "GICS Sub-Industry": "industry"
})[["ticker", "company_name", "sector", "industry"]]

print("Company dimension table:")
print(dim_company.head())
dim_company.to_csv("company_info.csv", index=False)
print("\ncompany_info.csv created")
print(f"Total companies in S&P 500: {len(dim_company)}\n")

# 3) Read stock_data.csv
print("Reading stock_data.csv...")
stock_data = pd.read_csv("csv_exports/stock_data.csv")
print(f"Stock data rows: {len(stock_data):,}")
print(f"Unique tickers in stock data: {stock_data['ticker'].nunique()}\n")

# 4) Merge stock data with company info
print("Merging stock data with company information...")
stock_data_enriched = stock_data.merge(
    dim_company, 
    on="ticker", 
    how="left"
)

# 5) Check merge results
print(f"Enriched stock data rows: {len(stock_data_enriched):,}")
print(f"Tickers with company info: {stock_data_enriched['company_name'].notna().sum():,}")
print(f"Tickers without company info: {stock_data_enriched['company_name'].isna().sum():,}\n")

# 6) Display sample of enriched data
print("Sample of enriched stock data:")
print(stock_data_enriched[['date', 'ticker', 'close', 'company_name', 'sector', 'industry']].head(10))

# 7) Save enriched stock data
stock_data_enriched.to_csv("csv_exports/stock_data_wiki.csv", index=False)
print("\n✓ stock_data_wiki.csv created successfully!")
print(f"  Location: csv_exports/stock_data_wiki.csv")
