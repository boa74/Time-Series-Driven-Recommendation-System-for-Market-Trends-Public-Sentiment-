# Step 1 - import data from the Mongo DB

from pymongo import MongoClient
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

client = MongoClient('localhost',27017)
db = client.tutorial

# PostgreSQL connection
PG_USER = 'postgres'  
PG_PASSWORD = '123'  
DB_NAME = 'tutorial_db'

# First, create the database if it doesn't exist
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        user=PG_USER,
        password=PG_PASSWORD,
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"✓ Database '{DB_NAME}' created")
    else:
        print(f"✓ Database '{DB_NAME}' already exists")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Database setup: {e}")

# Now connect to the tutorial_db
pg_engine = create_engine(f'postgresql://{PG_USER}:{PG_PASSWORD}@localhost:5432/{DB_NAME}')

# Step 2 - import the dataset - depression index from the csv file.
col_1 = db.Depression_index
docs = list(col_1.find())
print(len(docs)) 

docs_1 = list(col_1.find({}, {"_id": 0, "date": 1, "depression": 1}))

df_1 = pd.DataFrame(docs_1)  # DataFrame
df_1["date"] = pd.to_datetime(df_1["date"], utc=True).dt.date  # date cleaning
df_1 = df_1.rename(columns={"depression": "depression_index"})  # change the column names

print(df_1.head())

# Step 3 - import the dataset - CCnews_Depression from the MongoDB
col_2 = db.CCnews_Depression
docs_2 = list(col_2.find({}, {"_id": 0, "date":1, "title": 1, "text": 1}))
# DataFrame
df_2 = pd.DataFrame(docs_2)
df_2["Date"] = pd.to_datetime(df_2["date"]).dt.strftime("%Y-%m-%d")
print(df_2.head())

count_docs= col_2.count_documents({
    "text":{"$regex":"DEPRESSION","$options":"i"}
})
print(count_docs)

# --- 2) Clean date into exactly "YYYY-MM-DD" ---
df_2["date"] = pd.to_datetime(df_2["date"], errors="coerce")
df_2["date"] = df_2["date"].dt.strftime("%Y-%m-%d")   # FORCE format

print(df_2.head())

# Step 4 - import the dataset - StockData from the MongoDB
col_3 = db.StockData
docs_3 = list(col_3.find({}, {"_id": 0}))

df_3 = pd.DataFrame(docs_3)

df_3["Date"] = pd.to_datetime(df_3["Date"]).dt.strftime("%Y-%m-%d")
print(df_3.head())


# Step 4 - import the dataset - StockData from the MongoDB
col_3 = db.StockData
docs_3 = list(col_3.find({}, {"_id": 0}))

df_3 = pd.DataFrame(docs_3)

df_3["Date"] = pd.to_datetime(df_3["Date"]).dt.strftime("%Y-%m-%d")
print(df_3.head())


# Step 6 - import the dataset - SP500 from the MongoDB

col_5 = db.SP500
docs_5 = list(col_5.find({}, {"_id": 0}))
df_5 = pd.DataFrame(docs_5)
df_5["Date"] = pd.to_datetime(df_5["Date"]).dt.strftime("%Y-%m-%d")
print(df_5.head())

# Step 7 - import the dataset - Rainfall from the MongoDB

col_6 = db.Rainfall
docs_6 = list(col_6.find({}, {"_id": 0}))
df_6 = pd.DataFrame(docs_6)
df_6["Date"] = pd.to_datetime(df_6["Date"]).dt.strftime("%Y-%m-%d")
print(df_6.head())

# Step 8 - Send all DataFrames to PostgreSQL
print("\nSending data to PostgreSQL...")

# Send depression_index data
df_1.to_sql('depression_index', pg_engine, if_exists='replace', index=False)
print("✓ depression_index table created")

# Send CCnews_Depression data
df_2.to_sql('ccnews_depression', pg_engine, if_exists='replace', index=False)
print("✓ ccnews_depression table created")

# Send StockData
df_3.to_sql('stock_data', pg_engine, if_exists='replace', index=False)
print("✓ stock_data table created")

# Send SP500 data
df_5.to_sql('sp500', pg_engine, if_exists='replace', index=False)
print("✓ sp500 table created")

# Send Rainfall data
df_6.to_sql('rainfall', pg_engine, if_exists='replace', index=False)
print("✓ rainfall table created")

print("\nAll data successfully sent to PostgreSQL!")