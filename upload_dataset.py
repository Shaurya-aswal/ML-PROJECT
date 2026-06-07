import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


password = quote_plus("aswal@29")

engine = create_engine(
    f"mysql+pymysql://root:{password}@localhost/ml_project"
)

df = pd.read_csv("/Users/apple/Desktop/NYC.csv")

df.to_sql(
    "nyc",
    con=engine,
    if_exists="replace",
    index=False
)

print("Dataset uploaded successfully!")


