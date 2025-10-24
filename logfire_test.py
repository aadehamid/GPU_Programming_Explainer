# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi==0.119.1",
#     "logfire==4.14.0",
#     "polars==1.34.0",
#     "python-dotenv==1.1.1",
#     "sqlalchemy==2.0.44",
#     'logfire[fastapi,sqlalchemy,sqlite3,httpx]',
#     'httpx',
#     "pyarrow==21.0.0",
#     "pandas==2.3.3",
#     "sqlglot==27.28.1",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import logfire
    from fastapi import FastAPI
    import os
    from dotenv import load_dotenv
    import polars as pl

    # load_dotenv("/Users/hamidadesokan/Dropbox/2_Skill_Development/mojo/GPU_Programming_Explainer/.env")
    return logfire, mo, pl


@app.cell
def _(logfire):
    # LOGFIRE_TOKEN = os.getenv("LOGFIRE")
    logfire.configure(environment="local", service_name="Batch Job")
    return


@app.cell
def _(logfire):
    with logfire.span("nested"):
        logfire.info("Hello from Marimo!")
        logfire.debug("I am debugging!")
    return


@app.cell
def _(pl):
    df_bigmac = pl.read_csv("https://calmcode.io/static/data/bigmac.csv")
    return (df_bigmac,)


@app.cell
def _(df_bigmac):
    df_bigmac
    return


@app.cell
def _(logfire):
    import sqlalchemy

    DATABASE_URL = "sqlite:///db.sqlite"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    logfire.instrument_sqlalchemy(engine=engine)
    return (engine,)


@app.cell
def _(df_bigmac, engine, logfire):
    df_bigmac_pandas = df_bigmac.to_pandas()

    # insert the data into sqlite database
    df_bigmac_pandas.to_sql(
        name="bigmac", con=engine, if_exists="replace", index=False
    )
    logfire.info(
        f"Succesfully inserted {len(df_bigmac)} rows into the 'bigmac' table"
    )
    return


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        select * from bigmac limit 10;
        """,
        engine=engine,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
