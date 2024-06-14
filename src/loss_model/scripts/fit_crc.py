import pandas as pd
from cmdstanpy import CmdStanModel
from pathlib import Path
from get_data import get_data
import pickle as pkl
from gzip import GzipFile
import duckdb
from datetime import datetime

SCRIPT_DIR = "/home/andy/loss-model/stan"
DB_PATH = "/home/andy/loss-model/src/loss_model/scripts/results.db"


def fit_crc():
    # Load the model from this file:
    stan_file = Path(f"{SCRIPT_DIR}/crc.stan")

    # Get the data
    loss_data = get_data()
    data = {
        "N": loss_data.shape[0],
        "N_w": loss_data.acc.drop_duplicates().shape[0],
        "N_d": loss_data.dev.drop_duplicates().shape[0],
        "acc__raw": loss_data.acc.to_numpy(),
        "dev": loss_data.dev.to_numpy(),
        "premium": loss_data.premium.astype(float).to_numpy(),
        "cum_rpt_loss": loss_data.rpt_loss.astype(float).to_numpy(),
    }

    # Compile the model
    model = CmdStanModel(stan_file=stan_file)

    # Fit the model
    fit = model.sample(
        data=data,
        show_console=True,
        show_progress=True,
        chains=4,
        iter_warmup=1000,
        iter_sampling=2500,
        seed=42,
    )

    # Save & compress the fit
    with GzipFile("crc.pkl.gz", "wb") as f:
        pkl.dump(fit, f)

    # Save the fit to a database
    conn = duckdb.connect(DB_PATH)

    datadf = pd.DataFrame(
        {
            "acc": data["acc__raw"],
            "dev": data["dev"],
            "premium": data["premium"],
            "cum_rpt_loss": data["cum_rpt_loss"],
        }
    )

    conn.register("datadf", datadf)

    conn.execute(
        """
        CREATE OR REPLACE TABLE data (
            rowid INTEGER PRIMARY KEY,
            acc INTEGER,
            dev INTEGER,
            premium DOUBLE,
            cum_rpt_loss DOUBLE
        );
        """
    )

    conn.execute(
        "INSERT INTO data SELECT row_number() over () as rowid, * FROM datadf;"
    )

    fit_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("create or replace table timestamp (timestamp TIMESTAMP);")
    conn.execute(f"insert into timestamp values ('{fit_timestamp}');")

    for variable in [
        "alpha__loss",
        "beta__rpt_loss",
        "log_elr",
        "mu__rpt_loss",
        "log_lik__rpt_loss",
        "full_mu__rpt_loss",
        "cum_rpt_loss__pred",
        "cum_rpt_loss__residual",
        "inc_rpt_loss__pred",
        "inc_rpt_loss__residual",
        "generated_cum_rpt_loss__by_acc",
        "generated_cum_rpt_loss__by_acc__next_cal",
        "generated_cum_rpt_loss__by_acc__ultimate",
    ]:
        print(f"Saving {variable} to database...")
        fit_df = fit.draws_pd(variable)
        variable_cols = fit_df.columns.tolist()

        conn.register("fit_df", fit_df)

        def col_fmt(x):
            return x.replace("[", "__").replace("]", "")

        conn.execute(f"""
            CREATE OR REPLACE TABLE {variable} (
                {', '.join([col_fmt(col) + ' DOUBLE' for col in variable_cols])}
            );
            """)

        conn.execute(f"INSERT INTO {variable} SELECT * FROM fit_df;")

        conn.execute("DROP VIEW fit_df;")

    fit.draws_pd().to_parquet("crc_draws.parquet")

    print("Done!")

    conn.close()

    return fit


# if __name__ == "__main__":
#     fit_crc()
