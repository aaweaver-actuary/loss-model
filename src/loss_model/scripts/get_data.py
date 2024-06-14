import pandas as pd


def get_data():
    rpt_loss = pd.read_parquet(
        "/home/andy/loss-model/src/loss_model/data/rpt_loss.parquet"
    )

    paid_loss = pd.read_parquet(
        "/home/andy/loss-model/src/loss_model/data/paid_loss.parquet"
    )
    premium = pd.read_parquet(
        "/home/andy/loss-model/src/loss_model/data/premium.parquet"
    )

    return (
        rpt_loss.reset_index()
        .melt(var_name="dev", value_name="rpt_loss", id_vars=["ay"])
        .rename(columns={"ay": "acc"})
        .assign(dev=lambda x: x["dev"].astype(int))
        .assign(acc=lambda x: x["acc"].astype(int))
        .merge(premium.reset_index(), how="left", left_on="acc", right_on="ay")
        .drop(columns=["ay"])
        .merge(
            paid_loss.reset_index()
            .melt(var_name="dev", value_name="paid_loss", id_vars=["ay"])
            .rename(columns={"ay": "acc"})
            .assign(dev=lambda x: x["dev"].astype(int))
            .assign(acc=lambda x: x["acc"].astype(int)),
            how="left",
            on=["acc", "dev"],
        )
    )[["acc", "dev", "premium", "rpt_loss", "paid_loss"]]
