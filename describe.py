import argparse
import pandas as pd
import numpy as np


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="filename to get describe for"
    )
    return parser.parse_args()


def ft_count(df_slice: pd.DataFrame) -> float:
    pass

def ft_mean(df_slice: pd.DataFrame) -> float:
    pass

def ft_std(df_slice: pd.DataFrame) -> float:
    pass

def ft_min(df_slice: pd.DataFrame) -> float:
    pass

def ft_q1(df_slice: pd.DataFrame) -> float:
    pass

def ft_q2(df_slice: pd.DataFrame) -> float:
    pass

def ft_q3(df_slice: pd.DataFrame) -> float:
    pass

def ft_max(df_slice: pd.DataFrame) -> float:
    pass


def main():
    args = parse()
    data = pd.read_csv(args.dataset)
    
    stats = {
        "count": ft_count(),
        "mean": ft_mean(),
        "std": ft_std(),
        "min": ft_min(),
        "25%": ft_q1(),
        "50%": ft_q2(),
        "75%": ft_q3(),
        "max": ft_max()
    }
    describe = pd.DataFrame()

    for column_name in data:
        describe[column_name] = [do_stats(stat, data[column_name]) for stat in stats]

    # describe.columns = ["Index"]
    describe.index = stats
    
    
    print(describe)


if __name__ == "__main__":
    main()
    