import argparse
import pandas as pd
import numpy as np
# from ft_pypackage import ft_type

# pd.options.display.float_format = '{:.6f}'.format

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="filename to get describe for"
    )
    return parser.parse_args()

def is_valid(value: int | float) -> bool:
    """
    Check if the given value is a valid integer or float.

    This function first checks if the value is not NaN (Not a Number) using pandas' `isna` function.
    If the value is not NaN, it then checks if the value is an instance of either `int` or `float`.

    Args:
        value (int | float): The value to be checked.

    Returns:
        bool: True if the value is a valid integer or float and not NaN, False otherwise.
    """
    if not pd.isna(value):
        return isinstance(value, (int, float))
    return False


# @ft_type(float)
def ft_count(df_slice: pd.DataFrame) -> float:
    count = 0
    for value in df_slice:
        if is_valid(value):
            count += 1
    return count

# @ft_type(float)
def ft_mean(df_slice: pd.DataFrame) -> float:
    total = 0
    count = 0
    for value in df_slice:
        if is_valid(value):
            total += value
            count += 1
    return total / count

# @ft_type(float)
def ft_variance(df_slice: pd.DataFrame) -> float:
    mean = ft_mean(df_slice)
    sum_of_diff = 0
    count = ft_count(df_slice)
    for value in df_slice:
        if is_valid(value):
            sum_of_diff += (value - mean)**2
    return sum_of_diff / (count - 1) if count > 1 else float('nan')

# @ft_type(float)
def ft_std(df_slice: pd.DataFrame) -> float:
    return ft_variance(df_slice)**0.5

# @ft_type(float)
def ft_skewness(df_slice: pd.DataFrame) -> float:
    mean = ft_mean(df_slice)
    std = ft_std(df_slice)
    count = ft_count(df_slice)
    skewness = 0
    for value in df_slice:
        if is_valid(value):
            skewness += ((value - mean) / std) ** 3
    if count > 2:
        skewness *= count / ((count - 1) * (count - 2))
    else:
        skewness = float('nan')
    return skewness


# @ft_type(float)
def ft_min(df_slice: pd.DataFrame) -> float:
    for value in df_slice:
        if is_valid(value):
            if "minimum" not in locals():
                minimum = value
            elif value < minimum:
                minimum = value
    return minimum


# @ft_type(float)
def ft_q1(df_slice: pd.DataFrame) -> float:
    df = df_slice.dropna().sort_values()
    count = len(df)
    if count == 0:
        return float('nan')
    
    # Calculate the exact position of the 25th percentile
    q1_position = 0.25 * (count - 1)
    
    # Interpolate
    lower_index = int(q1_position)
    upper_index = lower_index + 1
    
    lower_value = df.iloc[lower_index]
    upper_value = df.iloc[upper_index]
    interpolation = lower_value + (upper_value - lower_value) * (q1_position - lower_index)
    
    return interpolation


# @ft_type(float)
def ft_q2(df_slice: pd.DataFrame) -> float:
    df = df_slice.dropna().sort_values()
    count = len(df)
    if count == 0:
        return float('nan')
    
    q2_position = 0.5 * (count - 1)
    
    # Interpolate
    lower_index = int(q2_position)
    upper_index = lower_index + 1
    
    lower_value = df.iloc[lower_index]
    upper_value = df.iloc[upper_index]
    print(f"interpolation = {lower_value} + ({upper_value} - {lower_value}) * ({q2_position} - {lower_index})")
    interpolation = lower_value + (upper_value - lower_value) * (q2_position - lower_index)
    
    return interpolation


# @ft_type(float)
def ft_q3(df_slice: pd.DataFrame) -> float:
    df = df_slice.dropna().sort_values()
    count = len(df)
    if count == 0:
        return float('nan')

    q3_position = 0.75 * (count - 1)
    
    # Interpolate
    lower_index = int(q3_position)
    upper_index = lower_index + 1
    
    lower_value = df.iloc[lower_index]
    upper_value = df.iloc[upper_index]
    interpolation = lower_value + (upper_value - lower_value) * (q3_position - lower_index)
    
    return interpolation


# @ft_type(float)
def ft_max(df_slice: pd.DataFrame) -> float:
    for value in df_slice:
        if is_valid(value):
            if "maximum" not in locals():
                maximum = value
            elif value > maximum:
                maximum = value
    return maximum

def main():
    args = parse()
    data = pd.read_csv(args.dataset)
    dropables = [col for col in data if data[col].dtype == "object"]
    data.drop(dropables, axis=1, inplace=True)

    stats = {
        "count": ft_count,
        "mean": ft_mean,
        "variance": ft_variance,
        "std": ft_std,
        "skewness": ft_skewness,
        "min": ft_min,
        "25%": ft_q1,
        "50%": ft_q2,
        "75%": ft_q3,
        "max": ft_max
    }
    describe = pd.DataFrame()

    for column_name in data:
        describe[column_name] = [func(data[column_name]) for _, func in stats.items()]

    # describe.columns = ["Index"]
    describe.index = stats.keys()
    
    print(describe)
    print(describe["Arithmancy"].dtype)


def test():
    args = parse()
    data = pd.read_csv(args.dataset)
    dropables = [col for col in data if data[col].dtype == "object"]
    data.drop(dropables, axis=1, inplace=True)

    for value in data["Arithmancy"]:
        if pd.isna(value):
            print(value, "is Nan")

if __name__ == "__main__":
    main()
    # test()
