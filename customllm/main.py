from sklearn import datasets
import pandas as pd

house_data = datasets.fetch_california_housing()

house_data_df = pd.DataFrame(house_data["data"],columns=house)