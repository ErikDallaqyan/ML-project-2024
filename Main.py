from sklearn.datasets import fetch_openml
from Utils import DataProccesor
import numpy as np 

heart_disease = fetch_openml(name="heart-disease", version=1, as_frame=True)
data = heart_disease.data
proccesor = DataProccesor(data)

