import numpy as np
import pickle

model = pickle.load(open("C:/Users/renan/Google Drive/notebooks/beginning/diabetes_pred_with_streamlit/diabetes_model.sav", "rb"))

# taking a sample from dataset: 1,189,60,23,846,30.1,0.398,59 -> 1 = diabetics
input_data = (1,189,60,23,846,30.1,0.398,59)

#input_data to numpy array and reshape it
input_data = np.asarray(input_data).reshape(1, -1)

pred = model.predict(input_data)
if pred[0] == 0:
  print('The person isn\'t diabetics')
else:
  print ('The person is diabetics')