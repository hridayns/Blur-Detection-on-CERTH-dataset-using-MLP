import numpy as np
import cv2
import os
import os.path as path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
#--------------------------------------------TRAIN SCRIPT-------------------------------------------------------------------------

training_folder = os.path.join(dir_path,'CERTH_ImageBlurDataset','TrainingSet')
artificial_blur_folder = 'Artificially-Blurred'
natural_blur_folder = 'Naturally-Blurred'
undistorted_folder = 'Undistorted'

artificial_blur_path = os.path.join(training_folder,artificial_blur_folder)
natural_blur_path = os.path.join(training_folder,natural_blur_folder)
undistorted_path = os.path.join(training_folder,undistorted_folder)

def variance_of_laplacian(image):
	return np.var(cv2.Laplacian(image, cv2.CV_64F))

def load_train_images(folders):
	xdata = []
	ydata = []
	for folder in folders:
		i = 0
		if folder == artificial_blur_path or folder == natural_blur_path:
			y_val = 1
		else:
			y_val = -1
		for filename in os.listdir(folder):
			img = cv2.imread(os.path.join(folder,filename))
			if img is not None:
				v = variance_of_laplacian(img)
				print(i,v)
				xdata.append(v)
				ydata.append(y_val)
				i = i + 1
	xdata_arr = np.asarray(xdata)
	ydata_arr = np.asarray(ydata)
	return np.reshape(xdata_arr,(xdata_arr.shape[0],1)),ydata_arr


train_data_folders = [artificial_blur_path,natural_blur_path,undistorted_path]
X_train,y_train = load_train_images(train_data_folders)

scaler = StandardScaler()
scaler.fit(X_train)

#------------------------------------------------TEST SCRIPT----------------------------------------------------------

eval_folder = os.path.join(dir_path,'CERTH_ImageBlurDataset','EvaluationSet')
digital_blur_folder = 'DigitalBlurSet'
natural_blur_folder = 'NaturalBlurSet'

digital_blur_path = os.path.join(eval_folder,digital_blur_folder)
natural_blur_path = os.path.join(eval_folder,natural_blur_folder)

natural_blur_file = os.path.join(eval_folder,'NaturalBlurSet.xlsx')
digital_blur_file = os.path.join(eval_folder,'DigitalBlurSet.xlsx')

natural_blur_data = pd.read_excel(natural_blur_file)
digital_blur_data = pd.read_excel(digital_blur_file)

digital_blur_data.rename(columns={'Unnamed: 1':'Blur Label'}, inplace=True)
digital_blur_data.rename(columns={'MyDigital Blur':'Image Name'}, inplace=True)

def load_test_images(folders):
	xdata = []
	ydata = []
	for folder in folders:
		i = 0
		if folder == natural_blur_path:
			df = natural_blur_data
		else:
			df = digital_blur_data
		for filename in os.listdir(folder):
			if folder == natural_blur_path:
				img_name = path.splitext(filename)[0]
			else:
				img_name = filename
			img = cv2.imread(os.path.join(folder,filename))
			if img is not None:
				v = variance_of_laplacian(img)
				print(i,v)
				xdata.append(v)
				img_idx = 0
				for idx,val in df.iterrows():
					if(img_name in val['Image Name']):
						img_idx = idx
						break

				y_val = df.iloc[img_idx,1]
				ydata.append(y_val)
				i = i + 1
	xdata_arr = np.asarray(xdata)
	ydata_arr = np.asarray(ydata)
	return np.reshape(xdata_arr,(xdata_arr.shape[0],1)),np.reshape(ydata_arr,(ydata_arr.shape[0],1))

test_data_folders = [digital_blur_path,natural_blur_path]
X_test,y_test = load_test_images(test_data_folders)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print('Accuracy: ', mlp.score(X_test,y_test) * 100.0)
