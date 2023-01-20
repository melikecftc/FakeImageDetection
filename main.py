import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load your dataset of images
real_images = ('C:\\Users\\EXCALIBUR\\PycharmProjects\\FakeImageDetection')
fake_images = ('C:\\Users\\EXCALIBUR\\PycharmProjects\\FakeImageDetection')

# Create an empty list to store the histograms
histograms = []

# Iterate through the real images
for real_image in real_images:
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram of the image
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # Flatten the histogram and normalize it
    hist = cv2.normalize(hist, hist).flatten()
    # Append the histogram to the list
    histograms.append(hist)

# Iterate through the fake images
for fake_image in fake_images:
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram of the image
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # Flatten the histogram and normalize it
    hist = cv2.normalize(hist, hist).flatten()
    # Append the histogram to the list
    histograms.append(hist)

# Create a list of labels for the images
labels = [1] * len(real_images) + [0] * len(fake_images)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2)

# Initialize the SVM classifier
clf = svm.SVC()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Iterate through the predictions and print them
for prediction in predictions:
    if prediction == 1:
        print("real")
    else:
        print("fake")




