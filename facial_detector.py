import numpy as np
import os
from copy import deepcopy
import timeit
import ntpath
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parameters import *
from process_images import *

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_svc = None
        self.best_nn = None
        self.best_cnn = None
        self.best_cnn_dad = None
        self.best_cnn_deedee = None
        self.best_cnn_dexter = None
        self.best_cnn_mom = None

    def load_models(self):
        self.best_cnn = torch.load(os.path.join('models', 'best_cnn'))
        self.best_cnn_dad = torch.load(os.path.join('models', 'best_cnn_dad'))
        self.best_cnn_deedee = torch.load(os.path.join('models', 'best_cnn_deedee'))
        self.best_cnn_dexter = torch.load(os.path.join('models', 'best_cnn_dexter'))
        self.best_cnn_mom = torch.load(os.path.join('models', 'best_cnn_mom'))

    def get_positive_descriptors(self, positive_dir):
        print('Getting positive descriptors...')
        positive_descriptors = []
        for character_name in self.params.characters:
            for file in os.listdir(os.path.join(positive_dir, character_name)):
                print(''.join(['Processing ', file]))
                image = cv.imread(os.path.join(positive_dir, character_name, file), cv.IMREAD_GRAYSCALE)
                descriptor = hog(image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(self.params.cells_per_block, self.params.cells_per_block), orientations=self.params.orientations, feature_vector=True)
                positive_descriptors.append(descriptor)
                if self.params.use_flip:
                    descriptor = hog(np.fliplr(image), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
                    positive_descriptors.append(descriptor)
        return positive_descriptors
    
    def get_negative_descriptors(self, negative_dir):
        print('Getting negative descriptors...')
        negative_descriptors = []
        for file in os.listdir(negative_dir):
            print(''.join(['Processing ', file]))
            image = cv.imread(os.path.join(negative_dir, file), cv.IMREAD_GRAYSCALE)
            descriptor = hog(image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(self.params.cells_per_block, self.params.cells_per_block), orientations=self.params.orientations, feature_vector=True)
            negative_descriptors.append(descriptor)
        return negative_descriptors
    

    def train_svc(self, train_data, train_labels):
        svm_file_name = os.path.join('models', 'best_svc_%d_%d_%d' % (self.params.dim_hog_cell, self.params.number_negative_examples, self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_svc = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c = %f' % c)
            model = LinearSVC(C=c)
            model.fit(train_data, train_labels)
            acc = model.score(train_data, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        scores = best_model.decision_function(train_data)
        self.best_svc = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.savefig('distributie.png')
        plt.show()


    def train_nn(self, train_data, train_labels):
        nn_file_name = os.path.join('models', 'best_nn_%d_%d_%d' % (self.params.dim_hog_cell, self.params.number_negative_examples, self.params.number_positive_examples))
        if os.path.exists(nn_file_name):
            self.best_nn = torch.load(nn_file_name)
            return
        
        best_accuracy = 0
        best_model = None
        learning_rate = 0.001
        epochs = 10
        batch_size = 32
        train_data = torch.tensor(train_data).float()
        train_labels = torch.tensor(train_labels).float()
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = NeuralNetwork(input_size=train_data.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                data, labels = batch
                output = model(data)
                loss = criterion(output, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                output = model(train_data)
                acc = accuracy_score(train_labels.cpu().numpy(), output.detach().cpu().numpy() > 0.5)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = deepcopy(model)
                print('Epoch %d, loss %f, accuracy %f' % (epoch, loss.item(), acc))

        torch.save(best_model, nn_file_name)
        self.best_nn = best_model

        output = best_model(train_data)
        acc = accuracy_score(train_labels.cpu().numpy(), output.detach().numpy() > 0.5)
        print(f'Performanta clasificatorului optim: {acc}')


    def train_cnn(self, train_data, train_labels, character=None):
        if character is None:
            cnn_file_name = os.path.join('models', 'best_cnn_%d_%d' % (self.params.number_negative_examples, self.params.number_positive_examples))
        else:
            cnn_file_name = os.path.join('models', f'best_cnn_{character}')
        if os.path.exists(cnn_file_name):
            if character is None:
                self.best_cnn = torch.load(cnn_file_name)
            elif character == 'dad':
                self.best_cnn_dad = torch.load(cnn_file_name)
            elif character == 'deedee':
                self.best_cnn_deedee = torch.load(cnn_file_name)
            elif character == 'dexter':
                self.best_cnn_dexter = torch.load(cnn_file_name)
            elif character == 'mom':
                self.best_cnn_mom = torch.load(cnn_file_name)
            return
        
        best_accuracy = 0
        best_model = None
        learning_rate = 0.001
        epochs = 10
        batch_size = 32

        train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
        train_data = torch.tensor(train_data).float()
        train_labels = torch.tensor(train_labels).float()
        validation_data = torch.tensor(validation_data).float()
        validation_labels = torch.tensor(validation_labels).float()
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ConvolutionalNeuralNetwork(input_size=(train_data.shape[2], train_data.shape[3]))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                data, labels = batch
                output = model(data)
                loss = criterion(output, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_output = model(validation_data)
                val_loss = criterion(val_output, validation_labels.unsqueeze(1))
                val_acc = accuracy_score(validation_labels.cpu().numpy(), val_output.detach().cpu().numpy() > 0.5)
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model = deepcopy(model)
                print('Epoch %d, val loss %f, val accuracy %f' % (epoch, val_loss.item(), val_acc))

        torch.save(best_model, cnn_file_name)
        if character is None:
            self.best_cnn = torch.load(cnn_file_name)
        elif character == 'dad':
            self.best_cnn_dad = torch.load(cnn_file_name)
        elif character == 'deedee':
            self.best_cnn_deedee = torch.load(cnn_file_name)
        elif character == 'dexter':
            self.best_cnn_dexter = torch.load(cnn_file_name)
        elif character == 'mom':
            self.best_cnn_mom = torch.load(cnn_file_name)

        val_output = best_model(validation_data)
        val_acc = accuracy_score(validation_labels.cpu().numpy(), val_output.detach().cpu().numpy() > 0.5)
        print(f'Performanta clasificatorului optim: {val_acc}')


    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True: 
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True: 
                        if self.intersection_over_union(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold or self.intersection_ratio(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold: is_maximal[j] = False
                        else:
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
    

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou
    

    def intersection_ratio(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_b[3], bbox_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        
        patch_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        
        intersection_ratio = inter_area / patch_area if patch_area != 0 else 0
        return intersection_ratio
    

    def run_svc(self, test_images_path):
        test_files = glob.glob(test_images_path)
        detections = None  
        scores = np.array([]) 
        file_names = np.array([])  
        w = self.best_svc.coef_.T
        bias = self.best_svc.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Processing test image %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            img_detections = []
            img_scores = []

            for resize_factor in self.params.img_resizes:
                resized_img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
                inverse_resize_factor = 1 / resize_factor
                hog_descriptors = hog(resized_img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(self.params.cells_per_block, self.params.cells_per_block), orientations=self.params.orientations, feature_vector=False)
                num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1
                num_cells = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(num_rows - num_cells):
                    for x in range(num_cols - num_cells):
                        descr = hog_descriptors[y : y + num_cells, x : x + num_cells].flatten()
                        score = np.dot(descr, w)[0] + bias
                        if score > self.params.threshold:
                            img_scores.append(score)
                            x_min = int(x * self.params.dim_hog_cell * inverse_resize_factor)
                            y_min = int(y * self.params.dim_hog_cell * inverse_resize_factor)
                            x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) * inverse_resize_factor)
                            y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) * inverse_resize_factor)
                            img_detections.append([x_min, y_min, x_max, y_max])

            if len(img_detections) > 0:
                img_detections, img_scores = self.non_maximal_suppression(np.array(img_detections), np.array(img_scores), img.shape)
            if len(img_detections) > 0:
                if detections is None:
                    detections = img_detections
                else:
                    detections = np.concatenate((detections, img_detections))
                scores = np.append(scores, img_scores)
                img_names = [ntpath.basename(test_files[i]) for _ in range(len(img_detections))]
                file_names = np.append(file_names, img_names)

            end_time = timeit.default_timer()
            print('Processing time of test image %d/%d is %f sec.'% (i, num_test_images, end_time - start_time))

        return detections, scores, file_names
    

    def run_nn(self, test_images_path):
        test_files = glob.glob(test_images_path)
        detections = None 
        scores = np.array([])  
        file_names = np.array([])  
        num_test_images = len(test_files)
        self.best_nn.eval()

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Processing test image %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            img_detections = []
            img_scores = []

            for resize_factor in self.params.img_resizes:
                resized_img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
                inverse_resize_factor = 1 / resize_factor
                hog_descriptors = hog(resized_img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(self.params.cells_per_block, self.params.cells_per_block), orientations=self.params.orientations, feature_vector=False)
                num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1
                num_cells = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(num_rows - num_cells):
                    for x in range(num_cols - num_cells):
                        descr = hog_descriptors[y : y + num_cells, x : x + num_cells].flatten()
                        score = self.best_nn(torch.tensor(descr).float()).item()
                        if score > self.params.threshold_cnn:
                            img_scores.append(score)
                            x_min = int(x * self.params.dim_hog_cell * inverse_resize_factor)
                            y_min = int(y * self.params.dim_hog_cell * inverse_resize_factor)
                            x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) * inverse_resize_factor)
                            y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) * inverse_resize_factor)
                            img_detections.append([x_min, y_min, x_max, y_max])

            if len(img_detections) > 0:
                img_detections, img_scores = self.non_maximal_suppression(np.array(img_detections), np.array(img_scores), img.shape)
            if len(img_detections) > 0:
                if detections is None:
                    detections = img_detections
                else:
                    detections = np.concatenate((detections, img_detections))
                scores = np.append(scores, img_scores)
                img_names = [ntpath.basename(test_files[i]) for _ in range(len(img_detections))]
                file_names = np.append(file_names, img_names)

            end_time = timeit.default_timer()
            print('Processing time of test image %d/%d is %f sec.'% (i, num_test_images, end_time - start_time))

        return detections, scores, file_names
    

    def run_cnn(self, test_images_path, character=None, dim_window=(36, 36)):
        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        file_names = np.array([])
        num_test_images = len(test_files)
        
        if character is None:
            model = self.best_cnn
        elif character == 'dad':
            model = self.best_cnn_dad
        elif character == 'deedee':
            model = self.best_cnn_deedee
        elif character == 'dexter':
            model = self.best_cnn_dexter
        elif character == 'mom':
            model = self.best_cnn_mom
        model.eval()

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Processing test image %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i])
            img_detections = []
            img_scores = []

            for resize_factor in self.params.img_resizes:
                resized_img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
                inverse_resize_factor = 1 / resize_factor
                # Turn image into numpy array
                resized_img = np.array(resized_img)
                # Normalize image
                resized_img = resized_img / 255.0
                # Move color channel to the first dimension
                resized_img = np.moveaxis(resized_img, -1, 0)
                # Convert image to tensor
                resized_img = torch.tensor(resized_img).float()

                for y in range(0, resized_img.shape[1] - dim_window[0], self.params.stride):
                    for x in range(0, resized_img.shape[2] - dim_window[1], self.params.stride):
                        patch = resized_img[:, y : y + dim_window[0], x : x + dim_window[1]]
                        output = model(patch.unsqueeze(0)).item()
                        if output > self.params.threshold_cnn:
                            img_scores.append(output)
                            x_min = int(x * inverse_resize_factor)
                            y_min = int(y * inverse_resize_factor)
                            x_max = int((x + dim_window[1]) * inverse_resize_factor)
                            y_max = int((y + dim_window[0]) * inverse_resize_factor)
                            img_detections.append([x_min, y_min, x_max, y_max])

            if len(img_detections) > 0:
                img_detections, img_scores = self.non_maximal_suppression(np.array(img_detections), np.array(img_scores), img.shape[:2])
            if len(img_detections) > 0:
                if detections is None:
                    detections = img_detections
                else:
                    detections = np.concatenate((detections, img_detections))
                scores = np.append(scores, img_scores)
                img_names = [ntpath.basename(test_files[i]) for _ in range(len(img_detections))]
                file_names = np.append(file_names, img_names)

            end_time = timeit.default_timer()
            print('Processing time of test image %d/%d is %f sec.'% (i, num_test_images, end_time - start_time))

        return detections, scores, file_names
    

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.shape_after_conv = self.get_shape_after_conv(input_size)
        self.fc1 = nn.Linear(self.shape_after_conv, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, self.shape_after_conv)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    
    def get_shape_after_conv(self, input_size):
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        return dummy_output.numel()
    