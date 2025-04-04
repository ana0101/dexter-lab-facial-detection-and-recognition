import os
import cv2 as cv
import numpy as np
import shutil

from parameters import *

class ProcessImages:
    def __init__(self, params: Parameters):
        self.params = params

    def make_directories(self):
        os.makedirs('evaluare/fisiere_solutie', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria/task1', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria/task2', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria_bonus', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria_bonus/task1', exist_ok=True)
        os.makedirs('evaluare/fisiere_solutie/332_Hodivoianu_Anamaria_bonus/task2', exist_ok=True)


    def copy_images_and_annotations(params, new_folder_name, output_annotation_file):
        new_folder_path = os.path.join('antrenare', new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        output_annotation_file = os.path.join('antrenare', output_annotation_file)

        character_folders = ['dad', 'deedee', 'dexter', 'mom']
        annotation_files = {character: f'{character}_annotations.txt' for character in character_folders}

        with open(output_annotation_file, 'w') as out_file:
            for character in character_folders:
                character_folder_path = os.path.join('antrenare', character)
                annotation_file_path = os.path.join('antrenare', annotation_files[character])
                
                with open(annotation_file_path, 'r') as ann_file:
                    annotations = ann_file.readlines()
                
                for img_name in os.listdir(character_folder_path):
                    if img_name.endswith(('.jpg')): 
                        src_path = os.path.join(character_folder_path, img_name)
                        new_img_name = f"{character}_{os.path.splitext(img_name)[0]}{os.path.splitext(img_name)[1]}"
                        dest_path = os.path.join(new_folder_path, new_img_name)
                        shutil.copy(src_path, dest_path)
                        print(f"Copied and renamed {src_path} to {dest_path}")

                        for annotation in annotations:
                            if annotation.startswith(img_name):
                                parts = annotation.strip().split()
                                new_annotation = f"{new_img_name} {' '.join(parts[1:])}\n"
                                out_file.write(new_annotation)
                                print(f"Copied annotation: {new_annotation.strip()}")


    def copy_and_rename_images(params, source_folder, target_folder):
        target_folder_path = os.path.join(source_folder, target_folder)
        os.makedirs(target_folder_path, exist_ok=True)

        character_folders = ['dad', 'deedee', 'dexter', 'mom', 'unknown']

        for character in character_folders:
            character_folder_path = os.path.join(source_folder, character)
            for img_name in os.listdir(character_folder_path):
                if img_name.endswith(('.jpg')): 
                    src_path = os.path.join(character_folder_path, img_name)
                    new_img_name = f"{character}_{img_name}"
                    dest_path = os.path.join(target_folder_path, new_img_name)
                    shutil.copy(src_path, dest_path)
                    print(f"Copied and renamed {src_path} to {dest_path}")


    def get_annotations(self, character_name):
        with open(os.path.join('antrenare', f'{character_name}_annotations.txt'), 'r') as file:
            lines = file.readlines()
            annotations = {}
            for line in lines:
                line = line.strip().split()
                file_name = line[0]
                x_min, y_min, x_max, y_max = map(int, line[1:5])
                if file_name not in annotations:
                    annotations[file_name] = []
                annotations[file_name].append((x_min, y_min, x_max, y_max, line[5]))
        return annotations


    def get_positive_examples(self):
        for character in self.params.known_characters:
            annotations = self.get_annotations(character)
            character_dir = os.path.join('antrenare', character)
            for file in os.listdir(character_dir):
                image = cv.imread(os.path.join(character_dir, file))
                for annotation in annotations[file]:
                    x_min, y_min, x_max, y_max, character_name = annotation
                    roi = image[y_min:y_max, x_min:x_max]
                    cv.imwrite(os.path.join('antrenare/positive_examples', character_name, file), roi)


    def get_positive_examples_square(self):
        for character in self.params.known_characters:
            character_dir = os.path.join('antrenare', character)
            annotations = self.get_annotations(os.path.basename(character_dir))
            for file in os.listdir(character_dir):
                image = cv.imread(os.path.join(character_dir, file))
                for annotation in annotations[file]:
                    x_min, y_min, x_max, y_max, character_name = annotation
                    roi = image[y_min:y_max, x_min:x_max]
                    
                    # Calculate the center and size of the square crop
                    height, width = roi.shape[:2]
                    size = max(height, width)
                    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    
                    # Calculate new coordinates to make the ROI square
                    x_start = max(center_x - size // 2, 0)
                    y_start = max(center_y - size // 2, 0)
                    x_end = min(center_x + size // 2, image.shape[1])
                    y_end = min(center_y + size // 2, image.shape[0])
                    
                    # Crop the image to a square
                    square_roi = image[y_start:y_end, x_start:x_end]
                    square_roi = cv.resize(square_roi, (self.params.dim_window[0], self.params.dim_window[1]))
                    
                    cv.imwrite(os.path.join('antrenare/positive_examples_square', character_name, file), square_roi)


    def get_positive_examples_character(self, source_folder, dest_folder, character, dim_window=(36, 42)):
        annotations = self.get_annotations(character)
        for file in os.listdir(source_folder):
            image = cv.imread(os.path.join(source_folder, file))
            for annotation in annotations[file]:
                x_min, y_min, x_max, y_max, character_name = annotation
                if character_name != character:
                    continue
                roi = image[y_min:y_max, x_min:x_max]
                
                # Calculate the desired aspect ratio
                desired_height, desired_width = dim_window
                height, width = roi.shape[:2]
                current_aspect_ratio = width / height
                desired_aspect_ratio = desired_width / desired_height

                if current_aspect_ratio > desired_aspect_ratio:
                    # Add pixels to height
                    new_height = int(width / desired_aspect_ratio)
                    pad_top = (new_height - height) // 2
                    pad_bottom = new_height - height - pad_top
                    y_start = max(y_min - pad_top, 0)
                    y_end = min(y_max + pad_bottom, image.shape[0])
                    roi = image[y_start:y_end, x_min:x_max]
                else:
                    # Add pixels to width
                    new_width = int(height * desired_aspect_ratio)
                    pad_left = (new_width - width) // 2
                    pad_right = new_width - width - pad_left
                    x_start = max(x_min - pad_left, 0)
                    x_end = min(x_max + pad_right, image.shape[1])
                    roi = image[y_min:y_max, x_start:x_end]

                # Resize the ROI
                resized_roi = cv.resize(roi, (desired_width, desired_height))
                
                # Save the resized ROI
                cv.imwrite(os.path.join(dest_folder, file), resized_roi)


    def get_negative_examples_square(self, sizes=[36, 72, 108, 144], intersection_threshold=0.2):
        print('Generating negative examples...')
        num_negative_per_size = self.params.number_negative_examples // len(sizes)
        print(f'num_negative_per_size: {num_negative_per_size}')
        for size in sizes:
            print(f'Generating negative examples for size {size}...')
            num_negative_per_image = num_negative_per_size // 4000
            print(f'num_negative_per_image: {num_negative_per_image}')
            example_index = 0
            while example_index < num_negative_per_size:
                print(f'Generating negative examples for size {size}... {example_index}/{num_negative_per_size}')
                for character_name in self.params.known_characters:
                    if example_index >= num_negative_per_size:  
                        break
                    for file in os.listdir(os.path.join('antrenare', character_name)):
                        if example_index >= num_negative_per_size:
                            break
                        annotations = self.get_annotations(os.path.basename(character_name))
                        image = cv.imread(os.path.join('antrenare', character_name, file))
                        for i in range(num_negative_per_image):
                            if example_index >= num_negative_per_size:
                                break
                            num_tries = 0
                            while num_tries < 10:
                                num_tries += 1
                                x_min = np.random.randint(0, image.shape[1] - size)
                                y_min = np.random.randint(0, image.shape[0] - size)
                                x_max, y_max = x_min + size, y_min + size
                                roi = image[y_min:y_max, x_min:x_max]
                                
                                # Check intersection with all annotations
                                too_much_intersection = False
                                for annotation in annotations[file]:
                                    ax_min, ay_min, ax_max, ay_max, _ = annotation
                                    intersection_ratio = self.intersection_over_union((x_min, y_min, x_max, y_max), (ax_min, ay_min, ax_max, ay_max))
                                    if intersection_ratio > intersection_threshold:
                                        too_much_intersection = True
                                        break
                                
                                if not too_much_intersection:
                                    roi = cv.resize(roi, (self.params.dim_window, self.params.dim_window))
                                    cv.imwrite(os.path.join('antrenare/negative_examples_square', f'{size}_{example_index}.jpg'), roi)
                                    example_index += 1
                                    break

    
    def get_negative_examples_character(self, character, dim_window, resizes=[1, 2, 3, 4], intersection_threshold=0.2):
        print(f'Generating negative examples for character {character}...')
        num_negative_per_resize = self.params.number_negative_examples // len(resizes)
        print(f'num_negative_per_resize: {num_negative_per_resize}')
        for resize in resizes:
            height = int(dim_window[0] * resize)
            width = int(dim_window[1] * resize)
            print(f'Generating negative examples for resize {resize}...')
            num_negative_per_image = num_negative_per_resize // 4000
            print(f'num_negative_per_image: {num_negative_per_image}')
            example_index = 0
            while example_index < num_negative_per_resize:
                print(f'Generating negative examples for resize {resize}... {example_index}/{num_negative_per_resize}')
                for character_name in self.params.known_characters:
                    if example_index >= num_negative_per_resize:  
                        break
                    for file in os.listdir(os.path.join('antrenare', character_name)):
                        if example_index >= num_negative_per_resize:
                            break
                        annotations = self.get_annotations(os.path.basename(character_name))
                        image = cv.imread(os.path.join('antrenare', character_name, file))
                        for i in range(num_negative_per_image):
                            if example_index >= num_negative_per_resize:
                                break
                            num_tries = 0
                            while num_tries < 10:
                                num_tries += 1
                                x_min = np.random.randint(0, image.shape[1] - width)
                                y_min = np.random.randint(0, image.shape[0] - height)
                                x_max, y_max = x_min + width, y_min + height
                                roi = image[y_min:y_max, x_min:x_max]

                                # Check intersection with character annotations
                                too_much_intersection = False
                                for annotation in annotations[file]:
                                    ax_min, ay_min, ax_max, ay_max, charac = annotation
                                    if charac != character:
                                        continue
                                    intersection_ratio = self.intersection_over_union((x_min, y_min, x_max, y_max), (ax_min, ay_min, ax_max, ay_max))
                                    if intersection_ratio > intersection_threshold:
                                        too_much_intersection = True
                                        break

                                if not too_much_intersection:
                                    roi = cv.resize(roi, (dim_window[1], dim_window[0]))
                                    cv.imwrite(os.path.join(f'antrenare/negative_examples_{character}', f'{character}_{resize}_{example_index}.jpg'), roi)
                                    example_index += 1
                                    break


    def get_numpy_images(self, folder):
        images = []
        for file in os.listdir(folder):
            image = cv.imread(os.path.join(folder, file))
            images.append(image)
        return np.array(images)
    
    def normalize_images(self, images):
        return images / 255.0


    def intersection_ratio(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_b[3], bbox_b[3])

        # Calculate the intersection area
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        
        # Calculate the area of the random patch
        patch_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])

        # Calculate the area of the annotation
        annotation_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        
        # Calculate the intersection ratio between the random patch and the annotation
        intersection_ratio_1 = inter_area / patch_area if patch_area != 0 else 0

        # Calculate the intersection ratio between the annotation and the random patch
        intersection_ratio_2 = inter_area / annotation_area if annotation_area != 0 else 0

        # Get the maximum intersection ratio
        intersection_ratio = max(intersection_ratio_1, intersection_ratio_2)

        return intersection_ratio
    

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


    def get_mean_and_std_aspect_ratio(self, folder):
        aspect_ratios = []
        for file in os.listdir(folder):
            image = cv.imread(os.path.join(folder, file))
            aspect_ratios.append(image.shape[1] / image.shape[0])
        mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        return mean_aspect_ratio, sum([(aspect_ratio - mean_aspect_ratio) ** 2 for aspect_ratio in aspect_ratios]) / len(aspect_ratios)
    

    def get_max_and_min_dim_pos_examples(self):
        max_dim = 0
        min_dim = 1000
        for character_name in self.params.characters:
            for file in os.listdir(os.path.join('antrenare/positive_examples', character_name)):
                image = cv.imread(os.path.join('antrenare/positive_examples', character_name, file))
                max_dim = max(max_dim, max(image.shape[:2]))
                min_dim = min(min_dim, min(image.shape[:2]))
        return max_dim, min_dim
    