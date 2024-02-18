import argparse
import os
from enum import Enum
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from skimage import exposure, filters, measure, transform
from skimage.measure import regionprops
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import utils

# NOTE: Arguments in this source file are only for your convenience. They will not be used during
# evaluation but you can use them for easier experimentation with your solution.
parser = argparse.ArgumentParser()
parser.add_argument("--example", default=None, type=int, help="Example argument.")

class SolveEnum(Enum):
    CLASSIFIED_TEXT = 1
    MATCHED_PHRASES = 2
    ORDERED_COMMANDS = 3

class PageReader:

    def __init__(self, note : str) -> None:
        # TODO: Place any initialisation of your method in here.
        # Argument 'note' is passed to this method by the evaluator. It is a single string argument that you
        # can use for something, e.g., name of the trained/loaded model in the fit method.
        self._phrases = utils.Phrases()
        self.model_filepath_letters = "saved_model/my_model_letters"
        self.model_filepath_digits = "saved_model/my_model_digits"

    def fit(self, training : bool) -> None:
        # TODO: Place your training, model saving and model loading code in here.
        # This method will be called once before any evaluation takes place so you should set up all objects
        # required for the task solving.
        # For example, running evaluation of a complete set of tasks will be done by creating an instance
        # of this class, calling this 'fit' method and then calling 'solve' for every task.
        # >>> pr = PageReader()
        # >>> pr.fit(args.training)
        # >>> pr.solve(task1)
        # >>> pr.solve(task2) etc.
        #
        # This method should be able to train your solution on demand. That means, if the argument 'training'
        # is 'True' then you should train your classification models and use the newly trained ones. If the argument
        # is 'False' then you should load models from saved files.

        # Train new models and save them to files.
        if training:
            # Train new models.
            # In the assignment we assume that line with page number is always separated and therefore
            # we can train two separated models.
            self.trainCNN(utils.CharType.LETTER)
            self.trainCNN(utils.CharType.DIGIT)

        # Load saved models from files.
        else:
            if not os.path.exists(self.model_filepath_letters):
                raise FileNotFoundError()
            else:
                self.model_letters = tf.keras.models.load_model(self.model_filepath_letters)
            if not os.path.exists(self.model_filepath_digits):
                raise FileNotFoundError()
            else:
                self.model_digits = tf.keras.models.load_model(self.model_filepath_digits)

    def trainCNN(self, type: utils.CharType):
            #=== Choose letter or numeric dataset
            if type == utils.CharType.LETTER:
                datasetPath = "emnist_data/emnist_train_big.npz"
            else:
                datasetPath = "emnist_data/emnist_train_num.npz"

            my_data = utils.EMnistDataset(datasetPath)

            # Preprocess the data
            images = np.expand_dims(my_data.imgs, axis=-1)  # Add a channel dimension

            # Encode labels
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(my_data.labels)

            # Just shuffle it for randomness
            images, labels_encoded = shuffle(images, labels_encoded, random_state=42)

            # Build the CNN model which will be main core of this program
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Finally, we train it.
            model.fit(images, labels_encoded, epochs=15)

            # Save the model to a file and also to this instance of PageReader.
            if type == utils.CharType.LETTER:
                model.save(self.model_filepath_letters)
                self.model_letters = model
            else:
                model.save(self.model_filepath_digits)
                self.model_digits = model


    def solve(self, pages : np.ndarray) -> Dict[str, Union[List[List[str]], List[str]]]:
        # TODO: This method should solve a single page reading task.
        # It gets a stack of page images on its input. You have to process and classify all pages
        # and return the text you extracted from each page, phrases which you matched on each page
        # (both in the input order of the pages) and the final list of movement commands in the correct order
        # according to the page numbers.

        classified_text : List[List[str]] = []
        matched_phrases : List[List[str]] = []
        ordered_commands : List[str] = []

        command_pages_dict = {} #Dict where keys are page numbers and values page lines.

        for i in range(pages.shape[0]): # iterate through pages (which is first dimension in this case)
            # Access the 2D array at index i
            current_2D_array = pages[i, :, :]
            
            img_array_normalized = exposure.rescale_intensity(current_2D_array)
            # Apply thresholding
            threshold_value = filters.threshold_otsu(img_array_normalized)
            binary_image = img_array_normalized > (threshold_value * 0.05)

            # Label connected components
            # (Below I will use Levenshtein metric to recognize phrases,
            # so handling spaces between words is not necessary here.)            
            label_image = measure.label(binary_image)

            # Get properties of labeled regions
            regions = regionprops(label_image)

            # Extract bounding boxes of regions
            bounding_boxes = [region.bbox for region in regions]

            linesWithBoxes = utils.GetLinesByBoundaryBoxes(bounding_boxes, pages.shape[1])

            classified_text.append([])
            matched_phrases.append([])     
            pages_command = [] # array of commands on single page

            for radek in range (0, len(linesWithBoxes[0])): # iterate through lines of page
                # Get bounding boxes within given line
                subarray_of_line = utils.GetBoxesByLine(bounding_boxes, linesWithBoxes[1][radek], linesWithBoxes[2][radek])

                #reading last line == line with page number
                lastLine = radek >= len(linesWithBoxes[0]) - 1

                if lastLine:
                    # In the assignment we assume that the page number consists of four digits, so for further
                    # recognization I will remove the smallest rectangles that are most likely some noise.
                    while len(subarray_of_line) > 4:
                        smallest_rectangle = min(subarray_of_line, key=utils.RectangleArea)
                        subarray_of_line.remove(smallest_rectangle)

                images_of_chars = [] # Images of single chars from the given line
                target_size = (28, 28)

                for line_character in subarray_of_line: # iterate through characters of line
                    # The performance is better if I add some padding to the separated characters.
                    # The horizontal padding is greater because widths has greater variances.
                    # (For example letter M is much wider than I, but it has same height).
                    padding_horizontal = 5
                    padding_vertical = 2
                    min_row, min_col, max_row, max_col = line_character[0] - padding_vertical, line_character[1] - padding_horizontal, line_character[2] + padding_vertical, line_character[3] + padding_horizontal
                    
                    # Get image of the character from the original input image by boundary box
                    subimage = img_array_normalized[min_row:max_row + 1, min_col:max_col + 1]

                    # Zoom it to the size 28x28 beacuse model expects this size
                    scale_factors = (
                        target_size[0] / subimage.shape[0],
                        target_size[1] / subimage.shape[1]
                    )
                    images_of_chars.append(zoom(subimage, scale_factors, order=3))

                images_of_chars_as_ND = np.stack(images_of_chars) # convert list to np.ndarray to pass it to the model

                if not lastLine: # === Line with command
                    predictions = self.model_letters.predict(images_of_chars_as_ND)
                    recognized_text = utils.ProbabilitiesToText(utils.CharType.LETTER, predictions)
                    command = utils.MostSimilarPhraseLevenshtein(recognized_text, self._phrases.phrasesToList())

                    classified_text[i].append(command)
                    matched_phrases[i].append(command)
                    pages_command.append(self._phrases.toCommand(command))

                else: # Last line with page number === last iteration
                    predictions = self.model_digits.predict(images_of_chars_as_ND)
                    recognized_text = utils.ProbabilitiesToText(utils.CharType.DIGIT, predictions)

                    classified_text[i].append(recognized_text)
                    command_pages_dict[int(recognized_text)] = pages_command

        # Order the command sequences by page number and flatten it into a single list
        ordered_commands = [word for page_number in sorted(command_pages_dict.keys()) for word in command_pages_dict[page_number]]

        return {
            # The direct result of character classification including the page numbers. One string for each line.
            SolveEnum.CLASSIFIED_TEXT : classified_text,
            # The result of matching classified lines of text onto phrases (no numbers). One string per line.
            SolveEnum.MATCHED_PHRASES : matched_phrases,
            # The final concatenated result of phrases translated into commands ordered according to the page numbers.
            SolveEnum.ORDERED_COMMANDS : ordered_commands,
        }

def main(args : argparse.Namespace) -> None:
    # NOTE: You can run any test or visualisation that you want here or anywhere else.
    # However, you must not change the signature of 'PageReader.__init__', 'PageReader.fit' or 'PageReader.solve'.
    #
    # Your solution will be evaluated using 'evaluator.py' as it was given to you. This means
    # that you should not change anything in 'evaluator.py'. Also, you should make sure that
    # your solution can be evaluated with on-demand training.
    #
    # Evaluation of your solution through the commandline can look like this:
    # >>> python .\evaluator.py --type=single --set=python_train --name=005 --verbose=2 --note=something # toto mi fungovalo at na 007
    # >>> python .\evaluator.py --type=full --set=python_train --verbose=1 --training
    # >>> python .\evaluator.py --type=full --set=python_validation --verbose=1

    # You can load maze run files using the following code:
    mr_path = "page_data/python_validation/001.npz"
    maze_run = utils.MazeRunLoader.fromFile(mr_path)
    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
