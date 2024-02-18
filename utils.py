from __future__ import annotations

import os
import sys
from enum import Enum
from typing import Sequence, List

import numpy as np



class EMnistDataset:
    """
    Loads the MNIST data saved in .npy or .npz files.

    If the 'labels' argument is left as None then the class assumes that the file
    in 'data' is .npz and creates attributes, with the same name as specified
    during the file creation, containing the respective numpy arrays.

    If the 'labels' argument is set to a string path then the class assumes that
    the files were saved as .npy and it will create two attributes: 'imgs' which
    contains the contents of the 'data' file and 'labels' with the contents of
    the 'labels' file.

    If you chose to save the arrays differently then you might have to modify
    this class or write your own loader.
    """

    def __init__(self, data : str = "emnist_train.npz", labels : str = None):

        if not os.path.exists(data):
            raise ValueError("Requested mnist data file not found!")
        if (labels is not None) and (not os.path.exists(labels)):
            raise ValueError("Requested mnist label file not found!")

        if labels is None:
            dataset = np.load(data)
            for key, value in dataset.items():
                setattr(self, key, value)
        else:
            self.imgs = np.load(data)
            self.labels = np.load(labels)

class Phrases:

    def __init__(self) -> None:
        self.phrases = [
            "MOVE FORWARD", "GO FORWARD", "GO STRAIGHT", "ROLL FORWARD", "STUMBLE ON", "SHAMBLE ON", "WALK FORWARD",
            "TURN LEFT", "WHEEL LEFT", "ROTATE LEFT", "SPIN LEFT",
            "TURN RIGHT", "WHEEL RIGHT", "ROTATE RIGHT", "SPIN RIGHT",
            "TURN BACK", "WHEEL BACK", "ROTATE BACK", "SPIN BACK", "TURN AROUND", "WHEEL AROUND",
            "MOVE LEFT", "WALK LEFT", "SHAMBLE LEFT", "GO LEFTISH", "STUMBLE LEFT", "SKULK LEFT",
            "MOVE RIGHT", "WALK RIGHT", "ROLL RIGHT", "GO RIGHTISH", "SKULK RIGHT",
            "MOVE BACK", "WALK BACK", "SHAMBLE BACK", "GO BACKWARD", "STUMBLE BACK", "SKULK BACK",
        ]
        self.commands = ["F", "L", "R", "B", "ML", "MR", "MB"]
        self._phrase_to_command_list = ["F"] * 7 + ["L"] * 4 + ["R"] * 4 + ["B"] * 6 + ["ML"] * 6 + ["MR"] * 5 + ["MB"] * 6
        self.phrase_to_command = {}
        self.command_to_phrase = {}
        for phrase, command in zip(self.phrases, self._phrase_to_command_list):
            self.phrase_to_command[phrase] = command
            self.command_to_phrase[command] = phrase

    def toCommand(self, phrase : str) -> str:
        return self.phrase_to_command[phrase]
    
    def toPhrase(self, command : str) -> str:
        return self.command_to_phrase[command]
    
    def phrasesToList(self):
        """
        Returns all possible phrases as a simple list.

        Returns:
        List of possible phrases.
        """

        return self.phrases

class MazeRunLoader:

    def __init__(self, maze : np.ndarray, pages : np.ndarray, text : Sequence[Sequence[str]], numbers : List[str], path : List[str]) -> None:
        self.maze = maze
        self.pages = pages
        self.text = text
        self.numbers = numbers
        self.path = path
        self.complete_text = self._completePageText()

    @classmethod
    def fromFile(cls, file_name : str) -> MazeRunLoader:
        with np.load(file_name, allow_pickle=True) as maze_run_handle:
            maze_run_dict = dict(maze_run_handle)
        maze_run = cls(maze_run_dict["maze"], maze_run_dict["pages"], [list(page_text) for page_text in maze_run_dict["text"]], list(maze_run_dict["numbers"]), list(maze_run_dict["path"]))
        return maze_run
    
    def _completePageText(self) -> List[List[str]]:
        """Adds numbers to the page text for per-character evaluation."""
        text = []
        for page_text, number in zip(self.text, self.numbers):
            text.append(list(page_text))
            text[-1].append(number)
        return text


class CharType(Enum):
    LETTER = 0
    DIGIT = 1

class CharacterDecoder:
    "Class to handle working with the emnist mapping file."

    def __init__(self, path: str = "emnist_mapping.txt"):
        self.hodnoty = {}
        with open(path, 'r') as file:
            for line in file:
                key, value = map(int, line.split())
                self.hodnoty[key] = value
        self.digit_offset = 10 #first 10 lines of the files are digits
        print(self.hodnoty)

    def getChar(self, type: CharType, index: int):
        """
        Finds correct string of recognized phrase using probabilities
        of each character.

        Parameters:
        - type: Determines whether we need to recognice alpha or numeric char.
                    (In the assigment we assume always separated line with page
                    number, thus we won't be ever recognizing alphanumeric string.)
        - index: Index of the character.

        Returns:
        Decoded character as string.
        """

        if type == CharType.DIGIT and index >= 0 and index <= 9:
            return str(chr(self.hodnoty[index]))
        elif type == CharType.LETTER and index >= 0 and index <= 51:
            return str(chr(self.hodnoty[index + self.digit_offset]))
        return ""
    
    
def ProbabilitiesToText(type: CharType, probs: np.ndarray):
    """
    Finds correct string of recognized phrase using probabilities
    of each character.

    Parameters:
    - type: Determines whether we need to recognice alpha or numeric text.
                (In the assigment we assume always separated line with page
                number, thus we won't be ever recognizing alphanumeric string.)
    - probs: Array with probabilites for each character at every position
                of recognized phrase.

    Returns:
    Recognized string.
    """

    result = ""
    cd = CharacterDecoder()
    for prob in probs:
        max_index = np.argmax(prob)
        result += cd.getChar(type, max_index)
    return result


def BoxIntersect(arrayOfBoxes, line):
    """
    Determines, whether line (given by Y coordinate) intersects
    any of the provided bounding boxes.

    Parameters:
    - arrayOfBoxes: Array of bounding boxes.
    - line: Line Y coordinate.

    Returns:
    Boolean: Line intersects any box.
    """

    for box in arrayOfBoxes:
        if box[0] <= line and box[2] >= line:
            return True
    return False

def GetLinesByBoundaryBoxes(arrayOfBoxes, imageHeight):
    """
    Finds coordinates of all lines from given set of boundary boxes.

    By line I mean space between two horizontal borders occupied
    by any of the given boundary boxes.

    We assume, that the lines don't intersect and there is also
    horizontal border of size at least 1 between the lines.

    Parameters:
    - arrayOfBoxes: Array of bounding boxes.
    - line: Pixel height of input image.

    Returns:
    List of three lists:
        - List with line indices (I provided here for testing purposes; not necessary otherwise).
        - List with Y coordinates of lower borders of the lines
        - List with Y coordinates of upper borders of the lines
    """
    readingLine = False
    readingLineChange = False

    tmpNrOfLines = 0
    line_indices = []
    minimum_lines_y = []
    maximum_lines_y = []

    for i in range(0, imageHeight):
        if BoxIntersect(arrayOfBoxes, i):
            if not readingLine:
                readingLineChange = True
            else:
                readingLineChange = False
            readingLine = True
            if readingLineChange:
                line_indices.append(tmpNrOfLines)
                minimum_lines_y.append(i)
                tmpNrOfLines += 1
        else:
            if readingLine:
                readingLineChange = True
            else:
                readingLineChange = False
            readingLine = False
            if (readingLineChange):
                maximum_lines_y.append(i)
            
    return [line_indices, minimum_lines_y, maximum_lines_y]


def RectangleArea(rect):
    """
    Calculates area of rectangle.

    Parameters:
    - rect (quadriple): Rectangle given by min_X, min_Y, max_X, max_Y.

    Returns:
    Rectangle area.
    """

    min_x, min_y, max_x, max_y = rect
    return (max_x - min_x) * (max_y - min_y)

def GetBoxesByLine(arrayOfBoxes, min_y, max_y):
    """
    Finds all bounding boxes within given boundaries.

    Parameters:
    - arrayOfBoxes: Array of bounding boxes.
    - min_Y: Y coordinate of lower boundary.
    - max_Y: Y coordinate of upper boundary.

    Returns:
    Bounding boxes within given horizontal lines.
    """

    boxes_result = []
    for box in arrayOfBoxes:
        if box[0] >= min_y and box[2] <= max_y:
            boxes_result.append(box)
    return sorted(boxes_result, key=lambda x: x[1])


def LevenshteinDistance(str1: str, str2: str):
    """
    Calculates Levenshtein (minimum edit) distance of two strings.

    Parameters:
    - str1 (str): String 1 to compare.
    - str2 (str): String 2 to compare.

    Returns:
    Levenshtein distance as integer.
    """

    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # Matrix for distances
    matrix = [[0 for n in range(len_str2)] for m in range(len_str1)]

    # Init matrix with values corresponding to the lengths of substrings
    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    # Populate the matrix using dynamic programming
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost  # Substitution
            )

    # Return result Levenshtein
    return matrix[len_str1 - 1][len_str2 - 1]


def MostSimilarPhraseLevenshtein(input_phrase, phrase_list):
    """
    For given phrase finds most similar phrase from provided list 
    using Levenshtein metrics.

    Parameters:
    - input_phrase (str): String with phrase to classify.
    - phrase_list (List): List of possible phrases.

    Returns:
    Most similar phrase.
    """

    min_distance = sys.maxsize
    most_similar = None

    for phrase in phrase_list:
        distance = LevenshteinDistance(input_phrase, phrase)
        if distance < min_distance:
            min_distance = distance
            most_similar = phrase

    return most_similar


