
from __future__ import annotations
import os
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
