# NOTE: You should not edit this source file because it is not part of your solution.
# You will hand in your solution without this file and if you submit it, it will be ignored.
# - So, any changes that break the original evaluation will look as mistakes in your solution.

import os
from pathlib import Path
import re
from typing import Sequence, Tuple
import argparse
import numpy as np
import utils
import page_reader

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="single", type=str, help="Type of evaluation; one of 'single' for evaluation of one task or 'full' for evaluation of an entire set.")
parser.add_argument("--set", default="python_train", type=str, help="Name of the evaluated set - it is a directory containing maze runs")
parser.add_argument("--name", default="001.npz", type=str, help="Name of the evaluated maze run in the selected set for type 'single'.")
parser.add_argument("--training", default=False, action="store_true", help="Calls training routine before evaluation.")
parser.add_argument("--verbose", default=2, type=int, help="Level of verbosity: '0' - prints only the final result, '1' - prints every maze run result, '2' - prints every page result.")
parser.add_argument("--note", default=None, type=str, help="String note passed to the solution in its constructor.")
        
class Evaluator:
    """Provides functions for maze run evaluation."""

    def __init__(self, args : argparse.Namespace) -> None:
        self.args = args
        self.solution = page_reader.PageReader(self.args.note)
        self.point_idx = { "text" : 0, "phrase" : 1, "number" : 2, "path" : 3 }

    def _compareLists(self, first_list : Sequence[object], second_list : Sequence[object]) -> int:
        points = 0
        for i in range(min(len(first_list), len(second_list))):
            if first_list[i] == second_list[i]:
                points += 1
        return points

    def _compareCharacters(self, detected_text : Sequence[str], true_text : Sequence[str]) -> Tuple[int, int]:
        max_points = max(np.sum([len(t) for t in detected_text]), np.sum([len(t) for t in true_text]))
        points = 0
        for i in range(min(len(detected_text), len(true_text))):
            points += self._compareLists(detected_text[i], true_text[i])
        return points, max_points

    def _compareWords(self, detected_words : Sequence[str], true_words : Sequence[str]) -> Tuple[int, int]:
        max_points = max(len(detected_words), len(true_words))
        points = self._compareLists(detected_words, true_words)
        return points, max_points

    def _printResuls(self, name : str, points : np.ndarray, max_points : np.ndarray) -> None:
        print("{} per-character accuracy: {:.2f}%".format(name, points[self.point_idx["text"]] / max_points[self.point_idx["text"]] * 100))
        print("{} per-phrase accuracy:    {:.2f}%".format(name, points[self.point_idx["phrase"]] / max_points[self.point_idx["phrase"]] * 100))
        print("{} page number accuracy:   {:.2f}%".format(name, points[self.point_idx["number"]] / max_points[self.point_idx["number"]] * 100))
        print("{} path (final) accuracy:  {:.2f}%".format(name, points[self.point_idx["path"]] / max_points[self.point_idx["path"]] * 100))

    def _evaluateRun(self, maze_run : utils.MazeRunLoader, name : str):
        """Evaluates one loaded maze run."""
        detected_dict = self.solution.solve(maze_run.pages)
        detected_text, detected_phrases, detected_path = detected_dict[page_reader.SolveEnum.CLASSIFIED_TEXT], detected_dict[page_reader.SolveEnum.MATCHED_PHRASES], detected_dict[page_reader.SolveEnum.ORDERED_COMMANDS]
        # Score counting.
        points, max_points = np.zeros((4), dtype=int), np.zeros((4), dtype=int)
        max_points[self.point_idx["number"]] = len(maze_run.complete_text)
        # Process each page separately.
        for p_detected_text, p_detected_phrases, p_true_text, p_true_phrases in zip(detected_text, detected_phrases, maze_run.complete_text, maze_run.text):
            (text_points, max_text_points), (phrase_points, max_phrase_points) = self._compareCharacters(p_detected_text, p_true_text), self._compareWords(p_detected_phrases, p_true_phrases)
            points[self.point_idx["text"]], max_points[self.point_idx["text"]] = points[self.point_idx["text"]] + text_points, max_points[self.point_idx["text"]] + max_text_points
            points[self.point_idx["phrase"]], max_points[self.point_idx["phrase"]] = points[self.point_idx["phrase"]] + phrase_points, max_points[self.point_idx["phrase"]] + max_phrase_points
            points[self.point_idx["number"]] += p_detected_text[-1] == p_true_text[-1]
            if self.args.verbose > 1:
                print("Page '{}' statistics".format(p_true_text[-1]))
                print("Per-character page accuracy: {:.2f}%".format(points[self.point_idx["text"]] / max_points[self.point_idx["text"]] * 100))
                print("Per-phrase page accuracy:    {:.2f}%".format(points[self.point_idx["phrase"]] / max_points[self.point_idx["phrase"]] * 100))
                print("Detected number/True number: {}/{}".format(p_detected_text[-1], p_true_text[-1]))
        # Compute points for the final path.
        points[self.point_idx["path"]], max_points[self.point_idx["path"]] = self._compareWords(detected_path, maze_run.path)
        if self.args.verbose > 0:
            print("Maze run '{}' statistics".format(name))
            self._printResuls("Maze run", points, max_points)
        return points, max_points

    def _evaluateMazeRuns(self, name : str, maze_run_files : Sequence[str]) -> None:
        """Evaluates a set of maze run tasks."""
        maze_runs = []
        for f in maze_run_files:
            maze_runs.append(utils.MazeRunLoader.fromFile(f))
        
        total_points, total_max_points = np.zeros((4), dtype=int), np.zeros((4), dtype=int)
        for i, maze_run in enumerate(maze_runs):
            points, max_points = self._evaluateRun(maze_run, Path(maze_run_files[i]).stem)
            total_points, total_max_points = total_points + points, total_max_points + max_points
        print("Overall evaluation of {}.".format(name))
        self._printResuls("Total", total_points, total_max_points)

    def _evaluateSingle(self) -> None:
        """Evaluates one maze run with loading of file selected through arguments."""
        filename = self.args.name if self.args.name.endswith(".npz") else self.args.name + ".npz"
        evaluated_file = os.path.join("page_data", self.args.set, filename)
        self._evaluateMazeRuns("the maze run '{}'".format(filename), [evaluated_file])

    def _evaluateFull(self):
        """Evaluates all maze runs found in the directory specified through arguments."""
        evaluated_folder = os.path.join("page_data", self.args.set)
        regex = re.compile('.*\.npz$')
        maze_run_files = []
        dir_list = os.listdir(evaluated_folder)
        for f in dir_list:
            if regex.match(f):
                maze_run_files.append(f)
        maze_run_files = [os.path.join(evaluated_folder, f) for f in maze_run_files]
        self._evaluateMazeRuns("the set '{}'".format(self.args.set), maze_run_files)

    def evaluate(self):
        """Evaluates the tasks requested through arguments."""
        self.solution.fit(self.args.training)
        evaluation = {
            "single" : self._evaluateSingle,
            "full" : self._evaluateFull,
        }
        if self.args.type not in evaluation:
            raise ValueError("Unrecognised type of evaluation: '{}', please, use one of: 'single'/'full'.".format(self.args.type))
        evaluation[args.type]()

def main(args : argparse.Namespace):
    evaluator = Evaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
