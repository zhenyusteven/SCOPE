from typing import List, Dict, Any, Optional, Tuple

class Editor:
    def __init__(self, lines: Optional[str] = None):
        if lines is None:
            self.lines = []
        else:
            self.lines = lines.split("\n")

    def insert_lines(self, line_id: int, new_lines: List[str]) -> None:
        self.lines = self.lines[:line_id] + new_lines + self.lines[line_id:]

    def remove_lines(self, start: int, end: int) -> None:
        self.lines = self.lines[:start] + self.lines[end:]

    def get_line_id_of_first_subcomponent(self) -> Optional[int]:
        for i, line in enumerate(self.lines):
            if '/* SUBCOMPONENT */' in line:
                return i
        return None

    @staticmethod
    def _get_num_digits(n):
        if n == 0:
            return 1
        count = 0
        while n:
            n //= 10
            count += 1
        return count

    # Getters
    def get_lines_range(self, start: int, end: int) -> str:
        start = max(0, start)
        end = min(len(self.lines), end)
        if start >= end:
            return ""
        line_id_space = self._get_num_digits(end)
        return '\n'.join(
            map(
                lambda x: f"{x[0]:>{line_id_space}}:{x[1]}",
                zip(range(start, end), self.lines[start:end])
            )
        )

    def get_lines_radius(self, center: int, radius: int) -> str:
        return self.get_lines_range(center - radius, center + radius + 1)

    def get_all_lines(self):
        return self.get_lines_range(0, len(self.lines))

    def __repr__(self) -> str:
        return self.get_all_lines()



if __name__ == "__main__":
    code = """class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, x):
        self.items.append(x)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            raise IndexError("dequeue from empty queue")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        /* SUBCOMPONENT */
    """

    editor = Editor(code)
    print(editor)
    print("\n\n\n")
    idx = editor.get_line_id_of_first_subcomponent()
    print(idx)
    print("\n\n\n")
    editor.remove_lines(idx, idx+1)
    print(editor.get_lines_range(idx-5, idx+5))
    print("\n\n\n")
    editor.insert_lines(idx, ["-----A-----", "-----B-----", "-----C-----"])
    print(editor.get_lines_radius(idx, 5))
    print("\n\n\n")
    print(editor.get_all_lines())


