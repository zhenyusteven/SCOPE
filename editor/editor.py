from typing import List, Optional

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
    def get_lines_range(self, start: int, end: int, with_line_id: bool = True) -> str:
        start = max(0, start)
        end = min(len(self.lines), end)
        if start >= end:
            return ""
        if with_line_id:
            line_id_space = self._get_num_digits(end)
            return '\n'.join(
                map(
                    lambda x: f"{x[0]:>{line_id_space}}:{x[1]}",
                    zip(range(start, end), self.lines[start:end])
                )
            )
        else:
            return '\n'.join(self.lines[start:end])

    def get_lines_radius(self, center: int, radius: int, with_line_id: bool = True) -> str:
        return self.get_lines_range(center - radius, center + radius + 1, with_line_id)

    def get_all_lines(self, with_line_id: bool = True) -> str:
        return self.get_lines_range(0, len(self.lines), with_line_id)

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
        
    def clear(self):
        self.items = []
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


