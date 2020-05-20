
import inspect
import os
import parser
import symbol
import sys

sys.path.insert(0, os.path.abspath(".."))
import recipipe


def st2str(st_list):
    result = []
    _st2str(st_list, result)
    return "".join(result)

def _st2str(st_list, result):
    if hasattr(st_list, "__iter__"):
        for i in st_list:
            if type(i) == str:
                result.append(i)
            else:
                _st2str(i, result)

def find_expr_stmt(st_list):
    if hasattr(st_list, "__iter__"):
        code = st_list[0]
        if code == symbol.expr_stmt and len(st_list) == 4:
            return [(st2str(st_list[1]), st2str(st_list[3]))]
        results = []
        for i in st_list:
            if type(i) != str:
                r = find_expr_stmt(i)
                if r:
                    results.extend(r)
        return results

code = inspect.getsource(recipipe)
st = parser.suite(code)
st_list = parser.st2list(st)
assign_expr = find_expr_stmt(st_list)

filename = os.path.join(os.path.dirname(__file__), "_generated/recipipe_aliases.csv")
with open(filename, "w") as f:
    f.write("Alias,Definition\n")
    for i, j in assign_expr:
        if i[0] != "_":
            j = j.replace('"', "'")
            o = getattr(recipipe, i)
            if inspect.isclass(o):
                # Remove modules starting from '_' to avoid wrong references
                # to the SKLearn documentation.
                m = ".".join([i for i in o.__module__.split(".") if i[0]!="_"])
                f.write(f'{i},":obj:`{m}.{o.__qualname__}`"\n')
            else:
                f.write(f'{i},"`{j}`"\n')

