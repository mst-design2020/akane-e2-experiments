import ast, sys
for fn in sys.argv[1:]:
    src = open(fn).read(); t = ast.parse(src)
    for node in ast.walk(t):
        if isinstance(node, (ast.List, ast.Set, ast.Tuple, ast.Dict)):
            elts = node.keys if isinstance(node, ast.Dict) else node.elts
            s = [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(s) >= 2:
                print(f"{fn}:{node.lineno}: {s[:8]}{' …' if len(s)>8 else ''} ({len(s)})")
