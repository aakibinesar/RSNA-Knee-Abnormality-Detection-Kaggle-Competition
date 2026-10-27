import ast, builtins, sys
src=open(sys.argv[1],encoding="utf-8").read(); tree=ast.parse(src)
defined=set(dir(builtins))
for n in ast.walk(tree):
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)): defined.add(n.name)
    elif isinstance(n,ast.Import): defined|={(a.asname or a.name).split('.')[0] for a in n.names}
    elif isinstance(n,ast.ImportFrom): defined|={a.asname or a.name for a in n.names}
    elif isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)): defined.add(n.id)
    elif isinstance(n,ast.arg): defined.add(n.arg)
    elif isinstance(n,ast.ExceptHandler) and n.name: defined.add(n.name)
used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
u=sorted(used-defined); print("undefined names:", u or "none"); sys.exit(1 if u else 0)
