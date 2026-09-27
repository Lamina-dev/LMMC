"""检查全部自有源码文件名与公开 C 命名空间。"""

import re

from check_integers import load_lexer, source_tokens


SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
C_FUNCTION = re.compile(r"lmmc_[a-z0-9]+(?:_[a-z0-9]+)*\Z")
C_TYPE = re.compile(r"lmmc_[a-z0-9]+(?:_[a-z0-9]+)*_t\Z")
C_MACRO = re.compile(r"LMMC_[A-Z0-9]+(?:_[A-Z0-9]+)*\Z")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
CONDITIONAL_PUBLIC_FUNCTIONS = frozenset(("lmmc_debug_leaks_get_count",))



def public_declarations(tokens):
    """跳过链接块外壳，收集 C 声明与 typedef 定义体。"""
    declaration = []
    depth = 0
    body = False
    for token in tokens:
        word = token["value"]
        if word == "{" and [part["value"] for part in declaration] == ["extern"]:
            declaration.clear()
            continue
        if word == "{" and depth == 0 and declaration and declaration[-1]["value"] == ")":
            yield declaration
            declaration = []
            body = True
        if word == "{":
            depth += 1
        elif word == "}":
            if depth == 0:
                declaration.clear()
                continue
            depth -= 1
            if depth == 0 and body:
                body = False
                continue
        if body:
            continue
        declaration.append(token)
        if word == ";" and depth == 0:
            yield declaration
            declaration = []


def declaration_name(declaration):
    words = [token["value"] for token in declaration]
    if not words:
        return None
    if words[0] == "typedef":
        for index in range(2, len(words) - 1):
            if words[index - 2:index] == ["(", "*"] and words[index + 1] == ")":
                return "type", declaration[index]
        if words[-1] == ";" and IDENTIFIER.fullmatch(words[-2]):
            return "type", declaration[-2]
        raise ValueError("Cannot identify public C typedef declarator")
    # 以首个顶层调用形词元定位公开函数名，属性仅作为声明信息。
    depth = 0
    for index, word in enumerate(words):
        if word == "(":
            if depth == 0 and index and IDENTIFIER.fullmatch(words[index - 1]):
                name = words[index - 1]
                if name not in ("__attribute__", "__declspec", "_Static_assert", "static_assert"):
                    return "function", declaration[index - 1]
            depth += 1
        elif word == ")":
            depth -= 1
    return None



def check_c_header(source, text, lexer_parts, report, violation):
    lexer, comment, string, _number = lexer_parts
    tokens, _normalized = source_tokens(text, lexer, comment, string)
    directives = {token["line"] for token in tokens if token["value"] == "#"}
    for index, token in enumerate(tokens[:-2]):
        if token["value"] != "#" or tokens[index + 1]["value"] != "define":
            continue
        name = tokens[index + 2]
        if not C_MACRO.fullmatch(name["value"]):
            violation(report, "naming.c_macro", source.relative,
                      f"Public macro must use LMMC_UPPER_CASE: {name['value']}", line=name["line"])
    # 将第二阶段续行归入同一预处理指令。
    lines = text.splitlines()
    for start in tuple(directives):
        line = start
        while line <= len(lines) and lines[line - 1].endswith("\\"):
            line += 1
            directives.add(line)
    declarations = list(public_declarations(
        token for token in tokens if token["line"] not in directives))
    functions = []
    for declaration in declarations:
        named = declaration_name(declaration)
        if named is None:
            continue
        kind, token = named
        pattern = C_TYPE if kind == "type" else C_FUNCTION
        if not pattern.fullmatch(token["value"]):
            violation(report, f"naming.c_{kind}", source.relative,
                      f"Invalid public C {kind} name: {token['value']}", line=token["line"])
        if kind == "function":
            functions.append((token["value"], token))
    return functions



def check_naming(root, sources, report, read_source, violation):
    lexer_parts = load_lexer()
    declarations = {}
    for source in sources:
        report["files"].append({"path": source.relative, "group": source.group})
        if not SNAKE_CASE.fullmatch(source.path.stem):
            violation(report, "naming.file", source.relative,
                      "Source/header filename must use snake_case")
        if source.public and source.path.suffix == ".h" and source.path.parent.name == "lmmc":
            for name, token in check_c_header(
                    source, read_source(source.path), lexer_parts, report, violation):
                if name not in CONDITIONAL_PUBLIC_FUNCTIONS:
                    declarations.setdefault(name, (source.relative, token["line"]))
    project = root / "LMMC" if (root / "LMMC").is_dir() else root
    manifest = project / "cmake" / "lmmc_public_symbols.txt"
    symbols = []
    symbol_lines = {}
    for line, text in enumerate(read_source(manifest).splitlines(), 1):
        symbol = text.strip()
        if not symbol or symbol.startswith("#"):
            continue
        if not C_FUNCTION.fullmatch(symbol):
            violation(report, "naming.c_export", manifest,
                      f"Export must use lmmc_snake_case: {symbol}", line=line)
        if symbol in symbol_lines:
            violation(report, "naming.duplicate_export", manifest,
                      f"Duplicate export: {symbol}", line=line)
        symbols.append(symbol)
        symbol_lines.setdefault(symbol, line)
    if not symbols:
        violation(report, "naming.empty_exports", manifest, "Public C export manifest is empty")
    declared = set(declarations)
    exported = set(symbols)
    for symbol in sorted(declared - exported):
        path, line = declarations[symbol]
        violation(report, "naming.export_missing", path,
                  f"Public C declaration is absent from the export manifest: {symbol}",
                  line=line, symbol=symbol)
    for symbol in sorted(exported - declared):
        violation(report, "naming.export_undeclared", manifest,
                  f"Export has no unconditional public C declaration: {symbol}",
                  line=symbol_lines[symbol], symbol=symbol)
    report["naming"] = {
        "public_c_declaration_count": len(declared),
        "public_c_export_count": len(symbols),
    }
