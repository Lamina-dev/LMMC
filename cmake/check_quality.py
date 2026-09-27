#!/usr/bin/env python3
"""执行 LMMC 与 LMCAS 共用的自有源码质量检查。

显式扫描全部自有源码，按原始计数计算比率；零分母显示为 0.00。
归属检查覆盖翻译单元及测试支持代码，头文件由其他检查处理。
"""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr
from dataclasses import dataclass
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import sys


LIZARD_VERSION = "1.24.0"
SOURCE_ROOTS = ("include", "src", "tests", "test", "examples", "example", "benchmarks")
TEST_ROOTS = frozenset(("tests", "test", "examples", "example", "benchmarks"))
EXCLUDED_DIRS = frozenset(("build", "dist", ".git"))
IMPLEMENTATIONS = frozenset((".c", ".cc", ".cpp", ".cxx"))
HEADERS = frozenset((".h", ".hh", ".hpp", ".hxx"))
FRAGMENTS = frozenset((".inc", ".ipp", ".tpp"))
THRESHOLDS = {
    "production": {"nloc": 80, "cyclomatic_complexity": 15,
                   "parameter_count": 10, "length": 150, "max_nesting_depth": 5},
    "test": {"nloc": 60, "cyclomatic_complexity": 12,
             "parameter_count": 10, "length": 120, "max_nesting_depth": 5},
}
MEASUREMENTS = tuple(THRESHOLDS["production"]) + ("max_nested_structures", "token_count")
GENERATED_TABLE = Path("src/ziggurat_table.inc")
TABLE_PROVENANCE = "/* Generated from the historical Ziggurat recurrence; exact binary64 literals. */"
BYPASS = re.compile(
    r"#\s*lizard\s+(?:forgive\w*|whitelist\w*|ignore\w*|exclude\w*|disable\w*)"
    r"|GENERATED\s+CODE|\bwhitelizard(?:\.txt)?\b", re.IGNORECASE,
)


@dataclass(frozen=True)
class Source:
    path: Path
    relative: str
    group: str
    public: bool


def violation(report, code, path, message, **fields):
    report["violations"].append({"code": code, "path": str(path),
                                 "message": message, **fields})


def load_lizard():
    try:
        installed = metadata.version("lizard")
    except metadata.PackageNotFoundError as error:
        raise ValueError("lizard==1.24.0 is required; distribution is not installed") from error
    if installed != LIZARD_VERSION:
        raise ValueError(f"lizard=={LIZARD_VERSION} is required; found {installed}")
    import lizard
    return lizard


def read_source(path):
    text = path.read_text(encoding="utf-8-sig")
    if "\0" in text:
        raise ValueError(f"NUL byte in source: {path}")
    return text


def check_whitelist(path, report):
    if path.name.lower() == "whitelizard.txt":
        violation(report, "bypass.whitelist", path,
                  "Lizard whitelist files are forbidden, including empty whitelists")


def walk_source_root(directory, report):
    """遍历显式根目录中的真实路径，覆盖被 gitignore 忽略的源码。"""
    def on_error(error):
        violation(report, "input.directory", error.filename or directory, str(error))

    for current, dirs, files in os.walk(directory, onerror=on_error, followlinks=False):
        parent = Path(current)
        descend = []
        for name in sorted(dirs):
            if name in EXCLUDED_DIRS:
                continue
            child = parent / name
            if child.is_symlink():
                violation(report, "input.symlink", child, "Source directory aliases are forbidden")
            else:
                descend.append(name)
        dirs[:] = descend
        for name in sorted(files):
            path = parent / name
            check_whitelist(path, report)
            if path.suffix.lower() not in IMPLEMENTATIONS | HEADERS | FRAGMENTS:
                continue
            if path.is_symlink():
                violation(report, "input.symlink", path, "Source file aliases are forbidden")
                continue
            yield path


def exclude_generated(path, project, report):
    """按精确路径与来源标记验证生成表例外。"""
    if path != project / GENERATED_TABLE:
        return False
    # 将递推表例外限定于 LMMC。
    if (project / "LMMC").is_dir():
        return False
    try:
        lines = read_source(path).splitlines()
        if not lines or lines[0] != TABLE_PROVENANCE:
            raise ValueError("Generated table provenance does not match the historical recurrence")
        literal = re.compile(r"\s*0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+,\s*", re.IGNORECASE)
        if len(lines) != 258 or any(not literal.fullmatch(line) for line in lines[1:]):
            raise ValueError("Generated table must contain exactly 257 binary64 literals and no source code")
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        violation(report, "input.provenance", path, str(error))
        return False
    report["excluded_generated"].append({"path": str(path), "provenance": TABLE_PROVENANCE})
    return True


def inventory(root, report, include_generated=False):
    if not root.is_dir():
        raise ValueError(f"Source root does not exist or is not a directory: {root}")
    projects = [root]
    nested = root / "LMMC"
    if nested.exists():
        if not nested.is_dir() or nested.is_symlink():
            raise ValueError(f"Nested LMMC must be a real directory: {nested}")
        projects.append(nested)
    sources = []
    for project in projects:
        for name in ("include", "src"):
            if not (project / name).is_dir():
                violation(report, "input.directory", project / name, "Required first-party directory is missing")
        whitelist = project / "whitelizard.txt"
        if whitelist.exists():
            check_whitelist(whitelist, report)
        for name in SOURCE_ROOTS:
            directory = project / name
            if not directory.exists():
                continue
            if directory.is_symlink() or not directory.is_dir():
                violation(report, "input.directory", directory, "First-party root must be a real directory")
                continue
            for path in walk_source_root(directory, report):
                if not include_generated and exclude_generated(path, project, report):
                    continue
                sources.append(Source(path, path.relative_to(root).as_posix(),
                                      "test" if name in TEST_ROOTS else "production",
                                      name == "include" and
                                      path.relative_to(directory).parts[0] not in ("internal", "visitors")))
    if not sources:
        violation(report, "input.empty", root, "No first-party source or header files found")
    return sorted(sources, key=lambda source: source.relative)


def comment_processor(source, report):
    """检测质量抑制指令，并保留注释中的换行供 Lizard 定位。"""
    def process(tokens, reader):
        line = 1
        for token in tokens:
            comment = reader.get_comment_from_token(token)
            if comment is None:
                yield token
            else:
                for match in BYPASS.finditer(comment):
                    violation(report, "bypass.annotation", source.relative,
                              "Quality suppression annotations are forbidden",
                              line=line + comment.count("\n", 0, match.start()),
                              annotation=match.group())
                # 保留多行注释中的换行，避免压缩函数长度与诊断行号。
                for _ in comment.splitlines()[1:]:
                    yield "\n"
            line += token.count("\n")
    return process


def analyze_source(source, text, lizard, report):
    # 同时加载 NS 与 ND，以获取最大嵌套深度。
    extensions = lizard.get_extensions(["NS", "ND"])
    extensions[extensions.index(lizard.comment_counter)] = comment_processor(source, report)
    analyzer = lizard.FileAnalyzer(extensions)
    diagnostics = io.StringIO()
    # 将分析器诊断视为失败，保证结果完整。
    with redirect_stderr(diagnostics):
        result = analyzer.analyze_source_code(str(source.path), text)
    if diagnostics.getvalue():
        raise ValueError(diagnostics.getvalue().strip())
    functions = []
    for function in result.function_list:
        measured = {field: getattr(function, field) for field in MEASUREMENTS}
        if any(type(value) is not int or value < 0 for value in measured.values()):
            raise ValueError(f"Invalid Lizard measurements for {function.name}")
        functions.append({"name": function.name, "long_name": function.long_name,
                          "start_line": function.start_line, "end_line": function.end_line,
                          **measured})
    return functions


def check_complexity(source, functions, report):
    limits = THRESHOLDS[source.group]
    for function in functions:
        exceeded = []
        for metric, maximum in limits.items():
            if function[metric] > maximum:
                exceeded.append(metric)
                violation(report, "complexity." + metric, source.relative,
                          f"{function['name']}: {metric} exceeds {maximum}",
                          line=function["start_line"], measured=function[metric],
                          maximum=maximum, function=function["name"])
        function["violated_metrics"] = exceeded
        function["warning"] = bool(exceeded)


def structure_limit(source):
    if source.path.suffix.lower() in HEADERS | FRAGMENTS:
        return 600 if source.public else 400
    return 600 if source.group == "test" else 800


def check_structure(source, entry, report):
    maximum = structure_limit(source)
    entry["physical_line_limit"] = maximum
    if entry["physical_lines"] > maximum:
        violation(report, "structure.physical_lines", source.relative,
                  f"Physical line count exceeds {maximum}",
                  measured=entry["physical_lines"], maximum=maximum)
    count = sum(function["nloc"] > 3 for function in entry["functions"])
    entry["nontrivial_function_count"] = count
    if source.path.suffix.lower() in IMPLEMENTATIONS and count > 35:
        violation(report, "structure.nontrivial_functions", source.relative,
                  "Implementation has more than 35 nontrivial function definitions",
                  measured=count, maximum=35)


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "display": f"{numerator / denominator:.2f}" if denominator else "0.00"}


def complexity_summary(files, group):
    functions = [function for entry in files if entry["group"] == group
                 for function in entry["functions"]]
    warned = [function for function in functions if function["warning"]]
    warning_nloc = sum(function["nloc"] for function in warned)
    nloc = sum(function["nloc"] for function in functions)
    return {"function_count": len(functions), "warning_count": len(warned),
            "warning_function_nloc": warning_nloc, "nloc_in_functions": nloc,
            "fun_rt": ratio(len(warned), len(functions)),
            "nloc_rt": ratio(warning_nloc, nloc)}


def unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path):
    with path.open(encoding="utf-8-sig") as stream:
        manifest = json.load(stream, object_pairs_hook=unique_json_keys)
    if not isinstance(manifest, dict) or set(manifest) != {"sources"}:
        raise ValueError("Manifest must be an object with exactly the 'sources' key")
    if not isinstance(manifest["sources"], list):
        raise ValueError("Manifest 'sources' must be an array")
    return manifest["sources"]


def manifest_owners(root, manifest, report):
    owners = defaultdict(list)
    for index, entry in enumerate(manifest):
        if (not isinstance(entry, dict) or set(entry) != {"path", "owner"}
                or any(not isinstance(value, str) or not value.strip() or "\0" in value
                       for value in entry.values())):
            violation(report, "input.manifest", "manifest", "Expected nonempty path/owner strings",
                      entry=index)
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve(strict=True)
            if not path.is_file():
                raise ValueError("Manifest source is not a regular file")
        except (OSError, ValueError, RuntimeError) as error:
            violation(report, "input.manifest", entry["path"], str(error), entry=index)
            continue
        owners[path].append(entry["owner"])
    return owners


def check_ownership(root, sources, manifest_paths, report):
    entries = [entry for path in manifest_paths for entry in load_manifest(path)]
    owners = manifest_owners(root, entries, report)
    expected = {source.path.resolve(): source.relative for source in sources
                if source.path.suffix.lower() in IMPLEMENTATIONS}
    for path, relative in expected.items():
        assigned = owners.get(path, [])
        if not assigned:
            violation(report, "ownership.absent", relative,
                      "Translation unit has no actual CMake target owner")
        elif len(assigned) != 1:
            violation(report, "ownership.duplicate", relative,
                      "Translation unit must belong to exactly one target", owners=assigned)
    for path, assigned in owners.items():
        if path not in expected:
            violation(report, "ownership.unexpected", path,
                      "Manifest source is outside the recursive first-party translation-unit inventory",
                      owners=assigned)
    report["ownership"] = {"inventory_count": len(expected),
                           "manifest_entry_count": sum(map(len, owners.values())),
                           "sources": [{"path": relative, "owners": owners.get(path, [])}
                                       for path, relative in expected.items()]}


def inspect_sources(sources, lizard, report):
    for source in sources:
        try:
            text = read_source(source.path)
            functions = analyze_source(source, text, lizard, report)
            entry = {"path": source.relative, "group": source.group,
                     "physical_lines": len(text.splitlines()), "functions": functions}
            report["files"].append(entry)
            if report["mode"] == "complexity":
                check_complexity(source, functions, report)
            elif report["mode"] == "structure":
                check_structure(source, entry, report)
        except (OSError, UnicodeError, ValueError, AttributeError, IndexError, RuntimeError) as error:
            violation(report, "input.analysis", source.relative, str(error))
    if report["mode"] == "complexity":
        report["complexity"] = {group: complexity_summary(report["files"], group)
                                for group in THRESHOLDS}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("complexity", "structure", "ownership", "integers", "naming"))
    parser.add_argument("--manifest", type=Path, action="append",
                        help="Actual target source manifest; repeat for independently configured consumers")
    parser.add_argument("--report", "--output", dest="report", type=Path,
                        help="Write complete JSON diagnostics (parent must exist)")
    args = parser.parse_args(argv)
    if (args.mode == "ownership") != (args.manifest is not None):
        parser.error("--manifest is required only for --mode ownership")
    return args


def emit_report(report, destination):
    counts = Counter(item["code"] for item in report["violations"])
    report["violation_count"] = len(report["violations"])
    report["violation_counts"] = dict(sorted(counts.items()))
    report["passed"] = not report["violations"]
    if destination is not None:
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for item in report["violations"]:
        location = item["path"] + (f":{item['line']}" if "line" in item else "")
        print(f"{location}: {item['code']}: {item['message']}", file=sys.stderr)
    for group, summary in report.get("complexity", {}).items():
        print(f"{group}: {summary['function_count']} functions, {summary['warning_count']} warnings; "
              f"Fun Rt={summary['fun_rt']['display']} "
              f"({summary['fun_rt']['numerator']}/{summary['fun_rt']['denominator']}), "
              f"NLOC Rt={summary['nloc_rt']['display']} "
              f"({summary['nloc_rt']['numerator']}/{summary['nloc_rt']['denominator']})")
    print(f"{report['mode']}: {len(report['files'])} files, {report['violation_count']} violations")
    if any(code.startswith("input.") for code in counts):
        return 2
    return 0 if report["passed"] else 1


def main(argv=None):
    args = parse_args(argv)
    report = {"schema_version": 1, "root": str(args.root), "mode": args.mode,
              "lizard_version": LIZARD_VERSION, "extensions": ["NS", "ND"],
              "thresholds": THRESHOLDS,
              "nontrivial_definition": "Lizard function nloc > 3",
              "zero_denominator_policy": "Retain 0/0 counts; display 0.00; no functions means no warnings",
              "files": [], "violations": [], "excluded_generated": []}
    try:
        root = args.root.resolve(strict=True)
        report["root"] = str(root)
        sources = inventory(root, report, include_generated=args.mode == "integers")
        if args.report is not None:
            destination = args.report.resolve()
            protected = {source.path.resolve() for source in sources}
            protected.update(Path(entry["path"]).resolve() for entry in report["excluded_generated"])
            protected.add(Path(__file__).resolve())
            protected.add(Path(__file__).with_name("check_naming.py").resolve())
            if args.mode in ("integers", "naming"):
                protected.add(Path(__file__).with_name("check_integers.py").resolve())
            if args.mode == "naming":
                project = root / "LMMC" if (root / "LMMC").is_dir() else root
                protected.add((project / "cmake" / "lmmc_public_symbols.txt").resolve())
            if args.mode == "integers":
                protected.add((root / "cmake" / "integer_semantics.json").resolve())
                if (root / "LMMC").is_dir():
                    protected.add((root / "LMMC" / "cmake" / "integer_semantics.json").resolve())
            if args.manifest is not None:
                protected.update(path.resolve() for path in args.manifest)
            if destination in protected:
                raise ValueError("--report must not overwrite a source, checker, or manifest input")
        if args.mode == "integers":
            from check_integers import check_integers
            check_integers(root, sources, report, read_source, violation, unique_json_keys)
        elif args.mode == "naming":
            from check_naming import check_naming
            check_naming(root, sources, report, read_source, violation)
        else:
            lizard = load_lizard()
            if args.mode == "ownership":
                report["files"] = [{"path": source.relative, "group": source.group}
                                   for source in sources]
                check_ownership(root, sources, args.manifest, report)
            else:
                inspect_sources(sources, lizard, report)
    except (OSError, UnicodeError, ValueError, ImportError, RuntimeError) as error:
        violation(report, "input.failure", args.root, str(error))
        # 输出路径校验失败时保留输入文件。
        if str(error).startswith("--report must not overwrite"):
            args.report = None
    try:
        return emit_report(report, args.report)
    except (OSError, ValueError) as error:
        print(f"quality report error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
