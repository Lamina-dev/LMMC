"""检查整数词法边界与项目语义审查记录。"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
from importlib import metadata
import json
from pathlib import PurePosixPath
import re


PYGMENTS_VERSION = "2.21.0"
CATEGORIES = frozenset(("machine", "bounded-kernel"))
REVIEW_FIELDS = frozenset(("fingerprint", "path", "normalized_line", "category", "rationale"))
INTEGER_TYPES = frozenset(("int64_t", "uint64_t"))
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
INTEGER_SUFFIX = re.compile(r"(?:u?ll|llu)$", re.IGNORECASE)
LONG_LONG_ABI = frozenset((
    "long long lmmc_debug_leaks_get_count(void);",
    "long long lmmc_debug_leaks_get_count(void) {",
))
SPLICE = re.compile(r"\\\n")
RAW_PREFIXES = frozenset(("R", "LR", "uR", "UR", "u8R"))


def load_lexer():
    try:
        installed = metadata.version("pygments")
    except metadata.PackageNotFoundError as error:
        raise ValueError(f"pygments=={PYGMENTS_VERSION} is required; distribution is not installed") from error
    if installed != PYGMENTS_VERSION:
        raise ValueError(f"pygments=={PYGMENTS_VERSION} is required; found {installed}")
    from pygments.lexer import bygroups, include, using, this
    from pygments.lexers.c_cpp import CFamilyLexer, CppLexer
    from pygments.token import Comment, Number, Punctuation, String

    class IntegerLexer(CppLexer):
        # 移除四条预处理入口规则，使宏与条件分支参与词法分析。
        # 保留注释和字符串规则，使用平坦根规则稳定函数签名分词。
        tokens = {
            "whitespace": CFamilyLexer.tokens["whitespace"][4:],
            "root": [
                (r"(^[ \t]*#[ \t]*(?:include|include_next|import)\b" +
                 CFamilyLexer._possible_comments + r")(<[^>\n]+>)",
                 bygroups(using(this), String)),
                (r"(?:0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*|"
                 r"0[bB][01](?:'?[01])*|[0-9](?:'?[0-9])*)"
                 r"(?:[uU]?[lL]{2}|[lL]{2}[uU]?)(?![\w.'])", Number.Integer),
                include("whitespace"), include("statements"),
                (r"[{};#\\]", Punctuation),
            ],
        }

    return IntegerLexer(), Comment, String, Number


def splice_source(text):
    """执行第二阶段续行拼接，保留诊断所需的原始偏移。"""
    pieces = []
    offsets = []
    start = 0
    for match in SPLICE.finditer(text):
        pieces.append(text[start:match.start()])
        offsets.extend(range(start, match.start()))
        start = match.end()
    pieces.append(text[start:])
    offsets.extend(range(start, len(text)))
    return "".join(pieces), offsets


def raw_literal_end(text, logical, offsets, start):
    # 在原始字符串中恢复续行，以定位真实结束符。
    quote = logical.index('"', start)
    opening = logical.index("(", quote)
    delimiter = text[offsets[quote] + 1:offsets[opening]]
    closing = ")" + delimiter + '"'
    end = text.find(closing, offsets[opening] + 1)
    if end < 0:
        raise ValueError("Unterminated raw string literal after restoring line continuations")
    return bisect_left(offsets, end + len(closing))


def translated_tokens(text, lexer, string):
    logical, offsets = splice_source(text)
    start = 0
    while start < len(logical):
        # 补齐末尾换行，满足词法器的行注释规则。
        for offset, kind, value in lexer.get_tokens_unprocessed(logical[start:] + "\n"):
            offset += start
            if offset >= len(offsets):
                continue
            if kind is string.Affix and value in RAW_PREFIXES:
                start = raw_literal_end(text, logical, offsets, offset)
                break
            yield offsets[offset], kind, value
        else:
            break


def source_tokens(text, lexer, comment, string):
    """提取有效词元与规范化行文本，保持行移动后的内容标识稳定。"""
    line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
    tokens = []
    lines = defaultdict(list)
    for original, kind, value in translated_tokens(text, lexer, string):
        if kind in comment or kind in string or not value.strip():
            continue
        line = bisect_right(line_starts, original)
        token = {"value": value, "kind": kind, "line": line,
                 "column": original - line_starts[line - 1] + 1}
        tokens.append(token)
        lines[line].append(value)
    normalized = {line: json.dumps(values, ensure_ascii=True, separators=(",", ":"))
                  for line, values in lines.items()}
    return tokens, normalized


def fingerprint(path, normalized_line):
    return hashlib.sha256((path + "\n" + normalized_line).encode("utf-8")).hexdigest()


def lexical_violations(tokens, number, path, source_lines, report, violation):
    long_count = 0
    type_words = frozenset(("long", "int", "unsigned", "signed", "const", "volatile"))
    for token in tokens:
        value = token["value"]
        fields = {"line": token["line"], "column": token["column"],
                  "source": source_lines[token["line"] - 1], "token": value}
        if value not in type_words:
            long_count = 0
        elif value == "long":
            long_count += 1
            if (long_count >= 2 and
                    source_lines[token["line"] - 1].strip() not in LONG_LONG_ABI):
                violation(report, "integers.forbidden_type", path,
                          "Use BigInt or an explicitly bounded machine type, not long long",
                          **fields)
        if token["kind"] in number and INTEGER_SUFFIX.search(value):
            violation(report, "integers.forbidden_suffix", path,
                      "LL/ULL/LLU integer suffixes (all case variants) are forbidden", **fields)
        for match in IDENTIFIER.finditer(value):
            word = match.group()
            if word.startswith("LLONG_") or word in ("stoll", "stoull", "llabs"):
                violation(report, "integers.forbidden_identifier", path,
                          f"Forbidden integer identifier: {word}", **fields)


def integer_records(tokens, normalized, source, project, root, source_lines):
    path = source.path.relative_to(project).as_posix()
    records = []
    for token in tokens:
        for match in IDENTIFIER.finditer(token["value"]):
            if match.group() not in INTEGER_TYPES:
                continue
            line = token["line"]
            normalized_line = normalized[line]
            records.append({"project": project.relative_to(root).as_posix(),
                            "path": path, "inventory_path": source.relative,
                            "line": line, "column": token["column"] + match.start(),
                            "type": match.group(), "source": source_lines[line - 1],
                            "context": [{"line": index + 1, "source": source_lines[index]}
                                        for index in range(max(0, line - 2), min(len(source_lines), line + 1))],
                            "normalized_line": normalized_line,
                            "fingerprint": fingerprint(path, normalized_line),
                            "review_status": "unreviewed"})
    return records


def manifest_entries(path, report, violation, unique_json_keys):
    try:
        if path.is_symlink():
            raise ValueError("Semantic review manifest must not be a symlink")
        with path.open(encoding="utf-8-sig") as stream:
            manifest = json.load(stream, object_pairs_hook=unique_json_keys)
        if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "reviews"}:
            raise ValueError("Expected exactly schema_version and reviews keys")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("Semantic review schema_version must be integer 1")
        if not isinstance(manifest["reviews"], list):
            raise ValueError("Semantic reviews must be an array")
        return manifest["reviews"]
    except FileNotFoundError:
        violation(report, "integers.manifest_missing", path,
                  "Missing checked-in semantic review manifest; the complete inventory is still reported")
    except (OSError, UnicodeError, ValueError) as error:
        violation(report, "integers.manifest_malformed", path, str(error))
    return []


def validate_review(entry):
    if not isinstance(entry, dict) or set(entry) != REVIEW_FIELDS:
        return "Review must contain exactly fingerprint, path, normalized_line, category and rationale"
    if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
        return "Every review field must be a nonempty string"
    path = PurePosixPath(entry["path"])
    if path.is_absolute() or ".." in path.parts or ":" in entry["path"] or "\\" in entry["path"]:
        return "Review path must be project-relative POSIX syntax without parent traversal"
    if path.as_posix() != entry["path"]:
        return "Review path must be normalized project-relative POSIX syntax"
    if entry["category"] not in CATEGORIES:
        return "Review category must be machine or bounded-kernel"
    rationale = entry["rationale"].strip().lower()
    if rationale in CATEGORIES or re.search(r"\b(?:todo|tbd|placeholder|unreviewed)\b", rationale):
        return "Review rationale must specifically explain the machine role or established kernel bounds"
    if not re.fullmatch(r"[0-9a-f]{64}", entry["fingerprint"]):
        return "Review fingerprint must be a lowercase SHA-256 hex digest"
    if entry["fingerprint"] != fingerprint(entry["path"], entry["normalized_line"]):
        return "Review fingerprint does not match its path and normalized_line"
    return None


def reviewed_entries(entries, current, manifest, report, violation):
    reviews = {}
    seen = set()
    duplicates = set()
    for index, entry in enumerate(entries):
        problem = validate_review(entry)
        if problem:
            violation(report, "integers.manifest_malformed", manifest, problem, entry_index=index)
            continue
        key = entry["fingerprint"]
        if key in seen:
            duplicates.add(key)
            violation(report, "integers.manifest_duplicate", manifest,
                      "Each fingerprint must have exactly one review", fingerprint=key, entry_index=index)
        seen.add(key)
        if key not in current:
            violation(report, "integers.manifest_stale", manifest,
                      "Review no longer matches any inventoried source line", fingerprint=key,
                      source_path=entry["path"], entry_index=index)
        else:
            reviews[key] = entry
    for key in duplicates:
        reviews.pop(key, None)
    return reviews


def review_project(project, records, report, violation, unique_json_keys):
    manifest = project / "cmake" / "integer_semantics.json"
    current = {record["fingerprint"] for record in records}
    entries = manifest_entries(manifest, report, violation, unique_json_keys)
    reviews = reviewed_entries(entries, current, manifest, report, violation)
    template = {}
    for record in records:
        key = record["fingerprint"]
        template[key] = {field: record[field] for field in ("fingerprint", "path", "normalized_line")}
        template[key].update(category="", rationale="")
        if key in reviews:
            record.update(review_status="reviewed", category=reviews[key]["category"],
                          rationale=reviews[key]["rationale"])
            template[key] = reviews[key]
        else:
            violation(report, "integers.unreviewed", record["inventory_path"],
                      f"{record['type']} requires an explicit project-local semantic review",
                      line=record["line"], column=record["column"], fingerprint=key,
                      manifest=str(manifest))
    ordered = sorted(template.values(), key=lambda entry: (entry["path"], entry["normalized_line"]))
    return {"manifest": str(manifest), "occurrence_count": len(records),
            "unique_line_count": len(current),
            "review_template": {"schema_version": 1, "reviews": ordered}}


def check_integers(root, sources, report, read_source, violation, unique_json_keys):
    lexer, comment, string, number = load_lexer()
    projects = [root] + ([root / "LMMC"] if (root / "LMMC").is_dir() else [])
    grouped = {project: [] for project in projects}
    report["integers"] = {
        "pygments_version": PYGMENTS_VERSION,
        "fingerprint_algorithm": "sha256(UTF-8(project-relative POSIX path + LF + normalized_line))",
        "normalized_line_format": "Compact ensure_ascii JSON array of Pygments token values, excluding whitespace, comments and strings",
        "review_scope": "One review per distinct path and normalized source line; identical lines share a review, every occurrence is reported",
        "rationale_contract": "Specific explanation of the machine role or the proof establishing bounded-kernel range; never automatic classification",
        "categories": sorted(CATEGORIES), "records": [], "projects": [],
    }
    for source in sources:
        project = projects[-1] if len(projects) == 2 and source.path.is_relative_to(projects[-1]) else root
        try:
            text = read_source(source.path)
            source_lines = text.splitlines()
            tokens, normalized = source_tokens(text, lexer, comment, string)
            lexical_violations(tokens, number, source.relative, source_lines, report, violation)
            records = integer_records(tokens, normalized, source, project, root, source_lines)
            grouped[project].extend(records)
            report["integers"]["records"].extend(records)
            report["files"].append({"path": source.relative, "group": source.group,
                                    "physical_lines": len(source_lines), "integer_occurrences": len(records)})
        except (OSError, UnicodeError, ValueError, IndexError, RuntimeError) as error:
            violation(report, "input.analysis", source.relative, str(error))
    for project, records in grouped.items():
        review = review_project(project, records, report, violation, unique_json_keys)
        review["project"] = project.relative_to(root).as_posix()
        report["integers"]["projects"].append(review)
