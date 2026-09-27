#!/usr/bin/env python3
"""构建、迁移并验证 LMCAS/LMMC 安装包及 ABI。"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


REQUIRED_TOOLING = (
    "check_quality.py", "check_package.py", "check_naming.py", "check_integers.py",
    "naming_negative_fixtures.py", "QualityGates.cmake", "PackageConsumerChecks.cmake",
    "quality-requirements.txt", "integer_semantics.json", "lmmc_public_symbols.txt",
    "prepare_quality_consumers.py",
)
BUILD_FILES = frozenset((
    "CMakeCache.txt", "cmake_install.cmake", "CTestTestfile.cmake",
    "compile_commands.json", "install_manifest.txt",
    "build.ninja", "rules.ninja", ".ninja_deps", ".ninja_log",
))
DARWIN_RUNTIME_IDS = {
    "LMCAS": "@rpath/liblmcas.dylib",
    "LMMC": "@rpath/liblmmc.dylib",
    "LMMP": "@rpath/liblmmp.1.dylib",
}
FIRST_PARTY_DYLIB = re.compile(r"lib(?:lmcas|lmmc|lmmp)(?:\.[0-9]+)*\.dylib\Z")



def source_files(root):
    """枚举 Git 可见的工作区文件与嵌套依赖。"""
    repository = Path(run(["git", "-C", root, "rev-parse", "--show-toplevel"],
                          capture=True).strip()).resolve()
    if repository != root:
        raise ValueError("Source dependency is not an initialized repository: " + str(root))
    records = run(["git", "-C", root, "ls-files", "--stage", "-z"], capture=True)
    entries = {}
    for record in records.split("\0"):
        if record:
            metadata, name = record.split("\t", 1)
            mode, _, stage = metadata.split()
            if stage != "0":
                raise ValueError("Unresolved source conflict: " + str(root / name))
            entries[name] = mode
    untracked = run(["git", "-C", root, "ls-files", "--others", "--exclude-standard", "-z"],
                    capture=True)
    entries.update((name, entries.get(name, "")) for name in untracked.split("\0") if name)
    for name, mode in sorted(entries.items()):
        path = root / name
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe source snapshot path: " + str(path))
        if any(part in {".git", "CMakeFiles", "Testing", "__pycache__"}
               for part in relative.parts) or path.name in BUILD_FILES:
            continue
        if any((parent / "CMakeCache.txt").is_file()
               for parent in path.parents if parent != root and root in parent.parents):
            continue
        if path.name == "Makefile" and not mode and (path.parent / "CMakeCache.txt").is_file():
            continue
        if path.is_symlink():
            raise ValueError("Source snapshots must not depend on symlink targets: " + str(path))
        if mode == "160000" or path.is_dir():
            for child in source_files(path.resolve()):
                yield relative / child
        elif path.is_file():
            yield relative


def snapshot_source(source, destination, lmmc):
    """仅导出 Git 可见内容，生成独立源码树。"""
    if destination == source or source in destination.parents:
        raise ValueError("Snapshot destination must be outside the source tree")
    if destination.exists():
        raise ValueError("Snapshot destination must not already exist: " + str(destination))
    files = sorted(source_files(source))
    relative_lmmc = lmmc.relative_to(source)
    required = {relative_lmmc / "cmake" / name for name in REQUIRED_TOOLING}
    missing = required.difference(files)
    if missing:
        raise ValueError("Required tooling is missing or not Git-visible: " +
                         ", ".join(str(path) for path in sorted(missing)))
    destination.mkdir(parents=True)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    print("Git-visible source snapshot:", destination, "files:", len(files), flush=True)


def run(command, *, capture=False, env=None):
    command = [str(value) for value in command]
    print("Running:", shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def cache_values(build):
    values = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)", line)
        if match:
            values[match[1]] = match[2]
    return values


def enabled(value):
    return str(value).upper() in {"1", "ON", "YES", "TRUE", "Y"}


def check_install_contents(prefix):
    forbidden_names = {"LMCASConfigVersion.cmake", "LMMCConfigVersion.cmake",
                       "lmcas_version.hpp"}
    for path in prefix.rglob("*"):
        if path.name in forbidden_names or (path.parent.name == "lmmc" and path.name == "version.h"):
            raise ValueError("First-party version metadata was installed: " + str(path.relative_to(prefix)))
        relative = path.relative_to(prefix).as_posix()
        if re.search(r"(^|/)(lib)?(gtest|gmock|rapidcheck|cmocka)([/_.-]|$)",
                     relative, re.IGNORECASE):
            raise ValueError("Development-only test dependency was installed: " + relative)


def expected_symbols(lmmc, cache):
    symbols = {line for line in (lmmc / "cmake/lmmc_public_symbols.txt")
               .read_text(encoding="utf-8").splitlines() if line.startswith("lmmc_")}
    tests = cache.get("LMMC_BUILD_TESTS", cache.get("LMCAS_BUILD_LMMC_TESTS", "OFF"))
    if enabled(tests):
        symbols.add("lmmc_memory_fail_after_for_test")
        symbols.add("lmmc_quad_gauss_hermite_with_iteration_limit_for_test")
    if enabled(cache.get("LMMC_DEBUG_LEAKS", "OFF")):
        symbols.add("lmmc_debug_leaks_get_count")
    if not symbols:
        raise ValueError("The public symbol manifest is empty")
    return symbols


def pe_exports(library, cache):
    text = run([cache["CMAKE_OBJDUMP"], "-p", library], capture=True)
    marker = "[Ordinal/Name Pointer] Table"
    if marker not in text:
        raise ValueError("PE export name table is absent: " + str(library))
    table = text.split(marker, 1)[1].split("\n\n", 1)[0]
    names = set()
    pattern = r"^\s*\[\s*\d+\]\s+(?:\+base\[\s*\d+\]\s+[0-9a-fA-F]+\s+)?(\S+)\s*$"
    for line in table.splitlines():
        match = re.match(pattern, line)
        if match:
            names.add(match[1])
    if not names:
        raise ValueError("PE export name table could not be decoded")
    return names


def nm_symbols(library, cache, shared):
    if cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Darwin":
        flags = ["-g", "-U", "-P"]
    else:
        flags = ["-g", "--defined-only", "-P"]
        if shared:
            flags.insert(0, "-D")
    text = run([cache["CMAKE_NM"], *flags, library], capture=True)
    names = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[1]) == 1 and fields[1].isalpha():
            name = fields[0]
            if cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Darwin" and name.startswith("_"):
                name = name[1:]
            names.add(name)
    return names


def check_lmmc_symbols(record, lmmc, cache):
    library = Path(record["file"])
    shared = record["type"] == "SHARED_LIBRARY"
    expected = expected_symbols(lmmc, cache)
    if shared and library.suffix.lower() == ".dll":
        actual = pe_exports(library, cache)
    else:
        actual = nm_symbols(library, cache, shared)
    missing = expected - actual
    unexpected = actual - expected if shared else set()
    if missing or unexpected:
        raise ValueError("LMMC ABI mismatch: " + json.dumps({"missing": sorted(missing),
                                                           "unexpected": sorted(unexpected)}))
    return {"mode": "export equality" if shared else "static public definitions",
            "required_symbols": len(expected), "actual_symbols": len(actual)}


def otool_paths(cache, mode, library):
    text = run([cache["CMAKE_OTOOL"], mode, library], capture=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].endswith(":"):
        raise ValueError("Unrecognized otool output for " + str(library))
    return [re.sub(r"\s+\(compatibility version .*\)\Z", "", line)
            for line in lines[1:]]


def check_runtime_identity(record, cache, name, forbidden_roots):
    library = Path(record["file"])
    if record["type"] != "SHARED_LIBRARY":
        return None
    if cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Darwin":
        identities = otool_paths(cache, "-D", library)
        expected = DARWIN_RUNTIME_IDS[name]
        if identities != [expected]:
            raise ValueError(
                f"Installed Mach-O identity mismatch for {name}: "
                f"expected {expected}, found {identities}")
        dependencies = otool_paths(cache, "-L", library)
        forbidden_roots = tuple(Path(path).resolve() for path in forbidden_roots)
        for dependency in dependencies:
            path = Path(dependency)
            if path.is_absolute():
                if FIRST_PARTY_DYLIB.fullmatch(path.name):
                    raise ValueError(
                        f"Absolute first-party Mach-O dependency in {library}: {dependency}")
                resolved = path.resolve()
                if any(resolved == root or resolved.is_relative_to(root)
                       for root in forbidden_roots):
                    raise ValueError(
                        f"Mach-O dependency exposes a producer path in {library}: {dependency}")
            if (FIRST_PARTY_DYLIB.fullmatch(path.name)
                    and dependency not in DARWIN_RUNTIME_IDS.values()):
                raise ValueError(
                    f"First-party Mach-O dependency must use its @rpath identity in "
                    f"{library}: {dependency}")
        return {"id": expected, "dependencies": dependencies}
    if name == "LMMP" or ".so" not in library.name:
        return None
    text = run([cache["CMAKE_READELF"], "-d", "--version-info", library], capture=True)
    match = re.search(r"\(SONAME\).*\[([^]]+)\]", text)
    expected = "lib" + name.lower() + ".so"
    if not match or match[1] != expected:
        raise ValueError("Installed SONAME mismatch: expected " + expected)
    if re.search(r"\bName:\s+(?:LMCAS|LMMC)_[0-9]", text):
        raise ValueError("Installed library contains a numeric first-party ELF version node: "
                         + str(library))
    return expected


def configuration_arguments(cache, config):
    command = ["-G", cache["CMAKE_GENERATOR"], "-DCMAKE_BUILD_TYPE=" + config]
    for key, option in (("CMAKE_GENERATOR_PLATFORM", "-A"),
                        ("CMAKE_GENERATOR_TOOLSET", "-T")):
        if cache.get(key):
            command.extend([option, cache[key]])
    settings = {
        "CMAKE_GENERATOR_INSTANCE", "CMAKE_TOOLCHAIN_FILE", "CMAKE_MAKE_PROGRAM",
        "CMAKE_SYSROOT", "CMAKE_SYSROOT_COMPILE", "CMAKE_SYSROOT_LINK",
        "CMAKE_OSX_ARCHITECTURES", "CMAKE_OSX_SYSROOT", "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_MSVC_RUNTIME_LIBRARY", "CMAKE_POSITION_INDEPENDENT_CODE",
        "CMAKE_CONFIGURATION_TYPES", "CMAKE_FIND_ROOT_PATH",
        "CMAKE_FIND_ROOT_PATH_MODE_PROGRAM", "CMAKE_FIND_ROOT_PATH_MODE_LIBRARY",
        "CMAKE_FIND_ROOT_PATH_MODE_INCLUDE", "CMAKE_FIND_ROOT_PATH_MODE_PACKAGE",
        "CMAKE_AR", "CMAKE_RANLIB", "CMAKE_LINKER", "CMAKE_NM", "CMAKE_OBJDUMP",
        "CMAKE_READELF", "CMAKE_OTOOL", "CMAKE_INSTALL_NAME_TOOL",
    }
    if enabled(cache.get("CMAKE_CROSSCOMPILING", "OFF")):
        settings.update(("CMAKE_SYSTEM_NAME", "CMAKE_SYSTEM_VERSION", "CMAKE_SYSTEM_PROCESSOR"))
    pattern = re.compile(
        r"CMAKE_(?:C|CXX|ASM)_(?:COMPILER(?:_ARG1|_TARGET|_EXTERNAL_TOOLCHAIN|_LAUNCHER)?"
        r"|FLAGS(?:_[A-Z0-9_]+)?|STANDARD_LIBRARIES)"
        r"|CMAKE_(?:EXE|SHARED|MODULE|STATIC)_LINKER_FLAGS(?:_[A-Z0-9_]+)?")
    command.extend("-D" + key + "=" + value for key, value in sorted(cache.items())
                   if not key.endswith("_INIT")
                   and (key in settings or pattern.fullmatch(key)))
    return command


def consumer_instrumentation(project, cache):
    prefixes = ("LMCAS", "LMMC") if project == "lmcas" else ("LMMC",)
    flags = []
    for prefix in prefixes:
        if enabled(cache.get(prefix + "_ENABLE_SANITIZERS", "OFF")):
            flags.extend(("-fsanitize=address,undefined", "-fno-omit-frame-pointer"))
        elif enabled(cache.get(prefix + "_ENABLE_TSAN", "OFF")):
            flags.extend(("-fsanitize=thread", "-fno-omit-frame-pointer"))
        elif enabled(cache.get(prefix + "_ENABLE_COVERAGE", "OFF")):
            flags.extend(("--coverage", "-fprofile-update=atomic", "-O0", "-g"))
    if not flags:
        return []
    flags = " ".join(dict.fromkeys(flags))
    return ["-D" + key + "=" + " ".join(filter(None, (cache.get(key, ""), flags)))
            for key in ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS")]


def configure_consumer(args, source, cache, prefix, consumer):
    source = source / ("tests/package_consumer" if args.project == "lmcas" else "test/package_consumer")
    command = ["cmake", "-S", source, "-B", consumer,
               *configuration_arguments(cache, args.config),
               *consumer_instrumentation(args.project, cache),
               "-DCMAKE_PREFIX_PATH=" + str(prefix),
               "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
               "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"]
    run(command)
    run(["cmake", "--build", consumer, "--config", args.config, "--parallel", args.parallel])
    return json.loads((consumer / ("package_check_" + args.config + ".json")).read_text(encoding="utf-8"))


def verify(args, *, build_producer=True, configuration=None, work=None):
    lmmc = Path(__file__).resolve().parent.parent
    source = lmmc.parent if args.project == "lmcas" else lmmc
    cache = cache_values(args.build_dir)
    if Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != source:
        raise ValueError("--project must match the producer's top-level source directory")
    if configuration is not None:
        cache.update(configuration)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if work is None:
        work = Path(tempfile.mkdtemp(prefix=args.project + "-", dir=args.work_dir))
    else:
        work.mkdir(parents=True, exist_ok=True)
        for name in ("original", "relocated"):
            prefix = work / name
            if prefix.exists():
                shutil.rmtree(prefix)
    original, relocated, consumer = (work / name for name in ("original", "relocated", "consumer"))
    if build_producer:
        run(["cmake", "--build", args.build_dir, "--config", args.config, "--parallel", args.parallel])
    run(["cmake", "--install", args.build_dir, "--config", args.config, "--prefix", original])
    shutil.move(str(original), relocated)
    if original.exists():
        raise ValueError("Original install prefix survived relocation")
    check_install_contents(relocated)
    report = configure_consumer(args, source, cache, relocated, consumer)
    namespace = args.project.upper() + "::"
    report["abi"] = check_lmmc_symbols(report["targets"][namespace + "lmmc"], lmmc, cache)
    runtime_targets = [("LMMC", "lmmc")]
    if args.project == "lmcas":
        runtime_targets.insert(0, ("LMCAS", "lmcas"))
    if cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Darwin":
        runtime_targets.append(("LMMP", "liblmmp"))
    forbidden_roots = (source, args.build_dir, original, relocated)
    report["runtime_ids"] = {
        name: check_runtime_identity(
            report["targets"][namespace + target], cache, name, forbidden_roots)
        for name, target in runtime_targets
    }
    env = os.environ.copy()
    directories = sorted({str(Path(record["file"]).parent)
                          for record in report["targets"].values()})
    if cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Darwin":
        env.pop("DYLD_LIBRARY_PATH", None)
        env.pop("DYLD_FALLBACK_LIBRARY_PATH", None)
    elif cache["LMMC_PACKAGE_SYSTEM_NAME"] == "Windows":
        env["PATH"] = os.pathsep.join([*directories, env.get("PATH", "")])
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*directories, env.get("LD_LIBRARY_PATH", "")])
    run([report["executable"]], env=env)
    report.update(passed=True, relocated_prefix=str(relocated), consumer_build=str(consumer),
                  source_ownership=str(consumer / "source_ownership.json"))
    destination = args.work_dir / "package_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Package verification passed:", destination, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("lmcas", "lmmc"), required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Only export Git-visible working sources to a new external directory")
    parser.add_argument("--source-root", type=Path,
                        help="Snapshot repository root (defaults to the selected project's root)")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    if args.snapshot_dir is None and (args.build_dir is None or args.work_dir is None):
        parser.error("--build-dir and --work-dir are required unless --snapshot-dir is used")
    if args.snapshot_dir is not None and (args.build_dir is not None or args.work_dir is not None):
        parser.error("--snapshot-dir cannot be combined with package build/work directories")
    if args.source_root is not None and args.snapshot_dir is None:
        parser.error("--source-root is only valid with --snapshot-dir")
    try:
        if args.snapshot_dir is not None:
            lmmc = Path(__file__).resolve().parent.parent
            source = args.source_root or (lmmc.parent if args.project == "lmcas" else lmmc)
            snapshot_source(source.resolve(), args.snapshot_dir.resolve(), lmmc)
        else:
            args.build_dir = args.build_dir.resolve()
            args.work_dir = args.work_dir.resolve()
            verify(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("Package verification failed:", error, file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stdout or "", error.stderr or "", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
