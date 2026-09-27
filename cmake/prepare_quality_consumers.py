#!/usr/bin/env python3
"""Prepare actual installed-package consumers for automatic source ownership gates."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from check_package import cache_values, configuration_arguments, run, verify


def producer_arguments(settings):
    options = {
        "LMMP_ROOT", "LMMP_INCLUDE_DIR", "LMMP_ASM", "BUILD_SHARED_LIBS",
        "LMCAS_LMMP_ASM", "LMMC_LMMP_ASM", "LMMC_BUILD_SHARED", "LMMC_DEBUG_LEAKS",
        "CMAKE_INSTALL_LIBDIR", "CMAKE_INSTALL_BINDIR", "CMAKE_INSTALL_INCLUDEDIR",
    }
    for project in ("LMCAS", "LMMC"):
        options.update(project + suffix for suffix in (
            "_ENABLE_SANITIZERS", "_ENABLE_TSAN", "_ENABLE_COVERAGE",
            "_WARNINGS_AS_ERRORS"))
    command = ["-D" + key + "=" + value for key, value in sorted(settings.items())
               if key in options]
    disabled = (
        "BUILD_TESTING", "LMCAS_BUILD_TESTS", "LMCAS_BUILD_LMMC_TESTS",
        "LMCAS_BUILD_BENCHMARKS", "LMCAS_ENABLE_QUALITY_GATES",
        "LMMC_BUILD_TESTS", "LMMC_BUILD_EXAMPLES", "LMMC_ENABLE_QUALITY_GATES",
        "LMMP_BUILD_TESTS", "LMMP_BUILD_EXAMPLES", "LMMP_BUILD_BENCHMARKS",
    )
    command.extend("-D" + key + "=OFF" for key in disabled)
    command.append("-DLMMP_LIBRARY:FILEPATH=")
    return command


def prepare_project(spec, args):
    project = spec["project"]
    settings = spec["settings"]
    build = Path(spec["build"])
    configuration = settings
    if not spec["reuse"]:
        build = args.work_dir / (project + "-producer")
        run(["cmake", "-S", spec["source"], "-B", build,
             *configuration_arguments(settings, args.config),
             *producer_arguments(settings)])
        run(["cmake", "--build", build, "--config", args.config,
             "--target", project, "--parallel", args.parallel])
        configuration = None
    package_args = argparse.Namespace(project=project, build_dir=build,
                                      work_dir=args.work_dir / project,
                                      config=args.config, parallel=args.parallel)
    return verify(package_args, build_producer=False, configuration=configuration,
                  work=package_args.work_dir / "verification")


def publish_manifest(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="ownership-",
                                     suffix=".json", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        shutil.copyfile(source, staged)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def preparation_specs(paths):
    specs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for spec in specs:
        if spec["project"] not in {"lmcas", "lmmc"}:
            raise ValueError("Unknown quality package project: " + spec["project"])
        cached = cache_values(Path(spec["build"]))
        cached.update(spec["settings"])
        spec["settings"] = cached
    lmmc = next((spec for spec in specs if spec["project"] == "lmmc"), None)
    if lmmc is not None:
        for spec in specs:
            if spec["project"] == "lmcas":
                spec["settings"].update((key, value) for key, value in lmmc["settings"].items()
                                        if key.startswith(("LMMC_", "LMMP_")))
    return specs


def prepare(args, report):
    specs = preparation_specs(args.project_config)
    destinations = {spec["project"]: args.work_dir / spec["project"] / "source_ownership.json"
                    for spec in specs}
    # A failed preparation cannot leave a previous successful ownership inventory usable.
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        (destination.parent / "package_report.json").unlink(missing_ok=True)
    manifests = []
    for spec in specs:
        started = time.monotonic()
        package = prepare_project(spec, args)
        project = spec["project"]
        report["projects"][project] = {
            "elapsed_seconds": time.monotonic() - started,
            "package_report": str(args.work_dir / project / "package_report.json"),
            "source_ownership": str(destinations[project]),
        }
        manifests.append((Path(package["source_ownership"]), destinations[project]))
    for source, destination in manifests:
        publish_manifest(source, destination)
    report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, action="append", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    started = time.monotonic()
    report = {"passed": False, "projects": {}}
    try:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        prepare(args, report)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
        print("Quality consumer preparation failed:", error, file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            report.update(command=error.cmd, stdout=error.stdout, stderr=error.stderr)
            print(error.stdout or "", error.stderr or "", file=sys.stderr)
    report["elapsed_seconds"] = time.monotonic() - started
    destination = args.work_dir / "preparation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Quality consumer preparation report:", destination,
          "elapsed seconds:", report["elapsed_seconds"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
