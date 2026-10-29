from pathlib import Path
import re
import subprocess
import sys


def main():
    project_dir = Path(__file__).resolve().parent
    pattern = re.compile(r"^(\d+)\..+\.py$", re.IGNORECASE)

    scripts = []

    for path in project_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.fullmatch(path.name)

        if match:
            scripts.append((int(match.group(1)), path))

    # Orden numérico: 1, 2, 3, ... 10, 11
    scripts.sort(key=lambda item: (item[0], item[1].name.lower()))

    if not scripts:
        print("No se encontraron scripts numerados.")
        return 1

    for index, (_, path) in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] Ejecutando {path.name}",
              flush=True)

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=project_dir
        )

        if result.returncode != 0:
            print(f"ERROR en {path.name}. Pipeline detenido.")
            return result.returncode

        print(f"COMPLETADO: {path.name}", flush=True)

    print("\nPipeline completado correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
