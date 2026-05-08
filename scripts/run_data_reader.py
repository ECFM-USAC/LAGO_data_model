import sys
import os
from pathlib import Path
from lago_data_model.reader import LagFileReader

def process_one(lag_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = LagFileReader(str(lag_path))
    prefix = lag_path.stem
    out = reader.save_as_parquet_streaming(str(out_dir), prefix=prefix)
    print(f"  -> {out}")
    return Path(out)

def main(argv):
    if len(argv) < 2:
        print("Uso:")
        print("  poetry run python scripts/run_data_reader.py <archivo.lag | directorio>")
        sys.exit(1)

    target = Path(argv[1])
    out_dir = Path("./data")

    if target.is_file():
        if target.suffix != ".lag":
            print(f"Aviso: {target} no termina en .lag (procesando igual)")
        print(f"Procesando 1 archivo: {target}")
        process_one(target, out_dir)
        return

    if target.is_dir():
        lag_files = sorted(target.glob("*.lag"))
        if not lag_files:
            print(f"No se encontraron archivos .lag en {target}")
            sys.exit(1)
        print(f"Procesando {len(lag_files)} archivos de {target}:")
        for f in lag_files:
            print(f"[{f.name}]")
            process_one(f, out_dir)
        return

    print(f"Error: {target} no existe")
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
