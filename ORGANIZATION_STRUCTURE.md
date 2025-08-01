# ProteinMD Project Organization

Dieses Dokument beschreibt die neue Ordnerstruktur des ProteinMD-Projekts nach der Reorganisation.

## Ordnerstruktur

### 📁 `docs/`
Alle Dokumentation und Reports
- **`reports/`** - Allgemeine Dokumentation (CLI_DOCUMENTATION.md, Fix-Reports, etc.)
- **`session_summaries/`** - Entwicklungs-Session-Zusammenfassungen
- **`task_reports/`** - Task-spezifische Completion Reports

### 📁 `demos/`
Demo-Skripte und Beispielanwendungen
- GUI-Integrationsdemos
- Feature-Demonstrationen

### 📁 `scripts/`
Utility-Skripte organisiert nach Zweck
- **`testing/`** - Test-Skripte (test_*.py, simplified_large_test.py)
- **`validation/`** - Validierungs-Skripte (validate_*.py)
- **`benchmarks/`** - Benchmark- und Performance-Tests
- Weitere Utility-Skripte (debug_pdb.py, etc.)

### 📁 `results/`
Alle Ergebnisdateien
- **`benchmarks/`** - Benchmark-Ergebnisse (JSON-Dateien)
- **`validation/`** - Validierungs-Ergebnisse und Log-Dateien
- **`test_outputs/`** - Test-Output-Dateien (NPZ, CSV, DAT, PNG, PDB)

### 📁 `media/`
Mediendateien (Bilder, etc.)

## Wichtige Dateien im Root-Verzeichnis

- `README.md` - Hauptdokumentation
- `setup.py` / `pyproject.toml` - Package-Konfiguration
- `Makefile` - Build-Konfiguration
- `docker-compose.yml` / `Dockerfile` - Container-Konfiguration
- `gui_launcher.py` - GUI-Starter
- `proteinmd_config.json` - Hauptkonfiguration
- `aufgabenliste.md` - Task-Liste

## Bestehende Ordner (unverändert)

- `proteinMD/` - Hauptpaket-Code
- `tests/` - Unit-Tests
- `examples/` - Beispiele
- `data/` - Datendateien
- `config/` - Konfigurationsdateien
- `logs/` - Log-Verzeichnis

## Nutzen der neuen Struktur

1. **Bessere Navigation** - Verwandte Dateien sind gruppiert
2. **Klarere Trennung** - Dokumentation, Code, Tests und Ergebnisse sind getrennt
3. **Einfachere Wartung** - Neue Dateien können leicht in die richtige Kategorie einsortiert werden
4. **Übersichtlichkeit** - Das Root-Verzeichnis ist jetzt viel sauberer

## Empfohlene Arbeitsweise

- **Neue Demo-Skripte** → `demos/`
- **Neue Test-Skripte** → `scripts/testing/`
- **Validierungs-Skripte** → `scripts/validation/`
- **Benchmark-Ergebnisse** → `results/benchmarks/`
- **Dokumentation** → `docs/reports/`
- **Task-Reports** → `docs/task_reports/`
