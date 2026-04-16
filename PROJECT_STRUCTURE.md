# Project Structure

## Core Runtime

These files are part of the working parking workflow and should be kept:

- `run.py`
- `config.py`
- `enhanced_parking_detector.py`
- `car_detector.py`
- `setup_directories.py`
- `data/`
- `tests/`

## Optional / Supporting

These files are useful, but not required for the main camera workflow:

- `gui.py`
- `README.md`
- `TROUBLESHOOTING.md`
- `requirements.txt`
- `requirements-dev.txt`
- `pytest.ini`
- `setup.cfg`

## Legacy / Experimental

These are older or standalone helper scripts and are not used by `run.py`:

- `main.py`
- `scratch.py`

They have been moved into `legacy/` and root wrappers are kept for compatibility.
