# ML Integration Project

A single web application that combines twelve machine-learning models,
each built by a separate group, behind one upload-train-evaluate-predict
interface.

## Structure
- `backend/` — FastAPI service, model interface, validator, the 12 model folders
- `frontend/` — React + Vite + TypeScript UI
- `.github/workflows/ci.yml` — lint, test, and validation on every pull request

## For model groups
Read `CODING_STANDARDS.md`. Your work goes in your own
`backend/models/group_<NN>_<name>/` folder. Fork this repo, implement your
model, run `python backend/validate_submission.py backend/models/group_<NN>_<name>/`,
then open a pull request against `main`.

## For the integration team
See `ARCHITECTURE.md`.
