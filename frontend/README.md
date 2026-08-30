# Frontend

React + Vite + TypeScript application. Owned by the integration team;
model groups do not edit this directory.

## Stack

Vite, React, TypeScript, Tailwind CSS v4. Design tokens
(`.claude/rules/frontend.md`) live in `src/index.css` as a Tailwind v4
`@theme` block — there is no separate `tailwind.config.js`.

## API types

`src/types/api.ts` is generated, not hand-written. It mirrors the backend's
Pydantic schemas (the authoritative contract — `.claude/rules/backend.md`) via
its live OpenAPI schema. Regenerate it whenever the backend contract changes:

```bash
cd backend && uvicorn app.main:app --reload --port 8050    # backend must be running
cd frontend && npm run gen:types
```

`src/api/client.ts` is a typed client (`openapi-fetch`) bound to those
generated types, so a contract change that isn't reflected in `api.ts` surfaces
as a `tsc` error, not a runtime failure. `src/api/errors.ts` normalises the
backend's single error shape and FastAPI's own validation-error shape into one
readable message.

## State

`useModels` and `useTraining` (`src/hooks/`) hold the app's data-fetching
state. No global store.

## Structure

- `src/components/StepShell.tsx`, `StepIndicator.tsx` — the seven-step shell
  (`.claude/rules/frontend.md`'s layout contract). Screens 1–7 land in
  Sessions 6–7; today each step but Upload renders a placeholder panel.
- `src/api/` — typed client + error normalisation.
- `src/hooks/` — `useModels`, `useTraining`.
- `src/types/` — generated API types (`api.ts`) plus `steps.ts`.

## Commands

```bash
npm run dev         # localhost:5173, proxies /api to localhost:8050
npm run build        # tsc -b && vite build
npm run test          # vitest run
npm run gen:types     # regenerate src/types/api.ts from the live backend
```
