# ML Data Validator — Frontend

React + Vite + TypeScript frontend for the ML Data Validator.

## Dev

```bash
npm install
npm run dev       # http://localhost:5173
```

Requires the FastAPI backend running on port 8000:
```bash
# from project root
python run.py
```

## Build (production)

```bash
npm run build     # outputs to dist/
```

The built `dist/` folder is automatically served by the FastAPI backend when you run `python run.py` — no separate frontend server needed in production.

## Stack

- **React 18** + TypeScript
- **Vite** — build tooling and dev server
- **Tailwind CSS** — utility-first styling
- **AG Grid** — interactive data table (validation results, training preview)

## Structure

```
src/
├── api/client.ts          # All API calls (typed)
├── types/index.ts         # Shared TypeScript types
├── components/
│   ├── validate/          # ValidateTab, ValidationGrid, CorrectionsPanel, QualityMetrics, ExportSection
│   ├── train/             # TrainTab, TrainingMetrics, ModelsList
│   └── Collapsible.tsx    # Shared collapsible panel
└── App.tsx                # Root component, tab routing
```
