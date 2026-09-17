# Render Deployment & Verification Checklist for THESIS Monad HQ

## Pre-Deployment Setup
- [x] Create `server/monad_manifest.json` defining 15 apps, 10 primitives, and 7 cloud engines.
- [x] Wire Express `/platform`, `/platform/apps`, `/platform/primitives`, `/platform/apps/:id/invoke`, and `/engines/*` endpoints in `server/index.js`.
- [x] Build THESIS Monad HQ Operator Console UI in `client/src/App.jsx` with glassmorphic styling in `client/src/styles.css`.
- [x] Create `render.yaml` deployment specification blueprint.

## Render Deployment Steps
1. Connect your repository to **Render** (https://dashboard.render.com).
2. Create a new **Web Service** selecting `render.yaml` or set:
   - **Build Command**: `cd client && npm install && npm run build && cd ../server && npm install`
   - **Start Command**: `node server/index.js`
   - **Environment Variable**: `MONAD_RPC_URL` (optional: set Monad RPC URL).
3. Deploy the service.

## Post-Deployment API Verification
Verify that the following endpoints return valid JSON on your Render URL:
- `GET https://<your-render-app>.onrender.com/platform` -> Returns status and item counts.
- `GET https://<your-render-app>.onrender.com/platform/apps` -> Returns 15 apps list.
- `GET https://<your-render-app>.onrender.com/platform/primitives` -> Returns 10 primitives matrix.
- `POST https://<your-render-app>.onrender.com/platform/apps/trading-desk/invoke` -> Returns execution status.
- `GET https://<your-render-app>.onrender.com/engines` -> Returns 7 cloud engine workers.
