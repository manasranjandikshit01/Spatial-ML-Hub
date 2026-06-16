---
name: ISRO Dashboard routing
description: Streamlit port config and routing constraints for the isro-dashboard artifact
---

## Rule
The dashboard runs on port **25295**, bound to `0.0.0.0`, at path `/`.
Both of these files must agree on the port or the dashboard breaks:
1. `isro-aqi-hcho/.streamlit/config.toml` — `server.port = 25295`
2. `artifacts/isro-dashboard/.replit-artifact/artifact.toml` — `localPort = 25295`

**Why:** The Replit reverse proxy routes incoming requests on `/` to `localhost:25295`.
If these disagree, the proxy can't reach Streamlit.

**How to apply:** If asked to change the port, update BOTH files atomically. Never use `st.image()` with external Wikipedia/CDN URLs in the sidebar — they are blocked in the Replit sandbox. Use emoji or local assets instead.

## Streamlit config.toml required settings
```toml
[server]
port = 25295
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false
```

## Demo data boot time
On cold start with no `data/processed/` CSVs, `generate_demo_data()` runs on the first request.
This takes ~5–10 seconds. The page appears blank during this time — it is not broken.
