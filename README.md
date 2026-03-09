# Portugal Seismic Hazard Dashboard

An interactive web dashboard for visualising seismic hazard results for Portugal, built with [Plotly Dash](https://dash.plotly.com/).

## What the app does

The dashboard has three tabs:

- **Hazard Maps** — displays ground motion intensity maps (mean and quantile) for different Ground Motion Models (GMMs) and intensity measures (PGA, SA).
- **Hazard Curves** — plots probability of exceedance curves for selected cities and GMMs, with options to compare cities or compare models side-by-side.
- **Uniform Hazard Spectra (UHS)** — plots response spectra at selected return periods for different cities and GMMs.

All data is sourced from OpenQuake Engine outputs. Charts are interactive and results can be downloaded as CSV.

## Deploying on Render

1. **Fork or clone this repository** to your own GitHub account.

2. Go to [render.com](https://render.com) and sign in (free account is sufficient).

3. Click **New → Web Service** and connect your GitHub repository.

4. Configure the service with the following settings:

   | Field | Value |
   |---|---|
   | **Language** | Python 3 |
   | **Build command** | `pip install -r requirements.txt` |
   | **Start command** | `gunicorn --chdir Mapping/dashboard dashboard:server --bind 0.0.0.0:$PORT` |
   | **Instance type** | Free |

5. Click **Deploy**. Render will build and launch the app — the public URL is shown at the top of the service page once deployment is complete.

> **Note:** Free instances spin down after ~15 minutes of inactivity. The first request after a period of no use may take 30–60 seconds to load. This can be avoided by setting up a free uptime monitor (e.g. [UptimeRobot](https://uptimerobot.com)) to ping the URL every 14 minutes.
