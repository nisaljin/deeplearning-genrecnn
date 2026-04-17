# Repo Sample Audio

Put deploy-safe sample audio clips here (for example, short `.mp3` files).

- The app scans this folder recursively.
- `/api/random-audio` returns one random sample.
- `/api/audio/:sampleId` streams that sample.
- `/api/predict` accepts `sampleId` and forwards bytes to the backend model API.

Keep files small enough for your deployment limits.
