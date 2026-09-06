---
title: Shelf Life
emoji: 📋
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: 1.36.0
app_file: app/streamlit_app.py
pinned: false
---

# shelf-life — the form

Predicts whether a job posting will be **removed from the board** within the
model's horizon, from what is knowable the moment the posting first appears.

**Removed from the board is not the same as filled.** A posting can be pulled,
expire, or move. That caveat is on the prediction screen too, where the number
is, because that is where the person who needs it is looking.

This Space is the **user interface only**. It holds no model and imports no
modelling code: it calls the API over HTTP, the same way any other client would.
That separation is the point — UI ≠ API ≠ model ≠ training pipeline — and here
it is enforced by physics rather than by a test, because `src/` is not deployed
to this Space at all.

Configure it with one Space secret:

| Name | Value |
|---|---|
| `SHELF_LIFE_API` | the base URL of the Cloud Run service |

The first request after the API has been idle wakes a scale-to-zero container
and takes noticeably longer than the ones after it. The client sets a 30-second
timeout for exactly that reason.

Source, method, leakage audit and the reasons behind every decision:
<https://github.com/professor3333/shelf-life>
