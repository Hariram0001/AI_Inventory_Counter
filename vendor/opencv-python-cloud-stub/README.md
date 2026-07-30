# opencv-python cloud stub

`inference-sdk` and `supervision` declare a hard dependency on the PyPI package
named `opencv-python` (GUI build). That wheel links against `libGL` and
`libgthread-2.0`, which Streamlit Community Cloud does not provide — and
`libglib2.0-0` cannot be installed via `packages.txt` on the current base image.

This local distribution **reuses the name** `opencv-python==4.10.0.84` but only
depends on `opencv-python-headless`, so pip never installs the GUI wheel.
`cv2` still comes from the headless package.
