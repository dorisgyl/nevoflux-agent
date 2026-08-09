"""Build a tiny stand-in for kokoro-voices-v1.0.bin.

The real bank is 28 MB of 54 voices; tests only need the container shape,
so this writes two voices with the same dtype and rank but a short first
axis. Run: python3 make_fixture.py
"""
import numpy as np, zipfile, io

def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

with zipfile.ZipFile("tiny-voices.bin", "w") as z:
    for name, fill in (("af_test", 0.5), ("zf_test", 0.25)):
        arr = np.full((4, 1, 256), fill, dtype="<f4")
        # Make row index recoverable so the style-lookup test can assert it.
        for i in range(4):
            arr[i, 0, 0] = float(i)
        z.writestr(f"{name}.npy", npy_bytes(arr))
print("wrote tiny-voices.bin")
