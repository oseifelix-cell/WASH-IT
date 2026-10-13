"""Continuous-sequence tests run on Kaggle (one call per recording set).
Requires test_continuous_sequence.py in the same folder and the recordings attached as Kaggle datasets.
Replace the dataset paths with your own recordings."""

exec(open("test_continuous_sequence.py").read())

RECORDING_SETS = [
    "/kaggle/input/<your-recordings>/recording_set_01",
    "/kaggle/input/<your-recordings>/recording_set_02",
    "/kaggle/input/<your-recordings>/recording_set_03",
    "/kaggle/input/<your-recordings>/recording_set_04",
    "/kaggle/input/<your-recordings>/recording_set_05",
    "/kaggle/input/<your-recordings>/recording_set_06",
    "/kaggle/input/<your-recordings>/recording_set_07",
    "/kaggle/input/<your-recordings>/recording_set_08",
    "/kaggle/input/<your-recordings>/recording_set_09",
    "/kaggle/input/<your-recordings>/recording_set_08",
    "/kaggle/input/<your-recordings>/recording_set_05",
    "/kaggle/input/<your-recordings>/recording_set_07",
]

for path in RECORDING_SETS:
    print("=" * 70, "\n", path)
    result = test_sequence(path, step_durations=[8, 8, 8, 8, 8, 8, 8], tolerance=3.0)
