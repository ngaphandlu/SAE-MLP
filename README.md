# SAE-MLP

This is our research work about some deep learning methods applied in indoor positioning based on WiFi.

## Getting Started

1. **Install required Python libraries.** Our main libraries include: `tensorflow`, `keras`, `pandas`, `numpy`, and `matplotlib`.

2. **Focus on running the main scripts** to evaluate our models:
   * Run `main_UJIIndoorLoc.py` to evaluate performance on the **UJIIndoorLoc** and **TUT** datasets.
   * Run `main_UJIIndoorLoc-B.py` for the **UJIIndoorLoc-B** dataset.

3. **Our model implementations** are in the `model.py` file, and the utility functions are in `data_helper.py`.

4. **The datasets** are stored in the `UJIIndoorLocData` and `TUTData` folders.

5. **Noise injection and model evaluation:**
   * Run `test_file_noise_injection.py` to generate noisy test datasets by injecting noise into the original test data.
   * Run `new_main_10_2026.py` to evaluate the proposed models using the generated noisy test datasets.
   * The evaluation results are used to assess model robustness and performance under different noise levels.

