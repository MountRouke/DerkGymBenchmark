# Benchmark for Derk's Gym

## Results

#### MSI Laptop
simulation_only=False Renderer=ANGLE (NVIDIA GeForce GTX 1070 Direct3D11 vs_5_0 ps_5_0) Vendor=Google Inc.
n_arenas | create env | reset | run
--- | --- | --- | ---
1 | 11.38662576675415 | 0.7041175365447998 | 6.889879465103149
16 | 9.555910348892212 | 2.7223434448242188 | 7.259873867034912
128 | 9.410910844802856 | 3.552177906036377 | 9.509382724761963
256 | 9.39712905883789 | 4.501130104064941 | 14.37168264389038
512 | 9.467995166778564 | 6.397855997085571 | 23.848492860794067

#### [Google Colab](https://colab.research.google.com/drive/1n5Bl1pdBpQphOCOGWC31uUbmjMubPUM1?usp=sharing)
gym_derk=0.3.31 simulation_only=False Renderer=ANGLE (NVIDIA Corporation, Tesla P100-PCIE-16GB/PCIe/SSE2, OpenGL 4.5 core) Vendor=Google Inc.
n_arenas | create env | reset | run
--- | --- | --- | ---
1 | 6.508690118789673 | 1.0353975296020508 | 14.979998588562012
16 | 7.248045206069946 | 6.955777883529663 | 15.178634405136108
128 | 6.8916120529174805 | 3.758395195007324 | 25.518474817276
256 | 7.1017746925354 | 19.98109531402588 | 47.711907625198364
512 | 7.296813726425171 | 64.13919425010681 | 88.16214418411255

## Results (simulation only)

With `simulation_only` we are skipping sending actions and reading observations, so it measures the raw simulation performance on a system.

#### MSI Laptop
simulation_only=True Renderer=ANGLE (NVIDIA GeForce GTX 1070 Direct3D11 vs_5_0 ps_5_0) Vendor=Google Inc.
n_arenas | create env | reset | run
--- | --- | --- | ---
1 | 10.696801662445068 | 1.2546467781066895 | 8.589587211608887
16 | 9.075611114501953 | 3.0731422901153564 | 7.435558319091797
128 | 11.010828018188477 | 3.5832571983337402 | 7.238966941833496
256 | 9.316890716552734 | 4.478029727935791 | 7.207346200942993
512 | 10.956312656402588 | 6.25600266456604 | 7.5004003047943115

#### Google Colab
simulation_only=True Renderer=ANGLE (NVIDIA Corporation, Tesla P100-PCIE-16GB/PCIe/SSE2, OpenGL 4.5 core) Vendor=Google Inc.
n_arenas | create env | reset | run
--- | --- | --- | ---
1 | 6.899520397186279 | 1.3478658199310303 | 12.156709909439087
16 | 6.550668716430664 | 9.02074384689331 | 14.280853033065796
128 | 6.671059846878052 | 39.776060819625854 | 20.020533084869385
256 | 6.6941001415252686 | 92.20242023468018 | 27.712412118911743
512 | 6.8093101978302 | 28.824622631072998 | 41.897865533828735
