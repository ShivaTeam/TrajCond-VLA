# Action codebooks

`build_codebook_from_gt.py` constructs action codebooks and variance files:

```text
Local [dx, dy, dyaw] actions -> codebook JSON + action variances JSON
```

KMeans and KDisks fit complete three-dimensional action vectors.
Each center is one token. No intermediate action file is needed, and importing
the script does not start file I/O. All code comments and docstrings are English.

## Total token size

`--size` and `--codebook-size` always specify the **total action vocabulary
size**, defaulting to 768.

| Method | Example size | Codebook shape | Tokens per action |
| --- | --- | --- | --- |
| KMeans | 768 | `(768, 3)` | 1 |
| KDisks | 768 | `(768, 3)` | 1 |
| Brohan | 768 | `(3, 256)` | 3 |

KMeans and KDisks accept positive integer sizes. Brohan divides the total
equally across three axes; its size must be a multiple of 3 and at least 6.
For example, `--method brohan --size 384` creates 128 values on each axis.

## Install and run

Brohan requires NumPy. KMeans also requires scikit-learn and threadpoolctl.
KDisks also requires SciPy for nearest-center assignment counts.

```bash
python -m pip install -r codebook/requirements.txt
python codebook/build_codebook_from_gt.py --method kmeans --size 768
python codebook/build_codebook_from_gt.py --method kdisks --size 768 --epsilon 0.03
python codebook/build_codebook_from_gt.py --method brohan --size 768
```

For a different output path or clustering seed:

```bash
python codebook/build_codebook_from_gt.py --method kmeans --size 512 --seed 42 --output /path/to/kmeans.json
```

The default method is Brohan and the default seed is 42.
Constant individual axes are valid for joint clustering, provided there
are enough distinct complete action vectors.

## Algorithms

KMeans fits all `(M, 3)` actions jointly using full scikit-learn `KMeans` with
`n_clusters=size`, `random_state=seed`, and `n_init="auto"`.
The method uses all samples, preserves duplicate
sample weights, and applies a single-thread limit during fitting.

KDisks uses one seeded NumPy generator to select a remaining complete action
vector, then removes candidates whose three-dimensional Euclidean distance is
`<= epsilon`. The default epsilon is `0.03`:

```text
distance = sqrt(delta_dx**2 + delta_dy**2 + delta_dyaw**2)
```

The distance uses raw local actions, without rescaling individual axes. Input
order, center selection order, and full numeric precision are preserved.
If candidates are exhausted before the requested number of centers is found,
the script reports the center count and exits without writing a codebook.
Reduce epsilon or size in this case.

For KDisks, nearest-center assignments use SciPy `cdist` and `argmin`; ties
select the first center. Distances are evaluated in chunks with about 8 MiB
per distance buffer, instead of allocating the full `M * size` distance matrix.
Selected rows are copied so each center does not retain an entire candidate
array. These memory changes preserve the algorithm's center values and order.

Brohan keeps an inclusive min/max grid for each axis, with `size/3` values
rounded to six decimals. Its output rows remain sorted ascending.

## Output

The generated codebook filenames are:

```text
kmeans_codebook_<size>.json
kdisks_codebook_<size>.json
brohan_codebook_1d_size_<size>.json
```

KMeans and KDisks use the original single-codebook schema:

```json
{
  "k": 2,
  "codebook": [[0.05, -1.95, -0.475], [4.1, 2.1, 0.45]],
  "cluster_sizes": [4, 2]
}
```

Each row stores `[dx, dy, dyaw]`. Token IDs are the row indices. Centers are
saved in their original order and precision, without sorting or rounding.
`cluster_sizes[i]` counts actions assigned to token i; the counts sum to M.
These counts are stored inside the codebook JSON.

Brohan retains `k=[size/3, size/3, size/3]`, the `(3, size/3)` codebook, and
`boundaries.min/max`. Here too, size in the output filename means total tokens.

Every method also saves `action_variances.json` in the codebook's output
directory, including when `--output` selects a different directory. It uses
all extracted training actions, in `[dx, dy, dyaw]` order:

```python
stds = np.std(actions, axis=0, ddof=0)
variances = np.maximum(stds, 1e-6) ** 2
```

The file contains only `{"variances": [var_dx, var_dy, var_dyaw]}`. The minimum
variance is `1e-12`. Variances depend on the training actions rather than the
selected method or token size. Both JSON files are written after the codebook
has been built successfully.

The terminal prints action counts,
min/max/mean/population standard deviation (`ddof=0`) for each axis, method,
codebook shape, total vocabulary size, and both output paths. Each JSON is written
atomically. The output files must be distinct. Exit codes are 0 for success
and 2 for errors.

## Verify

```bash
python -m unittest discover -s codebook/tests -v
```
