# Bipartite Tracking for Muscle Stem Cells

This repository implements a simple cell tracker based on bipartite graph matching, with some assumptions made that optimize tracking for muscle stem cells.

## Usage

To track cells for a given movie, place binary masks and images in a single directory
and use the `track_well.py` CLI:

```bash
python track_well.py movie_path \
  --img_regex "STRING_TO_GLOB_IMAGES" \
  --mask_regex "STRING_TO_GLOB_MASKS" \
```

See the help menu for further tracking options:

```bash
python track_well.py -h
```
