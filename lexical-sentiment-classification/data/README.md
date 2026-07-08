# Data

This project uses the blog corpus from Masrani, Murray, Field & Carenini (2017),
"Detecting dementia through retrospective analysis of routine blog posts by bloggers
with dementia" (BioNLP Workshop 2017).

## Why the raw data isn't included in this repository

The corpus consists of real, identifiable blog posts written by individuals with
Alzheimer's disease and their family members. Even where a dataset has been used in
prior published research, redistributing real patient-authored health text isn't
something to do by default — so it's deliberately left out of this repository.

To run the notebook, obtain the corpus directly from the original authors or their
published release, and place the two class files here as:

```
data/raw/class1.txt   # Alzheimer's / dementia blogger posts
data/raw/class2.txt   # control (family member) posts
```

Each file should contain one or more entries in the format:

```
<doc date="YYYY-MM-DD">
post text goes here
</doc>
```

## Note on data.csv / the .gitignore

`data/raw/` is excluded via `.gitignore` so raw corpus files never get committed by
accident, even if you place a real copy there locally.
