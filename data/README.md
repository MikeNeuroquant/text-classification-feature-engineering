# Data

This project uses the blog corpus from Masrani, Murray, Field & Carenini (2017),
"Detecting dementia through retrospective analysis of routine blog posts by bloggers
with dementia" (BioNLP Workshop 2017).

Data are available at the original source: https://github.com/ml-for-nlp/SVMs
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
