# Raw data is not committed to Git.


## After downloading and extracting the dataset, place the year folders under `data/raw/`.

Expected local structure:

```text
data/raw/
├── 2010/
│   └── 2010/
│       ├── image_001.jpg
│       ├── image_001.txt
│       └── ...
├── 2015/
│   └── 2015/
│       ├── image_002.jpg
│       ├── image_002.txt
│       └── ...
├── 2017/
│   └── 2017/
├── 2018/
│   └── 2018/
└── 2021/
    └── 2021/
```

Each image should have a corresponding `.txt` annotation file with the same filename stem.

Example:

```text
data/raw/2010/2010/example.jpg
data/raw/2010/2010/example.txt
```