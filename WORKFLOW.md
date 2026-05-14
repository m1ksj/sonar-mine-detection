## Branches

```text
main = stable version
dev = shared working version
name/task = personal work branch
```

Examples:

```text
mika/setup
halil/dataset-manifest
isik/splits
stijn/yolo-data
```

## Starting a task

```powershell
git checkout dev
git pull origin dev
git checkout -b name/task
```

## Before committing

```powershell
python -m unittest discover -v tests
pre-commit run --all-files
```

## Commit style

Use meaningful messages.

## Push branch

```powershell
git push -u origin name/task
```

## Merge rule

Prefer merging finished branches into `dev`.

Do not commit directly to `main`. !!!

## Data rule

Never commit:

```text
data/raw/
data/processed/
runs/
models/checkpoints/
```

## Communication rule

Before changing files someone else is actively working on, write in the group chat.

## Dataset rule

The downloaded raw dataset should be placed locally under:

```text
data/raw/
```

expected local structure:

```text
data/raw/2010/2010/
data/raw/2015/2015/
data/raw/2017/2017/
data/raw/2018/2018/
data/raw/2021/2021/
```

Do not commit these raw files.