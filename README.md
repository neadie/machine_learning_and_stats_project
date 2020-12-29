# Random numerical app.

# Linux
```bash
export FLASK_APP=powerProductionApp.py
python3 -m flask run
```

# Windows
```bash
set FLASK_APP=powerProductionApp.py
python -m flask run
```

```bash
docker build . -t powerProductionApp-image
docker run --name powerProductionApp-container -d -p 5000:5000 powerProductionApp-image
```