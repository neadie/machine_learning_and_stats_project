# Random numerical app.

# Linux
```bash

python3 powerProductionApp.py
```

# Windows
```bash

python powerProductionApp.py
```

```bash
docker build . -t powerproductionapp-image
docker run --name powerproductionapp-container -d -p 5000:5000 powerproductionapp-image
```