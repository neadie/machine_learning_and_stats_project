# Jupter notebook

1. Create virtual enviroment 
   use command **python -m venv { whatever path you want for virtual enviroment}**
2. Go to Scripts folder of the virtual enviroment and run **activate **
3. Cd into the project and to install all the requirements use command  **pip install --no-cache-dir -r requirements.txt**
4. To run the jupyter notebook run **jupyter lab**

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


