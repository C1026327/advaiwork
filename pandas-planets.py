import pandas as pd

planetsDF = pd.read_csv("resources/planets.csv")

## Full thing
#print(pd.DataFrame(planetsDF))

# First 5 Rows
print(planetsDF.head())

# Rows & Columns (Shape)
print(planetsDF.shape)

rows, columns = planetsDF.shape

print("There are ", rows, " rows in Planets.csv.")
print("There are ", columns, "columns in Planets.csv.")

for col in planetsDF:
    print(col)

print("\n")

moons = planetsDF['Number of Moons']
totalmoons = 0
for mooncount in moons:
    totalmoons += mooncount

print(totalmoons)

print(planetsDF.Planet)

print(planetsDF.info())
print(planetsDF.describe())

print(planetsDF.head())
print(planetsDF.tail())

planets = planetsDF['Planet']
for planet in planets:
    print(planet, end=", ")

print(planetsDF.iloc[3])

print(planetsDF[['Planet', 'Color']])

earth = planetsDF.iloc[2]
color = earth.Color

print("Earth's Colors are: ", color)
