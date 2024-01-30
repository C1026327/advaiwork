import pandas as pd

# Array Test

array=["java", "python", "groovy"]

panarray = pd.Series(array, index = ["a","b","c"])

print(panarray)

#####

# Data Frame test

data = {

    "Sheffield Hallam University" : [37000],
    "Shefield University": [30300]
}

myvar = pd.DataFrame(data, index = ["student population"])

print(myvar)

#####

# Python Dictionary

dict = {
    "Earth": "1",
    "Mars": "2",
    "Jupiter": "79"
}
print("\n")
pandadict = pd.Series(dict)
print(pandadict)

#####