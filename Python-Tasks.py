
turingQuote = "A computer would deserve to be called intelligent if it could deceive a human mind into believing it was human."
characters = len(turingQuote)
words = len(turingQuote.split())
removeSpecial = len(turingQuote) - (turingQuote.count(' ') + turingQuote.count('.'))
x = 0
userInput=input("What would you like to do today? \n 1) Print turingQuote \n 2) Print turingQuote in Caps \n 3) Input and recieve an all Capital word \n 4) Number of Characters and Words in turingQuote \n 5) Numbers Array \n 6) Turing Array \n 7) Turing Array with Human in caps. \n 8) Turing Array in alphabetic \n 9) Turing Array but only unique words. \n 10) PyDictionary \n 11) Tuples \n\n")
print("\n")
if userInput=="1":
    print(turingQuote, "\n")
elif userInput=="2":
    print(turingQuote.upper(), "\n")
elif userInput=="3":
    caps = input("Input your value; \n")
    print(caps.upper(), "\n")
elif userInput=="4":
    print("There are: ", characters, " characters in turingQuote.")
    print("There are: ", removeSpecial ," characters in turingQuote without Special Characters.")
    print("There are: ", words, " words in turingQuote.\n")

    print("Alternatively;")
    print(f"There are {words} words, {characters} characters, {removeSpecial} characters without words in turingQuote. \n")
elif userInput=="5":
    array = [7, 0, 14, -3, 81]
    for integer in array:
        print(f"The Integer in position {x} is {integer}")
        x+=1
elif userInput=="6":
    array = turingQuote.split(" ")
    for word in array:
        print(word)
elif userInput=="7":
    array = turingQuote.split(" ")
    for word in array:
        if word == "human" or word=="human.":
            word = word.upper()
        print(word)
elif userInput=="8":
    array = turingQuote.split(" ")
    sortedArray= sorted(array)
    for word in sortedArray:
        print(word, end=' ')
elif userInput=="9":
    lower = turingQuote.lower()
    removeSpecial = lower.replace(".", "")
    array = removeSpecial.split(" ")
    unique_array = []
    for x in array:
        if x not in unique_array:
            unique_array.append(x)
    for x in unique_array:
        print(x, end=" ")

elif userInput=="10":
    lower = turingQuote.lower()
    removeSpecial = lower.replace(".", "")
    array = removeSpecial.split(" ")

    counts = dict()

    for word in array:
        if word in counts:
            counts[word]+=1
        else:
            counts[word]=1
    print(counts)

elif userInput=="11":
    number = input("Insert your number;")
    number = int(number)

    if number==0:
        tuple =tuple((number, "Zero"))
    elif number % 2==0:
        tuple =tuple((number, "Even"))
    else:
        tuple =tuple((number, "Odd"))

    print(tuple)

else:
    print("You have not inputted a valid option. \n")


