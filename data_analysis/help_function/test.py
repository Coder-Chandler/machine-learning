
Pclass= input("enter Pclass: ")
Sex= input("enter Sex: ")
Age= input("enter Age: ")
SibSp= input("enter SibSp: ")
Parch= input("enter Parch: ")
Fare= input("enter Fare: ")
Embarked= input("enter Embarked: ")

def test(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    x=(-0.1673305943595034)*float(Pclass)
    y=(-0.52222839164542101)*float(Sex)
    z=(-0.0053560327484279725)*float(Age)
    a=(-0.041643391854120811)*float(SibSp)
    b=(0.0054941757863996594)*float(Parch)
    c=(-0.00033548031800063698)*float(Fare)
    d=0.045317312526614133*float(Embarked)
    return a+b+c+d+x+y+z+1.27831789091
print(test(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked))
