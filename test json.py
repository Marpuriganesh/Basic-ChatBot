
from cgi import test
import pandas as pd

dic = {
    1:[1,2,3,4,5],
    2:['a','b','c','d','e']
}

print(dic)
test = pd.DataFrame(dic)
print(test)

test.to_json("test2.json")