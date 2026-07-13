
class Phrase:
    def __init__(self,word:str,l:str):
        self.root = word
        self.lang = l

class Expression:
    def __init__(self):
        pass

class Present(Expression):
    def __init__(self):
        pass

class Future(Expression):
    def __init__(self):
        pass

class Past(Expression):
    def __init__(self):
        pass

class Moment(Expression):
    def __init__(self):
        super().__init__()

class Order(Expression):
    def __init__(self):
        super().__init__()

class Subject(Phrase):
    def __init__(self, word, l,person:int=1,plural=False):
        super().__init__(word,l)
        self.subject = (person,plural)

class Verb(Phrase):
    def __init__(self, word, l,exp=Order(),subject:tuple[int,bool]=(1,False)):
        super().__init__(word, l)
        self.expr = exp
        self.subject = subject