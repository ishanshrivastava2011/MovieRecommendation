
class Tree:

    def __init__(self):
        self.data = None
        self.splitVar = None
        self.splitVal = 0
        self.parent = None
        self.label = None
        self.__children = []
        self.nodeLen = 0

    def addChild(self, node):
        self.__children.append(node)

    def getChild(self, index):
        return self.__children[index]
    def isLeaf(self):
        return True if self.__children else False