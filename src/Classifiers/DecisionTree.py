from collections import defaultdict
from Classifiers.Tree import Tree
import math


class DecisionTree:
    def __init__(self):
        self.tree = Tree
        self.minLeaf = 20.0
        self.globalCount = 0
        self.globalData = dict()

    def __loadData(self, dataFrame):
        tempData = dict()
        for column in dataFrame.columns:
            columnData = dataFrame[column]
            tempData[column] = defaultdict(set)
            for item in list(columnData.iteritems()):
                tempData[column][item[1]].add(item[0])
        self.globalData = tempData

    def __targetEntropy(self, contextPos):
        if not contextPos:
            return 0.0
        target = self.globalData['TARGET']
        e_target = 0
        contextLen = len(contextPos)
        for data in target.items():
            p = len(set.intersection(set(contextPos), set(data[1]))) / contextLen
            if p > 0:
                e_target -= p * math.log2(p)

        return e_target

    def __pickAttribute(self, dataSet):
        entropyList = []
        initPosSet = set()

        for x in dataSet['TARGET'].values():
            for y in x:
                initPosSet.add(y)

        nodeEntropy = self.__targetEntropy(initPosSet)

        for attribute in dataSet:
            if attribute != 'TARGET':
                currentSet = initPosSet.copy()
                distinctValues = [sorted(dataSet[attribute].keys())][0]
                rigtPos = set(currentSet)
                leftPos = set()
                maxSplitList = []
                if len(distinctValues) == 2:
                    leftPos_C = dataSet[attribute][distinctValues[0]]
                    rigtPos_C = dataSet[attribute][distinctValues[1]]
                    # if rigtPos.__len__() == 0:
                    #     break
                    P_left = len(leftPos_C) / len(currentSet)
                    E_left = self.__targetEntropy(leftPos_C)
                    P_right = len(rigtPos_C) / len(currentSet)
                    E_right = self.__targetEntropy(rigtPos_C)
                    gain = nodeEntropy - P_left * E_left - P_right * E_right
                    entropyList.append((0,(attribute, gain)))
                else:
                    for distinctValue in distinctValues:
                        leftPos.update(dataSet[attribute][distinctValue])
                        rigtPos = set(x for x in rigtPos if x not in dataSet[attribute][distinctValue])
                        # if rigtPos.__len__() == 0:
                        #     break
                        P_left = len(leftPos) / len(currentSet)
                        E_left = self.__targetEntropy(leftPos)
                        P_right = len(rigtPos) / len(currentSet)
                        E_right = self.__targetEntropy(rigtPos)
                        gain = nodeEntropy - P_left * E_left - P_right * E_right
                        maxSplitList.append((distinctValue, gain))
                    try:
                        maxGain = max(maxSplitList, key=lambda x: x[1])
                    except:
                        maxGain = 0
                    entropyList.append((attribute, maxGain))
        return max(entropyList, key=lambda x: x[1][1])

    def __splitData(self, splitAttributeTuple, oldDataSet):

        splitVal = splitAttributeTuple[1][0]
        leftPosSet = set()
        rightPosSet = set()
        leftDataSet = dict()
        rightDataSet = dict()

        for data in oldDataSet[splitAttributeTuple[0]].items():
            if data[0] <= splitVal:
                leftPosSet.update(data[1])
            else:
                rightPosSet.update(data[1])

        for attribute in oldDataSet:
            leftDataSet[attribute] = dict()
            rightDataSet[attribute] = dict()
            for entry in oldDataSet[attribute].items():
                leftData = []
                rightData = []
                for data in entry[1]:
                    if data in leftPosSet:
                        leftData.append(data)
                    elif data in rightPosSet:
                        rightData.append(data)
                if leftData:
                    leftDataSet[attribute][entry[0]] = leftData
                if rightData:
                    rightDataSet[attribute][entry[0]] = rightData

        return leftDataSet, rightDataSet

    def __loadTarget(self, y):
        dependent = defaultdict(set)
        for item in list(y.iteritems()):
            dependent[item[1]].add(item[0])
        self.globalData['TARGET'] = dependent
        self.globalCount = sum([len(x[1]) for x in self.globalData['TARGET'].items()])

    def __buildTree(self, parent, nodeData):
        tree = Tree()
        tree.data = nodeData
        tree.parent = parent
        tree.label = max([(x[0], len(x[1])) for x in nodeData['TARGET'].items()],key=lambda y:y[1])[0]
        tree.nodeLen = sum([len(x[1]) for x in nodeData['TARGET'].items()])

        if tree.nodeLen > self.globalCount*(self.minLeaf/100.0):
            attrTuple = self.__pickAttribute(nodeData)
            tree.splitVal = attrTuple[1][0]
            tree.splitVar = attrTuple[0]
            if attrTuple[1][1] >= 0.00001:
                leftData, rightData = self.__splitData(attrTuple, tree.data)

                leftChild = self.__buildTree(tree, leftData)
                tree.addChild(leftChild)

                rightChild = self.__buildTree(tree, rightData)
                tree.addChild(rightChild)

        return tree

    def fit(self, X, y):
        self.__loadData(X)
        self.__loadTarget(y)
        self.tree = self.__buildTree(None, self.globalData)
        print()

    def traverse(self, data, node, targetList, indexList, start=0):
        if not node.isLeaf():
            for i in indexList:
                targetList[i-start] = node.label
        else:
            splitColumn = data[node.splitVar]
            splitVal = node.splitVal
            leftList = []
            rightList = []
            for i in indexList:
                if splitColumn[i] <= splitVal:
                    leftList.append(i)
                else:
                    rightList.append(i)
            self.traverse(data, node.getChild(0), targetList, leftList, start)
            self.traverse(data, node.getChild(1), targetList, rightList, start)




    def predict(self, X):
        targetList = ["Unknown" for i in range(len(X))]
        indexList = [x[0] for x in list(X.itertuples())]
        y = self.traverse(X, self.tree, targetList, indexList, indexList[0])
        return targetList



