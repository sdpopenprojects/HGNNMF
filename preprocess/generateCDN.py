import os
import pandas as pd
import javalang
import pandas as pd
import json
from gensim.models.word2vec import Word2Vec

node_id = []
node_className = []
node_metricsType = []
node_ast_wpdp = []
node_ast_cpdp = []
node_label = []

edge_srcName = []
edge_srcId = []
edge_dstName = []
edge_dstId = []


def getCDNClassInfo(cn, classNames):
    """Search for the complete class name and id number in the class list

    Args:
        cn (_type_): _description_
        classNames (_type_): _description_

    Returns:
        _type_: _description_
    """
    for idx, cn_ in enumerate(classNames):
        if cn_.find(cn) >= 0:
            return idx, cn_
    return -1, ""


def getDependClass(filePath=None) -> list:
    if filePath == None:
        return []

    dependList = []

    with open(filePath, "r", encoding="utf-8") as file:
        sourceData = file.read()
        tree = javalang.parse.parse(sourceData)
        for _, node in tree:
            if node.__class__.__name__ == "Import":
                dependList.append(node.path)
    return dependList


def readRelationalData(projectName, projectVersion, fileBasePaths):
    traditionDatas = pd.read_csv(
        "data/tradition/{}/{}.csv".format(projectName, projectVersion)
    )

    projectsVoc = {}

    for row in traditionDatas.itertuples():
        fileName = row[1]

        filePath = None

        for baseDir in fileBasePaths:
            baseDir = baseDir.format(projectName, projectVersion)

            tmp = os.path.join(
                baseDir,
                (
                    fileName.replace(".", "/") + ".java"
                    if fileName.rfind(".java") == -1
                    else fileName
                ),
            )

            if os.path.exists(tmp):
                filePath = tmp
                break

        if filePath != None:
            projectsVoc[fileName] = getDependClass(filePath)

    return projectsVoc


def fileToASTWPDP(maxVecLen, projectName, projectVersion, filePath=None) -> list:
    if filePath == None:
        return [0.0] * maxVecLen

    vecList = []
    w2cModel = Word2Vec.load(
        "data/build/w2c_{}_{}.bin".format(projectName, projectVersion)
    )

    with open(filePath, "r", encoding="utf-8") as f:
        try:
            treeAST = javalang.parse.parse(f.read())
            for _, node in treeAST:
                if node.__class__.__name__ == "MethodInvocation":
                    tmp = "{}_{}".format(node.__class__.__name__, node.member)
                    if tmp in w2cModel.wv:
                        vecList.append(w2cModel.wv[tmp][0])
                elif node.__class__.__name__ in [
                    "MethodDeclaration",
                    "EnumDeclaration",
                    "TypeDeclaration",
                ] and hasattr(node, "name"):
                    tmp = "{}".format(node.__class__.__name__)
                    if tmp in w2cModel.wv:
                        vecList.append(w2cModel.wv[tmp][0])
                elif node.__class__.__name__ in [
                    "IfStatement",
                    "WhileStatement",
                    "DoStatement",
                    "ForStatement",
                    "BreakStatement",
                    "ContinueStatement",
                    "ThrowStatement",
                    "TryStatement",
                    "SwitchStatement",
                ]:
                    tmp = node.__class__.__name__
                    if tmp in w2cModel.wv:
                        vecList.append(w2cModel.wv[tmp][0])
        except Exception as e:
            print(e)
    return vecList


def fileToASTCPDP(maxVecLen, projectName, projectVersion, filePath=None) -> list:
    if filePath == None:
        return [0.0] * maxVecLen

    vecList = []
    w2cModel = Word2Vec.load(
        "data/build/W2C_{}_{}.bin".format(projectName, projectVersion)
    )

    with open(filePath, "r", encoding="utf-8") as f:
        try:
            treeAST = javalang.parse.parse(f.read())
            for _, node in treeAST:
                if node.__class__.__name__ in [
                    "MethodDeclaration",
                    "EnumDeclaration",
                    "TypeDeclaration",
                ] and hasattr(node, "name"):
                    tmp = "{}".format(node.__class__.__name__)
                    if tmp in w2cModel.wv:
                        vecList.append(w2cModel.wv[tmp][0])
                elif node.__class__.__name__ in [
                    "MethodInvocation",
                    "IfStatement",
                    "WhileStatement",
                    "DoStatement",
                    "ForStatement",
                    "BreakStatement",
                    "ContinueStatement",
                    "ThrowStatement",
                    "TryStatement",
                    "SwitchStatement",
                ]:
                    tmp = node.__class__.__name__
                    if tmp in w2cModel.wv:
                        vecList.append(w2cModel.wv[tmp][0])
        except Exception as e:
            print(e)

    return vecList


def getAST(projectName, projectVersion, fileName, isWPDP, projects):
    filePath = None
    for baseDir in projects[projectName]["codefilepaths"]:
        baseDir = baseDir.format(projectName, projectVersion)

        tmp = os.path.join(
            baseDir,
            (
                fileName.replace(".", "/") + ".java"
                if fileName.rfind(".java") == -1
                else fileName
            ),
        )

        if os.path.exists(tmp):
            filePath = tmp
            break
    if filePath == None:
        return []

    return (
        fileToASTWPDP(
            projects[projectName]["maxVecLen"],
            projectName,
            projectVersion,
            filePath,
        )
        if isWPDP
        else fileToASTCPDP(
            projects[projectName]["maxVecLen"],
            projectName,
            projectVersion,
            filePath,
        )
    )


def cdnToCsv(projectName, projectVersion, fileBasePaths, projects):
    global node_id, node_className, node_metricsType, node_ast_wpdp, node_ast_cpdp, node_label, edge_srcName, edge_srcId, edge_dstName, edge_dstId
    traditionalData = pd.read_csv(
        "data/tradition/{}/{}.csv".format(projectName, projectVersion)
    )

    relationalData = readRelationalData(projectName, projectVersion, fileBasePaths)

    tmpNodeId = 0
    for row in traditionalData.itertuples():
        if row[1].find("/") >= 0 and not row[1].endswith(".java"):
            continue

        node_id.append(tmpNodeId)
        tmpNodeId += 1
        node_className.append(row[1])

        if row[1].endswith(".java"):
            node_metricsType.append([float(i) if i != "-" else 0.0 for i in row[2:-8]])
        else:
            node_metricsType.append([float(i) if i != "-" else 0.0 for i in row[2:-1]])

        node_ast_wpdp.append(
            getAST(projectName, projectVersion, row[1], True, projects)
        )
        node_ast_cpdp.append(
            getAST(projectName, projectVersion, row[1], False, projects)
        )
        node_label.append(0 if int(row[-1]) == 0 else 1)

    for k, v in relationalData.items():
        tmp = None

        if k.endswith(".java"):
            tmp = k[:-5]
            tmp = tmp.replace("/", ".")

        if len(v) > 0:
            for dstName in v:
                if node_className[0].endswith(".java"):
                    tmp = dstName.replace(".", "/") + ".java"
                else:
                    tmp = dstName

                id, completeName = getCDNClassInfo(tmp, node_className)
                if id != -1:
                    edge_srcName.append(k)
                    edge_srcId.append(node_id[getCDNClassInfo(k, node_className)[0]])
                    edge_dstName.append(completeName)
                    edge_dstId.append(node_id[id])


def writeData():
    global node_id, node_className, node_metricsType, node_ast_wpdp, node_ast_cpdp, node_label, edge_srcName, edge_srcId, edge_dstName, edge_dstId

    dataFrame = pd.DataFrame(
        {
            "node_id": node_id,
            "node_className": node_className,
            "node_metrics": node_metricsType,
            "node_ast_wpdp": node_ast_wpdp,
            "node_ast_cpdp": node_ast_cpdp,
            "node_label": node_label,
        }
    )
    dataFrame.to_csv(
        ("./data/graph/cdn/raw/{}/{}/nodes.csv").format(projectName, projectVersion),
        index=False,
        sep=",",
    )

    dataFrame = pd.DataFrame(
        {
            "edge_srcName": edge_srcName,
            "edge_srcId": edge_srcId,
            "edge_dstName": edge_dstName,
            "edge_dstId": edge_dstId,
        }
    )
    dataFrame.to_csv(
        ("./data/graph/cdn/raw/{}/{}/edges.csv").format(projectName, projectVersion),
        index=False,
        sep=",",
    )


if __name__ == "__main__":
    with open("config/project_new.json", "r") as f:
        projects = json.load(f)
    for projectName, project in projects.items():
        for projectVersion in project["versions"]:
            node_id = []
            node_className = []
            node_metricsType = []
            node_ast_wpdp = []
            node_ast_cpdp = []
            node_label = []

            edge_srcName = []
            edge_srcId = []
            edge_dstName = []
            edge_dstId = []

            cdnToCsv(
                projectName,
                projectVersion,
                projects[projectName]["codefilepaths"],
                projects,
            )
            writeData()

            print("project--{}--{} done!".format(projectName, projectVersion))
