import numpy as np
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],     #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet, labels                             #返回数据集和分类属性
# a,_ = createDataSet()
# a = np.array(a)
# print(np.shape(a))
def trainpbmodel_x(feature:np.array([])) ->dict:
        N,D = np.shape(feature)
        model = {}
        for d in range(D):
                model[d]={}
                label = feature[:,d]
                # print(label)
                key = set(label)

                for k in key:
                        # print(np.where(label==k)[0].shape[0])
                        model[d][int(k)]=float(np.where(label==k)[0].shape[0]/N)
        return model
def trainpb_model(dataSet:list,label:list)->dict:
        model={}
        labs = set(label)
        #P(Y)
        for lab in labs:
                model[lab]={}
                PY=label.count(lab)/len(label)
        # 获得 P(X|Y)
                index = np.where(np.array(label)==lab)[0].tolist()
                # print(index)
                feats = np.array(dataSet)[index]
                # print(feats)
                PX = trainpbmodel_x(feats)
                model[lab]['PY']=PY
                model[lab]['PX']=PX
        return model

def getPbfromMODEL(feats:np.array([]),model:dict,keys:set) -> dict:
        eps = 0.00001
        result ={}
        for key in keys:
                PX = model.get(key,eps).get('PX',eps)
                PY = model.get(key, eps).get('PY', eps)
                pbx = []
                for fea in range(len(feats)):
                        # print(PX.get(fea,eps).get())
                        pbx.append(PX.get(fea,eps).get(feats[fea],eps))
                # print(PY)
                # print(pbx)
                # print(np.log(pbx))
                result[key]=np.log(PY)+np.sum(np.log(pbx),axis=0)

        return result
if __name__ == '__main__':
        dataSet,_ = createDataSet()
        data = np.array(dataSet)[:,:-1].tolist()
        keys = np.array(dataSet)[:,-1].tolist()


        # key = set(keys)
        model = trainpb_model(data,keys)
        print(model)
        features = np.array([0,1,0,1])
        # print(features[0])
        result = getPbfromMODEL(features,model,set(keys))
        print(result)
        for k,v in result.items():
                if v == max(result.values()):
                        print("预测结果是",k)


