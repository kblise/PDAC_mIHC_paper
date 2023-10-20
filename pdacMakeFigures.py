#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code author: Katie E. Blise
Date: October 2023

This .py file contains all code needed to reproduce the figures in the paper:
"Machine learning links T cell function and spatial localization to neoadjuvant immunotherapy and clinical outcome in pancreatic cancer"


This .py file contains multiple functions:
    
    ***FUNCTIONS FOR ML MODELS***
    histoFeatures() = subset feature matrices to regions of specified histopathologic site(s)
    normalizeDensity() = load and normalize cell state density features
    normalizeInteractions() = load and normalize spatial interaction features
    normalizeBarcodes() = load and normalize T cell functionality barcode features
    setPositive() = set positive vs negative ML class for every observation
    classify() = train and test ML elastic net classifier
    performance() = evaluate ML classifier performance
    featImportance() = SHAP values and feature importance analyses
    pipeline() = run above ML classifier functions at once
    makeClassifierFiegures() = construct multiple ML models for each designated histopathologic site and compare results
    makeBarcodeMatrix() = correlate top T cell barcodes from ML models based on the T cell states expressing the barcodes
    
    ***FUNCTIONS FOR RCN ANALYSIS***
    makeNeighborhoods() = calculate spatial neighborhoods for every cell across specified regions
    elbowMethod() = run elbow method to determine optimal number of clusters (RCNs)
    clusterNeighborhoods() = cluster spatial neighborhoods based on cellular composition into RCNs
    clusterCountFrequency() = calculate frequency of RCNs present across data
    createCsvsWithClusterCol() = update region mIHC csvs with added cluster column to denote each cell's RCN assignment
   
    ***FUNCTIONS TO GENERATE ALL RESULTS (which call to the above functions)***
    fig1() = generates Figure 1C
    fig2() = generates Figures 2A-2C, Supplementary Tables S3-S5
    fig3() = generates Figures 3A-3C, 3E, Supplementary Figures S2A-S2E
    fig4() = generates Figures 4A-4C, 4E, Supplementary Figures S3A, S3B
    fig5() = generates Figures 5B-5H, Supplementary Figures S4A-S4F
   
Note: This program assumes the following items live in the same directory as this .py file:
    - 'data' folder, which houses 2 folders:
        -'mIHC_files' folder, which houses all mIHC data (.csv files)
        -'metadata' folder, which houses clinical and tissue data
    -'results' folder, which houses 3 folders:
        -'dfCreated' folder, which will store dataframes created by this code, and houses 1 folder:
            -'updatedCsvs' folder within 'dfCreated' folder, which will store revised mIHC csv files with RCN assignments
        -'figures' folder, which will store figures created by this code, and houses 1 folder:
            -'figureS4F' folder within 'figures' folder, which will store all scatterplots for Supp Fig S4F
        -'tables' folder, which will store tables created by this code


This program is intended for Python version 3.
"""
 


def histoFeatures(df,areaList):
    '''
    This function is used in figs 3&4 to create separate ML models for each histopath site.
    Input parameters:
        path = cwd
        areaList = list of histopath sites to subset to
   Outputs:
       returns:
           df = a subset of the dataframe with just the regions that match the desired histopath sites
   '''
   
    import re
    
    #pattern to match histopath
    pattern = re.compile(r".*_.*_(.*)_.*")

    #stores ROIs to drop
    dropList = []

    for roi in df.index:
        hist = re.match(pattern,roi).group(1)
        if hist not in areaList:
            dropList.append(roi)
    df = df.drop(dropList)
    
    #return the df with all ROIs that match the desired histopath type
    return df



def normalizeDensity(path,areaList):
    '''
    This function is used in figs 3&4 to read in cell state densities and log10 normalize features for ML models
    Input parameters:
        path = cwd
        areaList = list of histopath sites to subset to
   Outputs:
       returns df with cell state density features
    '''    

    import numpy as np
    import pandas as pd
    
    #read density file
    df = pd.read_csv(path+'/results/dfCreated/dfDensity.csv',index_col=0)

    #subset to just desired histopath site
    df = histoFeatures(df,areaList)

    #log10+1 normalize feature values
    df = np.log10(df+1)

    #now update column names to include 'density'
    newColMapDict = {}
    for col in df.columns:
        newCol = 'Density: '+col
        newColMapDict[col] = newCol

    df = df.rename(columns=newColMapDict)

    return df


   
def normalizeInteractions(path,areaList):
    '''
    This function is used in figs3&4 to read in spatial interactions and log10 normalize features for ML models
    Input parameters:
        path = cwd
        areaList = list of histopath sites to subset to
   Outputs:
       returns df with spatial interaction features
    '''    
    
    import numpy as np
    import pandas as pd
    import re

    #read interaction file
    df = pd.read_csv(path+'/results/dfCreated/dfInteractions.csv',index_col=0)

    #subset to just desired histopath site
    df = histoFeatures(df,areaList)

    #normalize first by density of cells involved in interaction, divide raw count by summed density of cell types
    dfAll = pd.read_csv(path+'/results/dfCreated/dfDensity.csv',index_col=0)

    #empty df to store normalized interaction values
    dfNormI = pd.DataFrame()

    #divide each interaction count by summed densities of cells types involved
    pattern1 = re.compile(r"(.*_.*)_.*_.*_..")
    pattern2 = re.compile(r".*_.*_(.*_.*)_..")

    for col in df.columns:
        cell1 = re.match(pattern1,col).group(1)
        cell2 = re.match(pattern2,col).group(1)

        for roi in df.index:
            #get summed densities for that region
            dens1 = dfAll.loc[roi,cell1]
            dens2 = dfAll.loc[roi,cell2]

            if cell1 != cell2: #if not the same cell types, then add densities
                tot = dens1 + dens2
            elif cell1 == cell2: #if same cell types, just use one of the densities
                tot = dens1

            #normalize = interactions/density total
            ints = df.loc[roi,col]
            if ints == 0:
                normI = 0
            else:
                normI = ints/tot

            #add roi,normalized value to new df
            dfNormI.loc[roi,col] = normI
        
    #now log10+1 normalize
    dfNormI = np.log10(dfNormI+1)
    
    #now update column names to include 'interaction' - patterns defined earlier
    newColMapDict = {}
    for col in dfNormI.columns:
        cell1 = re.match(pattern1,col).group(1)
        cell2 = re.match(pattern2,col).group(1)
        newCol = 'Interaction: '+cell1+' & '+cell2
        newColMapDict[col] = newCol

    dfNormI = dfNormI.rename(columns=newColMapDict)

    return dfNormI



def normalizeBarcodes(path,areaList):
    '''
    This function is used in figs3&4 to read in T cell functionality barcodes and log10 normalize features for ML models
    Input parameters:
        path = cwd
        areaList = list of histopath sites to subset to
    Outputs:
        returns df with T cell functionality barcode features
    '''    
     
    import numpy as np
    import pandas as pd
      
    #read T functionality barcode file
    df = pd.read_csv(path+'/results/dfCreated/dfBarcodeDensity.csv',index_col=0)
    #adjust index to match other dfCreated files
    df = df.set_index('Unnamed: 1')
       
    #subset to just desired histopath site
    df = histoFeatures(df,areaList)
    
    #log10+1 normalize features
    df = np.log10(df+1)

    #create function to translate barcode into new columns
    def barTranslate(b):
        # barcode order = 'PD1+_TOX+_TIM3+_LAG3+_CD39+_EOMES+_CD38+_CD44+_TCF1/7+_TBET+'
        barcodeKeyDict = {0:'PD1+',1:'TOX+',2:'TIM3+',3:'LAG3+',4:'CD39+',5:'EOMES+',6:'CD38+',7:'CD44+',8:'TCF1/7+',9:'TBET+'}
        #translate barcode to proteins
        barcode = b.replace('_', '') #drop _ in barcode
        #start with empty string to add to
        markerString = ''
        for i in range(len(barcode)):
            binary = barcode[i] #check if it's a 0 or 1
            if binary == '1': #if it's 1, get the correspoinding marker at that index
                marker = barcodeKeyDict[i]
                markerString = markerString+' '+marker
        if markerString == '': #if no markers are positive
            markerString = 'Negative for all'
        return markerString #return the translated barcode
    
    #now update column names to include 'Barcode' and translate the barcode
    newColMapDict = {}
    for col in df.columns:
        barT = barTranslate(col) #call translate function
        newCol = 'Barcode: '+barT
        newColMapDict[col] = newCol
    
    df = df.rename(columns=newColMapDict)
    
    return df
    


def setPositive(path,dfAvgNorm,figNum):
    '''
   This function is used in figs3&4 to set the appropriate regions to be the positive or negative ML class
   Input parameters:
       path = cwd
       dfAvgNorm = dataframe with all features for all regions inputted into the ML model
       figNum = figure number - either 'fig3' or 'fig4'; this affects which class is positive
   Outputs:
       returns:
       dfAvgNorm = updated dataframe with ML classes listed
       ptIdxList = patient indices; for splitting train/test to prevent data leakage
       mlIdxList = ml class indices, for splitting train/test to prevent data leakage
    '''
    
    import pandas as pd
    import numpy as np
    import re
    
    #add target value to each observation
    #read in clinical file
    dfClin = pd.read_csv(path+'/data/metadata/clinicalData.csv',index_col=0)

    
    #figure 3: tx classifier: set positive label in ML classifier to be aCD40-tx patients
    if figNum == 'fig3':
        dfClin['MLclass'] = np.where(dfClin['tx'] == 'aCD40',1,0) #0=naive=neg; 1=aCD40=pos
       
    #figure 4: aCD40 dfs classifier: set positive label in ML classifier to be short DFS of just aCD40 cohort
    elif figNum == 'fig4':
        #subset to just desired aCD40 cohort
        dfClin = dfClin[dfClin['tx'] == 'aCD40'] 
        #add long/short survival
        med = dfClin['DFS'].median() #get median of cohort survival - based on pt average in dfClin
        #set positive label in ML classifier to be short survivors
        dfClin['MLclass'] = np.where(dfClin['DFS'] <= med,1,0) #0=long survival, 1=short survival

    #now get pt list and ml class list and add class to dfAvgNorm
    patternP = re.compile(r"(.*)_.*_.*_.*") #patient ID
    ptList = []

    for file in dfAvgNorm.index:
        pt = re.match(patternP,file).group(1)
        ptList.append(pt)

    #add column for patient
    dfAvgNorm['pt'] = ptList
    #merge on pt to get the ML class
    dfAvgNorm = dfAvgNorm.merge(dfClin['MLclass'],left_on='pt',right_on='sample')
    #get patient order with idx for classify() to prevent data leakage when splitting train/test
    dfAvgNorm2 = dfAvgNorm.groupby('pt').mean()
    ptIdxList = list(dfAvgNorm2.index) #return this
    mlIdxList = list(dfAvgNorm2['MLclass']) #return this

    return dfAvgNorm,ptIdxList,mlIdxList

    

def classify(dfAvgNorm,ptIdxList,mlIdxList,kNeigh,cols):
    '''
    This function is used in figs3&4 to build the ML model using a leave-one-patient-out approach
    Input parameters:
        dfAvgNorm = dataframe to construct ML model from
        ptIdxList = list of patient IDs corresponding to each region in the df
        mlIdxList = list of ML classes corresponding to each region in the df
        kNeigh = used for SMOTE balancing of train set; default is 5 but must set to 3 when looking at NAP site because of small dataset
        cols = list of columns in order in dfAvgNorm
    Outputs:
        returns:
            dfPredictTest = predictions for test set
            shapTestArray = SHAP values for test set
            featTestArray = feature names for test set, in same order as shap values
            dfFeatTest = dataframe of test set features with corresponding ML class
    '''
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import MinMaxScaler
    import shap
    from shap.maskers import Independent    

    #first randomly sort dfAvgNorm, starting at same spot w seed for reproducibility
    dfAvgNorm = dfAvgNorm.sort_index() #first sort on index so every run starts the same
    dfAvgNorm = dfAvgNorm.sample(frac=1,random_state=123) #randomly shuffle the rows, same way each run so you can compare

    #separate into features and target
    xAll = dfAvgNorm.values[:,:-2] #don't include the pt column if using all regions
    ptList = list(dfAvgNorm['pt']) #get list of patients in order they appear in dfAvgNorm; store in list for indexing
    yAll = dfAvgNorm['MLclass'].values

    #define CV method: LOO cross-validator
    splitter = LeaveOneOut()

    #define [0,1] minmax scaler
    scaler = MinMaxScaler()

    #define balancing method: synthetic upsampling: imblearn smote
    balancer = SMOTE(sampling_strategy=1,k_neighbors=kNeigh,random_state=123) #synthetic upsampling of minority class; kNeigh default=5; set random seed to compare

    #create ML classifier model: logistic regression model w/ EN penalty
    model = LogisticRegression(penalty='elasticnet', solver='saga',l1_ratio=0.5,max_iter=5000,random_state=4) #EN classifier model; set random seed to compare

    a = 0 #need to count to do shap arrays; a=cv loop number

    #create empty lists to store TEST values
    aTestList = [] #tells you the split number
    patientTestList = [] #patient ID
    roiTestList = [] #counts number of ROIs per patient
    trueTestList = [] #true label
    predictTestList = [] #predicted label
    confidenceTestList = [] #confidence score for each prediction
    
    #loop through CV splits - leave one patient out; ptIdxList and mlIdxList come from setPositive()
    for indexTrain, indexTest in splitter.split(ptIdxList,mlIdxList):

        #need to convert patient indices to region indices while making sure patients don't spill between train/test
        #empty lists to store indices to all ROIs for the train and test sets; resets for each split/repeat in the CV loop
        idxTrain = []
        idxTest = []

        #match idx for patient with idx for all regions of that patient
        #get TRAIN indices for all regions
        for iTrain in indexTrain:
            pt = ptIdxList[iTrain] #get patient that corresponds to that index
            idxTrain = idxTrain + [i for i, p in enumerate(list(ptList)) if p==pt] #get indices of all ROIs for that patient in train

        #get TEST indices for all regions
        for iTest in indexTest:
            pt = ptIdxList[iTest] #get patient that corresponds to that index
            idxTest = idxTest + [i for i, p in enumerate(list(ptList)) if p==pt] #get indices of all ROIs for that patient in test

        #get data for train/test feat/targ splits using the indices from the splitter.split()
        xTrain, yTrain = xAll[idxTrain], yAll[idxTrain] #training data
        xTest, yTest = xAll[idxTest], yAll[idxTest] #testing data

        #SCALE the train, test data
        #fit on train data, then apply that fitted scaler on train, and then test data
        scalerFit = scaler.fit(xTrain)
        xTrainScaled = scalerFit.transform(xTrain)
        xTestScaled = scalerFit.transform(xTest)        
        
        #Clip outliers in test set: if test has any outliers (aka greater than 1 or less than 0), set them to 1 or 0
        #note that train is scaled [0,1] so no clipping needed for train
        xTestScaled[xTestScaled > 1] = 1
        xTestScaled[xTestScaled < 0] = 0    

        #Balance Train set only if it is unbalanced
        uniqueTrain, countsTrain = np.unique(yTrain,return_counts=True) #get counts per class in the train set
        if abs(countsTrain[0]-countsTrain[1]) > 0:
            #balance training set using SMOTE method defined above
            xTrainBalanced, yTrainBalanced = balancer.fit_resample(xTrainScaled,yTrain)

        #otherwise if classes are already balanced, then don't balance but rename for code
        else:  
            xTrainBalanced = xTrainScaled
            yTrainBalanced = yTrain

        #fit model to balanced training set
        modelFit = model.fit(xTrainBalanced, yTrainBalanced)

        #run prediction on TRAIN
        #predict_train = modelFit.predict(xTrainBalanced)

        #run prediction on TEST [not balanced bc it is real world data and don't want to alter]
        predict_test = modelFit.predict(xTestScaled)

        #store all TEST results
        aTestList += len(predict_test) * [a] #cv loop
        patientTestList += list(dfAvgNorm['pt'].iloc[idxTest]) #patient ID
        roiTestList += len(predict_test) * [len(predict_test)] #get count of how many ROIs for that pt
        trueTestList += list(yTest) #ground truth
        predictTestList += list(predict_test) #predicted classes
        confidenceTestList += list(modelFit.decision_function(xTestScaled)) #confidence scores

        #SHAP VALUES for each split; store train/test shap values separately
        #first create the background masker based on scaled/balanced train set
        #then create the explainer using the training masker
        masker = Independent(data=xTrainBalanced) #shap.maskers.Independent = True to model [okay when using EN model]
        explainer = shap.LinearExplainer(model=modelFit,masker=masker,feature_names=cols)

        #train shap
        #shapValsTrain = explainer(xTrainBalanced)

        #test shap
        shapValsTest = explainer(xTestScaled)

        #store test shap values
        #if this is the first cv split, create new arrays
        if a == 0:
            shapTestArray = np.array(shapValsTest.values)
            featTestArray = np.array(shapValsTest.data)

        #if not the first cv split, stack on existing array
        else:
            shapTestArray = np.vstack((shapTestArray,shapValsTest.values))
            featTestArray = np.vstack((featTestArray,shapValsTest.data))

        #a impacts stacking np arrays; used to initialize first np array and count the CV loop
        a = a + 1

    #TEST dfPredict
    dfPredictTest = pd.DataFrame()
    dfPredictTest['split'] = aTestList
    dfPredictTest['patient'] = patientTestList
    dfPredictTest['rois'] = roiTestList
    dfPredictTest['True Label'] = trueTestList
    dfPredictTest['Predicted Label'] = predictTestList
    dfPredictTest['Confidence Score'] = confidenceTestList
    dfPredictTest['Match'] = np.where(dfPredictTest['True Label'] == dfPredictTest['Predicted Label'],'Good','Bad')
    
    #to store directionality of feature predictions - for later bubble plot
    dfFeatTest = pd.DataFrame(featTestArray,columns=cols)
    dfFeatTest['MLclass'] = trueTestList
    
    return dfPredictTest, shapTestArray, featTestArray, dfFeatTest
        

    
def performance(path,dfPredictTest,histo,figNum):
    '''
    This function is used in figs3&4 to calculate performance metrics
    Input parameters:
        path = cwd
        dfPredictTest = test set predictions and ground truth
        histo = which histopath site the model was derived from
        figNum = either 'fig3' or 'fig4'
    Outputs:
        returns:
            perfList = list of accuracy and f1 score for test set 
        Saves plots to 'figures' folder for figure 3B or 4B    
    '''
    
    from sklearn import metrics
    import matplotlib.pyplot as plt
    
    perfList = [] #empty list to store test [accuracy, f1]

    #TEST perf: accuracy, f1
    acc = metrics.accuracy_score(dfPredictTest['True Label'], dfPredictTest['Predicted Label'])
    f1 = metrics.f1_score(dfPredictTest['True Label'], dfPredictTest['Predicted Label'],pos_label=1)  
    
    ##ROC CURVE & AUC
    #don't make ROC curve for poor aCD40 DFS performing models (T, TAS)
    if (figNum == 'fig4') and (histo != 'IA'):
        pass
        
    else:
        fpr,tpr,thresh = metrics.roc_curve(dfPredictTest['True Label'], dfPredictTest['Confidence Score'],pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('ROC Curve on Test Dataset')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, zorder=1, clip_on=False)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(path+'/results/figures/figure'+figNum[-1]+'B_'+histo+'.png',format='png')
        plt.close()

    #add test perfs to perfList to return at the end
    perfList = [acc,f1]
        
    return perfList



def featImportance(path,shapTestArray,featTestArray,cols,histo,dfAvgNorm,figNum):
    '''
    This function is used in figs3&4 to calculate feature importance
    Input parameters:
        path = cwd
        shapTestArray = SHAP values for test set
        featTestArray = feature names for test set, in same order as shap values
        cols = columns of inputted df into ML model
        histo = histopath site model is derived from
        dfAvgNorm = inputted dataframe used to construct ML model
        figNum = either 'fig3' or 'fig4'
    Outputs:
        returns:
            dfShap = dataframe of shap values
        Saves plots to 'figures' folder for figure S2A or S3A, S2B-S2E or S3B
        Prints % importance of top 30 features based on their SHAP values contributing to ML model's predictions
        Prints p-values for top 15 features
    '''
    
    import plotly.express as px
    import numpy as np
    import pandas as pd
    import shap
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection
    import matplotlib.pyplot as plt
    import warnings
   
    #safe to ignore warnings for these analyses; relates to verison issues w matplotlib/shap and plot formatting
    warnings.filterwarnings("ignore", message="No data for colormapping provided via 'c'. Parameters 'vmin', 'vmax' will be ignored")
    warnings.filterwarnings("ignore", message='Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.')

    #test shap values; show top 30
    shap.summary_plot(shapTestArray,features=featTestArray,feature_names=cols,max_display=30,show=False)

    #save shap plots - https://github.com/shap/shap/issues/153
    fig = plt.gcf()
    fig.savefig(path+'/results/figures/figureS'+str(int(figNum[-1])-1)+'A'+'_'+histo+'.png',format='png',bbox_inches='tight')
    plt.gca().set_aspect('equal')
    ##note: this figure causes warnings to appear related to the figure appearance - these are package version issues btwn shap/matplotlib; leaving as is-results are correct
    plt.close(fig)    
        
    #put test shap value feature names into df for further exploration
    #helpful: https://github.com/slundberg/shap/issues/632
    vals = np.abs(shapTestArray).mean(0)
    dfShap = pd.DataFrame(list(zip(cols,vals)),columns=['Feature','Test Shap Value'])
    dfShap.sort_values(by=['Test Shap Value'],ascending=False,inplace=True)
    
    #calculate and print shap value %s - aka how much importance the top 30 features / all features comprise
    topSum = dfShap.iloc[0:30,1:2].sum() #get top 30 features and their test shap value
    allSum = dfShap.iloc[:,1:2].sum() #get all shap values added together
    print('% Importance of top 30 features: '+str(int((topSum/allSum*100)))+'%') #print % (out of 100) of shap value (impact on model output)
    
    
    #box plot for Figure SB-E
    featList = list(dfShap['Feature'].values[0:15])
    featList.append('MLclass')
    featList.append('pt')
    dfTop = dfAvgNorm[featList].copy()
    
    
    if figNum == 'fig3':
        dfTop['Class'] = np.where(dfTop['MLclass'] == 0,'Naive','aCD40')
        colorSeq = {'Naive':'#ff7f0e','aCD40':'#1F77B4'}
        figDict = {'T':'B','IA':'C','TAS':'D','NAP':'E'} #dict to save figure number

    elif figNum == 'fig4':
        dfTop['Class'] = np.where(dfTop['MLclass'] == 0,'Long DFS','Short DFS')
        colorSeq = {'Short DFS':'#990099','Long DFS':'#109618'}
        figDict = {'IA':'B'} #dict to save figure number

        
    fig = px.box(dfTop,y=dfTop.columns[0:15],color='Class',points='all',labels={'variable':'Top Features','value':'Log10+1 Normalized Feature Value'},color_discrete_map=colorSeq) #,height=400,width=500 #add for making figures
#     fig.update_xaxes(visible=False, showticklabels=False) #toggle for making figures
    fig.write_image(path+'/results/figures/figureS'+str(int(figNum[-1])-1)+figDict[histo]+'.png')
    
    #run stats on diffs between 2 classes (naive vs aCD40; long vs short dfs) - mann-whitney u test & benjamini-hochberg correction
    #split dfTop into 2 classes
    if figNum == 'fig3':
        df1 = dfTop[dfTop['Class'] == 'Naive']
        df2 = dfTop[dfTop['Class'] == 'aCD40']
  
    elif figNum == 'fig4':
        df1 = dfTop[dfTop['Class'] == 'Short DFS']
        df2 = dfTop[dfTop['Class'] == 'Long DFS']        

    pList = [] #MWU
    colList = []

    for col in df1.columns[0:15]:        
        pVal = mannwhitneyu(df1.loc[:,col],df2.loc[:,col],alternative='two-sided')[1] #two-tail independent, not normal distributions
        pList.append(pVal)
        colList.append(col)

    mhtList = fdrcorrection(pList,alpha=0.05)[1]

    print('\nMHT-corrected P-values for Figure S'+str(int(figNum[-1])-1)+figDict[histo]+':')
    for i in range(len(colList)):
        feature = colList[i]
        mhtP = mhtList[i]
        print(feature,': ',round(mhtP,6)) #print rounded p value for each of the top features
    
    return dfShap



def pipeline(path,areaList,figNum,kNeigh,histo):
    '''
    This function is used in figs3&4 to run all functions to construct feature dataframe, build ML model, evaluate performance, and calculate SHAP values
    Input parameters:
        path = cwd
        areaList = histopath site(s) to derive ML model from
        figNum = either 'fig3' or 'fig4'
        kNeigh = used for SMOTE balancing of train set; default is 5 but must set to 3 when looking at NAP site because of small dataset
        histo = histopath site model is derived from
    Outputs:
        returns:
            perfList = list of accuracy, f1 score
            dfShap = dataframe of SHAP values
            dfAvgNorm = dataframe of features used to construct ML model
            dfFeatTest = top features from feat importance analysis
    '''
    
    import pandas as pd
        
    #empty list to store the dfAvgNorms that come out of functions
    dfAvgNormList = []

    #first log10+1 normalize cell density
    dfAvgNormD = normalizeDensity(path=path,areaList=areaList)
    dfAvgNormList.append(dfAvgNormD)

    #then add interaction features and normalize
    dfAvgNormI =normalizeInteractions(path=path,areaList=areaList)
    dfAvgNormList.append(dfAvgNormI)

    #then add T cell functionality barcodes and normalize
    dfAvgNormB = normalizeBarcodes(path=path,areaList=areaList)
    dfAvgNormList.append(dfAvgNormB)

    #merge feature dfs - density + interactions + barcodes into dfAvgNorm and get the columns
    dfAvgNorm = pd.concat(dfAvgNormList,axis=1) #dfAvgNorm will have 1260 features (bc 8 spatial interactions do not exist in the whole dataset but are in the df, so those columns have all zeros (do not affect model))
    cols = dfAvgNorm.columns

    #then set aCD40 to be the positive class label
    dfAvgNorm,ptIdxList,mlIdxList = setPositive(path=path,dfAvgNorm=dfAvgNorm,figNum=figNum)

    #do the classification
    dfPredictTest, shapTestArray, featTestArray, dfFeatTest = classify(dfAvgNorm=dfAvgNorm,ptIdxList=ptIdxList,mlIdxList=mlIdxList,kNeigh=kNeigh,cols=cols)

    #calculate  performance metrics & save shap plots & top feature box plots; pass histopath site for saving figure names
    perfList = performance(path=path,dfPredictTest=dfPredictTest,histo=histo,figNum=figNum)
    
    #only get feature importance for IA histopath for the aCD40 DFS model
    if (figNum == 'fig4') and (histo != 'IA'):
        dfShap = None
        pass
    else:
        dfShap = featImportance(path=path,shapTestArray=shapTestArray,featTestArray=featTestArray,cols=cols,histo=histo,dfAvgNorm=dfAvgNorm,figNum=figNum)

    return perfList, dfShap, dfAvgNorm, dfFeatTest



def makeClassifierFigures(path,figNum):
    '''
    This function is used in figs3&4 to create figures 3/4A, 3/4C, loops through each histopath site
    Input parameters:
        path = cwd
        figNum = 'fig3' or 'fig4'
    Outputs:
        Saves plots to 'figures' folder for figures 3A or 4A, 3C or 4C
    '''
    
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly
    import seaborn as sns
    import matplotlib.pyplot as plt

    histPerfDict = {} #empty dict to store aggregated performance for each histopath site

    #only run NAP model if it's the tx classifier bc not enough aCD40 regions to split on dfs
    if figNum == 'fig3':
        areaListList = [['T'],['IA'],['TAS'],['NAP']]
        histOrder = ['T','IA','TAS','NAP'] #same order as areaListList
    
    elif figNum == 'fig4':
        areaListList = [['T'],['IA'],['TAS']]
        histOrder = ['T','IA','TAS'] #same order as areaListList

    kNeigh = 5 #5 is default for smote; changes to 3 when NAP model

    #empty lists to store shap/feature values for each histopath models
    featList = [] #top feature names
    shapList = [] #top test shap values
    colorList = [] #feature values to determine directionality class colors from
    histList = [] #store the associated histopath model type
    shapRankList = [] #store the rank of the top features
    shapRank = list(range(1,16)) #create a resuable list from 1-15, then reverse it so size of bubbles are intuitive
    shapRank.reverse() #does it in place, returns None


    #run classifier pipeline for each histopath site
    for areaList in areaListList:

        #adjust kNeigh in smote if NAP, regardless of obs=all or avg
        if areaList[0] == 'NAP':
            kNeigh = 3

        print('\nBuilding Histopathologic Model: '+areaList[0]+'...\n')

        perfList, dfShap, dfAvgNorm, dfFeatTest = pipeline(path=path,areaList=areaList,figNum=figNum,kNeigh=kNeigh,histo=areaList[0])

        #put perfDict into its own df
        dfPerf = pd.DataFrame(perfList,columns=['Test'])
        dfPerf['Metric'] = ['Accuracy','F1']
        dfPerf['Hist'] = areaList[0]
        #now add dfPerf to histPerfDict
        histPerfDict[areaList[0]] = dfPerf

        #Figure Main C - Bubble chart of top features per histopath model
        if (figNum == 'fig4') and (areaList[0] != 'IA'):
            pass
        else:
            #save top 15 shap/feature values
            topList = list(dfShap['Feature'].values[0:15])
            featList.extend(topList) #add top 15 features
            shapList.extend(list(dfShap['Test Shap Value'].values[0:15])) #add top 15 shap values
            histList.extend([areaList[0]] * 15) #hist=areaList[0]
            shapRankList.extend(shapRank) #shapRank defined outside of for loop

            #now identify which direction (aka tx cohort or dfs group) the higher feature value codes for = will be the color in the bubble plot
            #subset features in test shap colors to top 15 features
            topList.append('MLclass')
            dfFeatTestTop = dfFeatTest[topList]

            #compare which class has larger feature values (NOTE THESE ARE NOT SHAP VALUES) - for color coding
            df1Feat = dfFeatTestTop[dfFeatTestTop['MLclass'] == 0]
            df2Feat = dfFeatTestTop[dfFeatTestTop['MLclass'] == 1]
            colorList.extend(list(df1Feat.mean()[:-1] < df2Feat.mean()[:-1])) #False= naive/longdfs is bigger; True = acd40/shortdfs is bigger- use this for coloring!


    #concatenate all hists performances to plot
    dfPerfAll = pd.concat(histPerfDict.values(), ignore_index=True)

    #Figure Main A - plot test performance metrics
    colorMap = {'T':plotly.colors.qualitative.Dark2[3],'IA':plotly.colors.qualitative.Dark2[4],'TAS':plotly.colors.qualitative.Dark2[5],'NAP':plotly.colors.qualitative.Dark2[2]}
    fig = px.bar(dfPerfAll,x='Metric',y='Test',barmode='group',category_orders={'Hist':histOrder},color='Hist',color_discrete_map=colorMap,range_y=[0,1.02],labels={'TEST':'Performance','Metric':''})
    fig.update_layout(width=400,height=500)
    fig.write_image(path+'/results/figures/figure'+figNum[-1]+'A.png')

    #once all models have been run, create Figure Main C - bubble chart of top 15 features per model
    dfAll = pd.DataFrame()
    dfAll['Feature'] = featList
    dfAll['Shap'] = shapList
    dfAll['Shap Rank'] = shapRankList
    dfAll['Model: Hist'] = histList
    dfAll['Color'] = colorList

    #update Color column to have meaning
    if figNum == 'fig3':
        dfAll['MLclass'] = np.where(dfAll['Color'] == False,'Naive','aCD40') #False = naive is bigger; True=aCD40 is bigger
        colorDict = {'Naive':'tab:orange','aCD40':'tab:blue'}
        sns.set(rc={'figure.figsize':(10,20)},font_scale=2)

    elif figNum == 'fig4':
        dfAll['MLclass'] = np.where(dfAll['Color'] == False,'Long','Short') #False = long dfs is bigger; True=short dfs is bigger
        colorDict = {'Short':'purple','Long':'green'}
        sns.set(rc={'figure.figsize':(3,7)},font_scale=1.2)

    #sort dfAll so that the features of the same type are together in order 
    dfAll['Order1'] = np.where(dfAll['Feature'].str.contains('Interaction'),2,1) #label interactions
    dfAll['Feature Order'] = np.where(dfAll['Feature'].str.contains('Density'),0,dfAll['Order1']) #label 
    dfAll = dfAll.sort_values('Feature Order')
    dfAll = dfAll.drop(columns='Order1')
    dfAll.to_csv(path+'/results/dfCreated/dfTopFeatures_'+figNum+'.csv') #save dfAll to use in Fig Main D

    #sort columns extra if figure 3 w/ multiple histopath models
    if figNum == 'fig3':
        sorterIndex = dict(zip(histOrder, range(len(histOrder))))
        dfAll['Model Order'] = dfAll['Model: Hist'].map(sorterIndex)
        dfAll = dfAll.sort_values(['Feature Order','Model Order'])

    #plot Fig Main C
    graph = sns.scatterplot(data=dfAll,x='Model: Hist',y='Feature',size='Shap Rank',hue='MLclass',sizes=(100,1000),palette=colorDict)
    graph.legend(bbox_to_anchor=(1.02,1))
    graph.set_xlabel('Histopathologic Classifier Model')
    graph.set_ylabel('Feature')
    graph.figure.savefig(path+'/results/figures/figure'+figNum[-1]+'C.png',format='png',bbox_inches='tight')
    plt.close()



def makeBarcodeMatrix(path,figNum):
    '''
    This function is used in figs3&4 to create correlation matrix for top barcodes based on T cell composition
    Input parameters:
        path = cwd
        figNum = 'fig3' or 'fig4' 
    Outputs:
        Saves plots to 'figures' folder for figure 3E or 4E, note that the figure is saved in two parts
    '''

    import re
    from matplotlib.patches import Patch
    from matplotlib import pyplot as plt
    import plotly.express as px
    import plotly
    import numpy as np
    import pandas as pd
    import seaborn as sns

    #get df with all features listed and subset to barcode features only
    dfAllFeat = pd.read_csv(path+'/results/dfCreated/dfTopFeatures_'+figNum+'.csv',index_col=0)
    dfBarFeat = dfAllFeat[dfAllFeat['Feature'].str.contains('Barcode')]

    #read file to see which t cell states express each barcode combo
    dfCombo = pd.read_csv(path+'/results/dfCreated/dfBarcodeMaster.csv',index_col=0)

    #add annotation columns to dfCombo - pt tx hist
    patternP = re.compile(r"(.*)_.*_.*_.*") #patient ID
    patternH = re.compile(r".*_.*_(.*)_.*") #histopath

    ptList = []
    histList = []

    for file in dfCombo['file']:
        pt = re.match(patternP,file).group(1)
        ptList.append(pt)
        hist = re.match(patternH,file).group(1)
        histList.append(hist)
    dfCombo['pt'] = ptList
    dfCombo['hist'] = histList

    #if tx model, get tx
    if figNum == 'fig3':
        dfCombo['tx'] = np.where(dfCombo['pt'].str.contains('V'),'aCD40','Naive')
        colName = 'tx' #for subsetting dfCombo later

    #if dfs model, get dfs classes
    elif figNum == 'fig4':
        #add survival class to each T cell
        #read in clinical file
        dfClin = pd.read_csv(path+'/data/metadata/clinicalData.csv',index_col=0)

        #subset to just desired aCD40 cohort
        dfClin = dfClin[dfClin['tx'] == 'aCD40'] 

        #add long/short survival
        med = dfClin['DFS'].median() #get median of cohort survival - based on pt average in dfClin

        #positive label in ML classifier should be SHORT SURVIVORS
        dfClin['MLclass'] = np.where(dfClin['DFS'] <= med,'Short','Long') #0=long survival, 1=short survival

        #merge on pt to get the ML class
        dfCombo = dfCombo.merge(dfClin['MLclass'],left_on='pt',right_on='sample')
        colName = 'MLclass' #for subsetting dfCombo later



    # translate barcode in dfCombo and add new column
    def barcode(f):
        # barcode order = 'PD1+_TOX+_TIM3+_LAG3+_CD39+_EOMES+_CD38+_CD44+_TCF1/7+_TBET+'
        barcodeKeyDict = {0:'PD1+',1:'TOX+',2:'TIM3+',3:'LAG3+',4:'CD39+',5:'EOMES+',6:'CD38+',7:'CD44+',8:'TCF1/7+',9:'TBET+'}
        barcode = f.replace('_', '') #drop _ in barcode
        markerString = 'Barcode: ' #start with empty string to add to
        for i in range(len(barcode)):
            binary = barcode[i] #check if it's a 0 or 1
            if binary == '1': #if it's 1, get the correspoinding marker at that index
                marker = barcodeKeyDict[i]
                markerString = markerString+' '+marker
        if markerString == 'Barcode: ': #if no markers are positive
            markerString = 'Barcode: Negative for all'
        return markerString

    #apply barcode() function to translate
    dfCombo['Barcode'] = dfCombo['barcode'].apply(barcode)

    #now loop through each top barcode features - match tx, hist, barcode - in dfCombo and store its cell comp in dict
    barCellDict = {} #store {tx_hist_barcode:{cell1:%,cell2:%...},...}

    #for each barcode, get its tx/hist
    for index, row in dfBarFeat.iterrows():
        b = row['Feature']
        t = row['MLclass']
        h = row['Model: Hist']

        #find that combo in dfCombo
        dfSubset = dfCombo[(dfCombo['Barcode'] == b) & (dfCombo[colName] == t) & (dfCombo['hist'] == h)]

        #save which T cells express this barcode for this tx cohort and histopath site as a dict
        cellPerc = dict(dfSubset['state'].value_counts(normalize=True)*100)

        #save to bigger dict
        barCellDict[t+'_'+h+':  '+b[10:]] = cellPerc #drop the "Barcode:  "

    #put barCellDict into df
    cellList = ['CD8 T Cells_Naive','CD8 T Cells_T Effector','CD8 T Cells_Tem','CD8 T Cells_Temra','CD8 T Cells_Early Exhaustion','CD8 T Cells_Terminal Exhaustion','CD8 T Cells_Other CD44-','CD8 T Cells_Other CD44+',
                'CD4 Th1 helper Cells_T Effector','CD4 Th1 helper Cells_Tem','CD4 Th1 helper Cells_Temra','CD4 Th1 helper Cells_Other CD44-','CD4 Th1 helper Cells_Other CD44+',
                'Other CD4 Th Cells_Other CD44-','Other CD4 Th Cells_Other CD44+',
                'Tregs_Naive','Tregs_mTreg','Tregs_ExTreg']
    dfBarCell = pd.DataFrame(barCellDict.values(),index=barCellDict.keys(),columns=cellList).fillna(0)

    #now cluster dfBarCell based on cell composition to get ordering of barcodes that will be used in correlation heatmap
    if figNum == 'fig3':
        #add annotation columns to dfCorr - tx hist
        patternT = re.compile(r"(.*)_.*:.*") #tx
        patternH = re.compile(r".*_(.*):.*") #histopath

        txList = []
        histList = []

        for i in dfBarCell.index:
            tx = re.match(patternT,i).group(1)
            txList.append(tx)
            hist = re.match(patternH,i).group(1)
            histList.append(hist)

        dfBarCell['tx'] = txList
        dfBarCell['hist'] = histList

        #add color annotation columns to put next to heatmap since ordering will be the same
        palette1 = {'Naive':'#FF7F0E','aCD40':'#1F77B4'}
        palette2 = {'T':'#e7298a','IA':'#66a61e','TAS':'#e6ab02','NAP':'#7570b3'}
        grouping1 = dfBarCell['tx']
        grouping2 = dfBarCell['hist']
        colors1 = pd.Series(grouping1,name='Tx Cohort').map(palette1)
        colors2 = pd.Series(grouping2,name='Histopath').map(palette2)
        dfColors = pd.concat([colors1,colors2],axis=1)

        colI = -2 #for making dendro

    elif figNum == 'fig4':
        #add annotation columns to dfCorr dfs long vs short
        patternD = re.compile(r"(.*)_.*:.*") #Dfs
        dfsList = []

        for i in dfBarCell.index:
            dfs = re.match(patternD,i).group(1)
            dfsList.append(dfs)

        dfBarCell['DFS'] = dfsList

        #add color annotation columns to put next to heatmap since ordering will be the same
        palette1 = {'Short':'purple','Long':'green'} #short=purple,long=green
        grouping1 = dfBarCell['DFS']
        colors1 = pd.Series(grouping1,name='DFS').map(palette1)
        dfColors = pd.concat([colors1],axis=1)

        colI = -1 #for making dendro

    #plot clustering of barcodes based on proportions of T cells that express them - note this is NOT the correlation matrix in Fig 3D
    #use this to get the ordering of the rows to plot the correlation matrix via dendro variable
    dendro = sns.clustermap(data=dfBarCell.iloc[:,:colI],yticklabels=True,xticklabels=True,cmap='crest',method='ward',metric='euclidean',row_colors=dfColors)
    plt.close() #don't need to show

    #now correlate barcodes and plot in order of dendrogram above
    idxList = dendro.dendrogram_row.reordered_ind #get order of clustering indices
    barList = list(dfBarCell.index)

    newOrderList = [] #to store updated ordering of barcodes

    #for index; map to barcode
    for i in idxList:
        #get corresponding barcode
        bar = barList[i]
        newOrderList.append(bar)

    #correlate barcodes based on cell types expressing them
    dfCorr = dfBarCell.iloc[:,:colI].T.corr() #drop the annotation columns when correlating

    #reorder dfCorr to match dendrogram ordering
    dfCorr = dfCorr[newOrderList]
    dfCorr = dfCorr.reindex(newOrderList)

    #plot correlation heatmap w/ dendrogram ordering - NOTE: use clustermap to get row_colors (https://stackoverflow.com/questions/73433022/adding-row-colors-to-a-heatmap)
    graph = sns.clustermap(data=dfCorr, figsize=(26,24),row_colors=dfColors, row_cluster=False, col_cluster=False,yticklabels=True,xticklabels=True,cmap='crest')

    #add legends and plot
    ax = graph.ax_heatmap

    if figNum == 'fig3':
        handles1 = [Patch(facecolor=palette1[c]) for c in palette1]
        leg1 = ax.legend(handles1, palette1, title='Tx Cohort', bbox_to_anchor=(1.15, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
        handles2 = [Patch(facecolor=palette2[c]) for c in palette2]
        leg2 = ax.legend(handles2, palette2, title='Histopath', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='center right')
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        barGap = 0.05

    elif figNum == 'fig4':
        handles1 = [Patch(facecolor=palette1[c]) for c in palette1]
        leg1 = ax.legend(handles1, palette1, title='DFS', bbox_to_anchor=(1.15, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
        ax.add_artist(leg1)
        barGap = 0.025

    plt.savefig(path+'/results/figures/figure'+figNum[-1]+'E_matrix.png',format='png',bbox_inches='tight')
    plt.close()

    #now get the stacked bars of t cell composition expressing each barcode and save that part of the figure
    #define color palette
    colorList = ['Other CD4 Th Cells_Other CD44-', 'Tregs_ExTreg',
           'CD8 T Cells_Other CD44-', 'Tregs_Naive',
           'CD4 Th1 helper Cells_Other CD44-', 'Tregs_mTreg',
           'Other CD4 Th Cells_Other CD44+', 'CD8 T Cells_Temra',
           'CD8 T Cells_Other CD44+', 'CD8 T Cells_Tem',
           'CD4 Th1 helper Cells_Tem', 'CD8 T Cells_Naive',
           'CD4 Th1 helper Cells_Other CD44+', 'CD4 Th1 helper Cells_Temra',
           'CD4 Th1 helper Cells_T Effector', 'CD8 T Cells_Early Exhaustion',
           'CD8 T Cells_T Effector', 'CD8 T Cells_Terminal Exhaustion',
            'B Cells_Total',
            'Myeloid Cells_Total',
            'Fibroblasts_Total',
            'Tumor_Ki67-',
            'Tumor_Ki67+']
    palette = dict(zip(colorList,plotly.colors.qualitative.Dark24)) 

    #reorder the index of dfBarCell to follow the ordering in the heatmap
    dfBarCell = dfBarCell.reindex(newOrderList)

    #plot the stacked bars of T cell composition
    fig = px.bar(dfBarCell.iloc[:,:colI],barmode='stack',color_discrete_map=palette,labels={'value':'T Cells Expressing Barcodes','index':''})
    fig.update_layout(bargap=barGap)
    fig.update_layout(xaxis={'visible': False})    
    fig.update_layout(showlegend=False)
    fig.write_image(path+'/results/figures/figure'+figNum[-1]+'E_stackbars.png')


 
def makeNeighborhoods(path,csvList):
    '''
    This function is used in fig5 to calculate the neighborhoods for every cell in all aCD40-treated IA regions
    Input parameters:
        path = cwd
        csvList = list of aCD40 IA regions 
    Outputs:
        Saves dfNeighobrhoods csv to 'dfCreated' folder
    '''
    
    import pandas as pd
    from scipy import spatial

    #empty list to hold dictionaries to create new rows of dfClust; outside of for file in csvList loop
    allNeighList = []

    #list of columns that make up the barcodes - to convert to strings later
    comboColList = ['Cellsp_PD1p','Cellsp_TOXp','Cellsp_TIM3p','Cellsp_LAG3p','Cellsp_CD39p','Cellsp_EOMESp','Cellsp_CD38p','Cellsp_CD44p','Cellsp_TCF17p','Cellsp_TBETp','Cellsp_KI67p','Cellsp_GRZBp']

    #list of t cell state types
    tCellList = ['CD8 T Cells_Naive',
                 'CD8 T Cells_T Effector',
                 'CD8 T Cells_Tem',
                 'CD8 T Cells_Temra',
                 'CD8 T Cells_Early Exhaustion',
                 'CD8 T Cells_Terminal Exhaustion',
                 'CD8 T Cells_Other CD44-',
                 'CD8 T Cells_Other CD44+',
                 'CD4 Th1 helper Cells_T Effector',
                 'CD4 Th1 helper Cells_Tem',
                 'CD4 Th1 helper Cells_Temra',
                 'CD4 Th1 helper Cells_Other CD44-',
                 'CD4 Th1 helper Cells_Other CD44+',
                 'Other CD4 Th Cells_Other CD44-',
                 'Other CD4 Th Cells_Other CD44+',
                 'Tregs_Naive',
                 'Tregs_mTreg',
                 'Tregs_ExTreg']

    print('Calculating neighborhoods. This will likely take several (~15+) minutes...')
    
    #loop through each ROI in csvList
    for file in csvList:
        # read df according to path
        df = pd.read_csv(path+'/data/mIHC_files/'+file+'.csv', index_col=0)
        #add state column to get child level cell state defn
        df['state'] = df['CellType']+'_'+df['DefinedName']  
        #add barcode column for all cells - doesn't matter that it's happening on non-T cells for this part
        #add version without ki67/grzb as well as with both
        #convert the functional columns to be strings so you can concatenate them into barcode
        df[comboColList] = df[comboColList].astype(str)
        df['barcode'] = df[['Cellsp_PD1p','Cellsp_TOXp','Cellsp_TIM3p','Cellsp_LAG3p','Cellsp_CD39p','Cellsp_EOMESp','Cellsp_CD38p','Cellsp_CD44p','Cellsp_TCF17p','Cellsp_TBETp']].agg('_'.join, axis=1)
        df['barcodeKG'] = df[['Cellsp_PD1p','Cellsp_TOXp','Cellsp_TIM3p','Cellsp_LAG3p','Cellsp_CD39p','Cellsp_EOMESp','Cellsp_CD38p','Cellsp_CD44p','Cellsp_TCF17p','Cellsp_TBETp','Cellsp_KI67p','Cellsp_GRZBp']].agg('_'.join, axis=1)

        #create filtered dataframe without 'other cells'
        filt_df = df[(df['CellType'] != 'Other Cells') ]

        #get all possible class values; for later counting - from prior knowledge of possible cell types
        classOptions = ['CD8 T Cells_Naive',
                         'CD8 T Cells_T Effector',
                         'CD8 T Cells_Tem',
                         'CD8 T Cells_Temra',
                         'CD8 T Cells_Early Exhaustion',
                         'CD8 T Cells_Terminal Exhaustion',
                         'CD8 T Cells_Other CD44-',
                         'CD8 T Cells_Other CD44+',
                         'CD4 Th1 helper Cells_T Effector',
                         'CD4 Th1 helper Cells_Tem',
                         'CD4 Th1 helper Cells_Temra',
                         'CD4 Th1 helper Cells_Other CD44-',
                         'CD4 Th1 helper Cells_Other CD44+',
                         'Other CD4 Th Cells_Other CD44-',
                         'Other CD4 Th Cells_Other CD44+',
                         'Tregs_Naive',
                         'Tregs_mTreg',
                         'Tregs_ExTreg',
                         'B Cells_Total',
                         'Myeloid Cells_Total',
                         'Fibroblasts_Total',
                         'Tumor_Ki67-',
                         'Tumor_Ki67+']
        
        #generate count variable names for final created df's columns
        classCountNames = []
        for c in classOptions: #loop through all possible neighboring cells
            classCountNames.append('count'+c)

        ##get nearest neighbors of seed cells defined by seed param
        #create np array of just x,y coordinates
        ptsArray = filt_df[['Location_Center_X','Location_Center_Y']].values

        #create kdtree
        tree = spatial.KDTree(ptsArray)

        #loop through each cell and get its neighbors
        for i in range(len(ptsArray)):

            classType = filt_df['state'].values[i]
            neighs = tree.query_ball_point(ptsArray[i], 120) #get neighbors within 120px = 60µm
            #NOTE: want to include the seed cell as a neighbor to get composition of entire neighborhood

            neighClassList = [] #empty list to store neighboring cells' classes
            neighIdxList = [] #new empty list to store neighboring T cell's original indices from the df.loc index; could remain empty if no t cell neighbors

            #loop through each neighbor and get its class; add it to neighClassList
            for j in neighs:
                #get its class
                neighClass = filt_df['state'].values[j]
                neighClassList.append(neighClass)

                #if the neighboring cell is a t cell, get its original index so you can link back to its barcode
                #this ID will be in the form of the original csv's name aka df.loc index; just like how the seed cell's idx is stored
                if neighClass in tCellList:

                    #Then check if it's actually the seed cell
                    #if the "neighbor" is the seed cell, we don't need to store its index since the seed cell's idx and barcode is saved
                    if i != j: #if seed is not neighbor, then store its idx
                        neighIdxList.append(filt_df.iloc[j].name) #original neighboring cell's idx value in the form of its original csv's name aka df.loc index

            #get counts of neighboring cell types
            classCounts = []
            for c in classOptions: #loop through all possible neighboring cells
                count = neighClassList.count(c) #count the specified cell type
                classCounts.append(count) #add the counts of the specified cell type to classCounts

            #reset dictionary to hold counts of neighbors; one per seed
            seedDict = {}

            #add counts to a dictionary (one per seed cell); also add original seed cell's idx value 
            seedDict['file'] = file
            seedDict['index'] = filt_df.iloc[i].name #original seed cell's idx value in the form of its original csv's name aka df.loc index

            #also add seed cell's original cell state value
            seedDict['seed state'] = classType

            #add seed cell's barcode if it's a t cell
            if classType in tCellList:
                barcode = filt_df['barcode'].values[i] #SEED CELL'S BARCODE W/O KI67/GRZB
                barcodeFun = filt_df['barcodeKG'].values[i] #SEED CELL'S BARCODE W/ KI67/GRZB

            else: #if not a t cell set barcode to '--' bc this never gets used elsewhere
                barcode = '--'
                barcodeFun = '--'

            seedDict['barcode'] = barcode #no ki67/grzb
            seedDict['barcodeKG'] = barcodeFun #with ki67/grzb

            #for each possible neighboring cell class, add its count to seedDict both raw and as a %
            for n in range(len(classCountNames)):
                seedDict[classCountNames[n]] = classCounts[n] #raw count
                if sum(classCounts) != 0:
                    seedDict[classCountNames[n]+'%'] = classCounts[n]/sum(classCounts) #percentage
                else: #avoid division by zero if there are no neighbors
                    seedDict[classCountNames[n]+'%'] = 0 #set % to zero if there are no neighbors

            #then add the neighboring t cell index list
            seedDict['neighT idx list'] = neighIdxList
            #add each seed's neighbor dictionary to the overall list; one dictionary per row of df
            allNeighList.append(seedDict)

    #create one new df to hold data for clustering; pass in allNeighList as the data; format is one row per seed cell across all csvs
    #column names from a seedDict's keys (all seedDicts have the same keys)
    dfClust = pd.DataFrame(data = allNeighList, columns = list(seedDict.keys()))
    #NOTE: only using the columns (cells present) from the final csv that was analyzed. If a cell is not present at all in this csv then its column will not be present in the final csv created.

    #convert any NaN values to zeros [note that NaN values arise when a csv lacks any of a cell type that does exist in other csvs]
    dfClust = dfClust.fillna(0)

    #store dfClust as a csv
    dfClust.to_csv(path+'/results/dfCreated/dfNeighborhoods.csv')
    print("Cellular neighborhoods calculated for aCD40 IA regions.")
    
    
    
def elbowMethod(path,file,steps):
    '''
    This function is used in fig5 to run the elbow method to determine the optimal number of clusters for k-means clustering.
    Input parameters:
        path = cwd
        file = name of file to run clustering on (dfNeighborhoods)
        steps = max number of clusters (k) to test
    Output:
        saves supplementary figure S4A to 'figures' folder
    '''
    
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.cluster import MiniBatchKMeans #minibatchkmeans is better when n > 10,000 samples

    #load dfNeighborhoods file
    df = pd.read_csv(path+'/results/dfCreated/'+file+'.csv', index_col=0)
    
    #drop all rows that have no cells in the neighborhood (aka when the only neighbor is itself; sum>2 because 1=count of itself + 1=%count of itself=2)
    df['sum'] = df.iloc[:,5:-2].sum(axis=1) #get sum of just how many neighboring cells there are - update the column numbers to account for additional columns now
    df = df[df['sum'] > 2]

    #generate column list to cluster on based on if there is a % in the column name
    colList = list(df.columns[['%' in col for col in list(df.columns)]])

    #get only features we want to cluster on
    df = df[colList]
    data = df.values

    #empty list to store error value
    wcss = []

    #calculate error for each k value (k=number of clusters)
    for k in range(1, steps):
        #generate kmeans model
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        #fit model to data
        kmeans.fit(data)
        #add the sum of squares to wcss list; for plotting elbow
        wcss.append(kmeans.inertia_)

    #generate elbow plot
    plt.plot(range(1, steps), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    plt.savefig(path+'/results/figures/figureS4A.png',format='png')
    plt.close()
    
    

def clusterNeighborhoods(path,file,k):
    '''
    This function runs k-means clustering on a given neighborhood clustering csv.
    Input parameters:
        path = cwd
        file = name of file to run clustering on excluding the .csv (ex. dfNeighborhoods)
        k = number of clusters; use elbow method to determine optimal number
    Outputs:
        dfNeighborhoodClusters csv is saved to the 'dfCreated/' folder
    '''

    import pandas as pd
    from sklearn.cluster import MiniBatchKMeans
    
    #read csv with neighborhood data    
    df = pd.read_csv(path+'/results/dfCreated/'+file+'.csv', index_col=0)

    #drop all rows that have no cells in the neighborhood (aka when the only neighbor is itself; sum>2 because 1=count of itself + 1=%count of itself=2)
    df['sum'] = df.iloc[:,5:-2].sum(axis=1) #get sum of just how many neighboring cells there are - update the column numbers to account for additional columns now
    dfFilt =df[df['sum'] > 2].copy()

    #get lists from original df to add back later after clustering
    roiList = list(dfFilt['file']) #get list of ROIs in order to add to dfNoNa later
    idxList = list(dfFilt['index']) #get list of cell indices in order to add to dfNoNa later
    seedStateList = list(dfFilt['seed state']) #list of seed cell state defns
    barcodeList = list(dfFilt['barcode']) #list of seed cell barcode w/o ki67/grzb
    barcodeFunList = list(dfFilt['barcodeKG']) #list of seed cell barcode with ki67/grzb
    neighTIdxList = list(dfFilt['neighT idx list']) #list of lists of neighboring t cell original indices

    #generate column list to cluster on based on if there is a % in the column name
    colList = list(dfFilt.columns[['%' in col for col in list(dfFilt.columns)]])

    #get only features we want to cluster on
    dfFilt = dfFilt[colList]
    data = dfFilt.values

    #=k-means clustering of cells with k clusters
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    predict = kmeans.fit_predict(data) #fit model to data and predict index (cluster labels); same results as first fitting and then predicting

    #add predicted cluster labels to df as a new column and see how many cells are in each cluster
    dfFilt['cluster'] = predict
    dfFilt['cluster'].value_counts() #unecessary unless you want to see how many cells are in each cluster and then you should print it

    #add original ROI to df to check which ROIs are in each cluster
    dfFilt['file'] = roiList

    #need to add original indices to this df to check which ROIs are in each cluster; will also need to use ROI ID to pair with cell index (same index could be had by two cells from diff ROIs)
    dfFilt['index'] = idxList #idxList stores the row value of filt_df.iloc[row,column] command

    #add back original annotations to dfFilt
    dfFilt['seed state'] = seedStateList
    dfFilt['barcode'] = barcodeList
    dfFilt['barcodeKG'] = barcodeFunList
    dfFilt['neighT idx list'] = neighTIdxList
    
    #save df to a csv
    dfFilt.to_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv')



def clusterCountFrequency(path):
    '''
    This function takes in a clustered csv and plots number of RCNs per region, patient
    Input parameters:
        path = cwd
    Outputs:
        Saves figures S4D, S4E to 'figures' folder
    '''

    import pandas as pd
    import re
    import plotly.express as px
    
    #read clustered neighborhood csv
    df = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)

    dictCounts = {} #empty dict to store raw counts for each cluster per ROI

    #for each roi in the big df
    for roi in df['file'].unique():
        dfSingleROI = df[df['file'] == roi] #df contains cells from only one region

        #raw frequency counts of cluster
        freq = dfSingleROI['cluster'].value_counts()

        #add to dict to store results
        dictCounts[roi] = freq

    #put counts of clusters into a df
    dfClustCounts = pd.DataFrame(dictCounts).T
    dfClustCounts = dfClustCounts.fillna(0)
    #reorder to match prior ordering
    dfClustCounts = dfClustCounts.rename(columns={0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'})

    obsList = ['all','avg'] #all=Region; avg=Patient

    for obs in obsList:
        #only do something if needing to avg; otherwise stick with what is already present
        if obs == 'avg':
            dfNew = dfClustCounts.copy()
            #average regions per patient
            newList = [] #to store new indices
            patternP = re.compile(r"(.*)_.*_.*_.*") #get first info from index name = pt id
            for idx in dfNew.index:
                pt = re.match(patternP,idx).group(1)
                newList.append(pt)
            dfNew.index = newList
            #groupby patient and take average of all columns across patients
            dfNew = dfNew.groupby(dfNew.index).mean()
            #set x axis label to be patient
            xAx = 'Patient'
            figNum = 'E'

        elif obs == 'all':
            dfNew = dfClustCounts.copy()
            xAx = 'Region'
            figNum = 'D'

        #convert raw counts to % columns
        for c in dfNew.columns:
            dfNew[c+'_%'] = dfNew[c]/dfNew.sum(axis=1)

        #just get % columns
        dfPerc = dfNew[dfNew.columns[['%' in col for col in list(dfNew.columns)]]]

        #reorder from 1-7
        dfPerc = dfPerc[['RCN1_%','RCN2_%','RCN3_%','RCN4_%','RCN5_%','RCN6_%','RCN7_%']]
        #drop the %s from the col name
        dfPerc = dfPerc.rename(columns={'RCN1_%':'RCN1','RCN2_%':'RCN2','RCN3_%':'RCN3','RCN4_%':'RCN4','RCN5_%':'RCN5','RCN6_%':'RCN6','RCN7_%':'RCN7'})
        #sort descending RCN1
        dfPercSort = dfPerc.sort_values('RCN1',ascending=False)

        fig = px.bar(dfPercSort,y=dfPercSort.columns,barmode='stack',labels={'index':xAx,'value':'Fraction of RCNs Present','variable':'RCN'})
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(legend_traceorder="reversed")
        fig.write_image(path+'/results/figures/figureS4'+figNum+'.png')



def createCsvsWithClusterCol(path,csvList):
    '''
    This function creates new csvs for each original csv with an added cluster column, which is used to plot scatterplot images
    Input parameters:
        path = cwd
        csvList = list of csvs to update
    Outputs:
        Saves one csv per aCD40 IA region to the /results/dfCreated/updatedCsvs/ folder. This new csv has an additional column called 'cluster.'
    '''

    import pandas as pd

    #get clustered df; contains ALL cells from all IA aCD40 regions
    dfClust = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)

    #turn off pandas warning for adding values to specified column with 'loc' command
    #per: https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
    pd.options.mode.chained_assignment = None 

    dfDict = {} #empty dict to store roi:filtered dataframes with new cluster column

    #first break df into different dataframes based on region
    for roi in dfClust['file'].unique():
        dfROI = dfClust[(dfClust['file'] == roi)]
        #get original df based on dfClust['file'] and then iloc the seed cell using dfClust['index']
        df = pd.read_csv(path+'/data/mIHC_files/'+roi+'.csv', index_col=0)
        #get index and cluster number from dfROI and add these pairs as a tuple to an overall list
        idxClustList = list(zip(dfROI['index'], dfROI['cluster']))
        #add cluster column to filt_df with correct cluster number using idxClustList
        for tup in idxClustList: #for each tuple in idxClustList
             #use loc index and convert to loc's label (name)
            df.loc[tup[0],'cluster'] = tup[1] #tup[0] = original df's .loc index; tup[1]=cluster value of the seed cell
        dfDict[roi] = df


    #loop through each df again and update the cluster column to map back to the cell class if it doesn't have a neighborhood- aka for the other cells
    for roi,df in dfDict.items():
        df['cluster']=df['cluster'].fillna(df['CellType']) #if cluster value is a np NaN value, then replace it with its value from the class column

    #save newly updated dfs with their cluster column as new csvs
    for roi,df in dfDict.items():
        df.to_csv(path+'/results/dfCreated/updatedCsvs/'+roi+'_RCN.csv')
        
    

def fig1():
    '''
    Designing a novel mIHC antibody panel to deeply phenotype T cells within the PDAC TME.
    This function creates figure 1C. 
    Input parameters:
        None  
    Outputs:
        Saves plot to 'figures' folder for figure 1C    
    '''
    
    import os
    import pandas as pd
    import re
    import plotly
    import plotly.express as px

    print('\n\n***FIGURE 1 - mIHC panel and tissue region overview***\n')

    #get current working directory
    path = os.getcwd()
        
    ##FIGURE 1C - TISSUE REGIONS PER PATIENT BY HISTOPATH SITE
    df = pd.read_csv(path+'/data/metadata/dfArea.csv',index_col=0)
    
    histList = []
    ptList = []
    
    #add columns for pt and hist annotations
    patternH = re.compile(r".*_.*_(.*?)_.*") #histopath annotation
    patternP = re.compile(r"(.*)_.*_.*_.*") #patient ID
    
    for file in df.index:
        hist = re.match(patternH,file).group(1) #get the histopath site
        pt = re.match(patternP,file).group(1) #get patient ID
        histList.append(hist)
        ptList.append(pt)
    
    #add columns
    df['hist'] = histList
    df['patient'] = ptList
    
    df = df.sort_values('patient')
    
    #define color palette
    colorMap = {'T':plotly.colors.qualitative.Dark2[3],'IA':plotly.colors.qualitative.Dark2[4],'TAS':plotly.colors.qualitative.Dark2[5],'NAP':plotly.colors.qualitative.Dark2[2]}
    
    #plot histopath annotations
    fig = px.bar(df,x='patient',color='hist',category_orders={'hist':['T','IA','TAS','NAP']},labels={'count':'Number of Regions','patient':'Patient','hist':'Histopath Annotation'},color_discrete_map=colorMap)
    fig.update_layout(legend_traceorder="reversed")
    fig.update_layout(xaxis={'visible': False})    
    fig.write_image(path+'/results/figures/figure1C.png')
    print("Figure 1C saved to 'figures' folder.")
    print('Figure 1 complete.')
    
    
    
def fig2():
    '''
    Interrogating cell states and spatial interactions within the PDAC TME.
    This function creates figures 2A-2C and supplementary tables S3-S5.
    Input parameters:
        None  
    Outputs:
        Saves csvs to 'dfCreated' folder for files containing cell state densities, T cell functionality barcodes, and spatial interactions
        Saves plot to 'figures' folder for figures 2A-2C
        Saves tables to 'tables' folder for supplementary tables S3-S5
    '''
    
    import os
    import pandas as pd
    import numpy as np
    import re
    import plotly
    import plotly.express as px
    from scipy.spatial import distance_matrix
    from collections import Counter
    import seaborn as sns
    from matplotlib import pyplot as plt
    import warnings

    print('\n\n***FIGURE 2 - cell states, T cell functionality, and spatial organization***\n')

    #get current working directory
    path = os.getcwd()
    
    #define csvList and colOrder to be used for fig2 analyses
    csvList = [x[0:-4] for x in os.listdir(path+'/data/mIHC_files/')] #drop the .csv
    csvList.sort()
    
    #get correct order for df columns
    cd8List = ['CD8 T Cells_Naive','CD8 T Cells_T Effector','CD8 T Cells_Tem','CD8 T Cells_Temra','CD8 T Cells_Early Exhaustion','CD8 T Cells_Terminal Exhaustion','CD8 T Cells_Other CD44-','CD8 T Cells_Other CD44+']
    th1List = ['CD4 Th1 helper Cells_T Effector','CD4 Th1 helper Cells_Tem','CD4 Th1 helper Cells_Temra','CD4 Th1 helper Cells_Other CD44-','CD4 Th1 helper Cells_Other CD44+']
    othThList = ['Other CD4 Th Cells_Other CD44-','Other CD4 Th Cells_Other CD44+']
    tRegList = ['Tregs_Naive','Tregs_mTreg','Tregs_ExTreg']      
    bCellList = ['B Cells_Total']    
    myeList = ['Myeloid Cells_Total']
    fibList = ['Fibroblasts_Total']
    tumList = ['Tumor_Ki67-','Tumor_Ki67+']
    colOrder = cd8List+th1List+othThList+tRegList+bCellList+myeList+fibList+tumList    
    
    
    ##FIGURE 2A - CELL STATE DENSITIES
    print('Calculating cell state densities...')

    #first create dfDensity and dfCounts and save
    #get density of cell states = cell type_defined name
    roiDenDict = {} #dict to store file: density pairs
    roiCountDict = {} #dict to store file: raw cell count pairs
    
    #get region tissue areas
    dfArea = pd.read_csv(path+'/data/metadata/dfArea.csv',index_col=0)
    
    #loop through each mIHC file
    for file in dfArea.index:
    
        area = dfArea.loc[file,'area'] #get region's corresponding area
        
        #get mIHC csv file
        df = pd.read_csv(path+'/data/mIHC_files/'+file+'.csv',index_col=0)
        df = df[df['CellType'] != 'Other Cells'] #drop "other cells" from downstream analyses
        df['state'] = df['CellType']+'_'+df['DefinedName'] #add new state column with cell type and defined function together
        
        denDict = dict(df['state'].value_counts()/area) #divide by the area to get densities in cells/mm2
        countDict = dict(df['state'].value_counts()) #store raw counts
        
        #add file: density (or) counts to dictionaries
        roiDenDict[file] = denDict
        roiCountDict[file] = countDict
    
    #create csvs of counts and densities; sort by dfArea index to match original ordering for reproducibility
    dfDensity = pd.DataFrame.from_dict(roiDenDict, orient='index',columns=colOrder).reindex(dfArea.index)
    dfDensity = dfDensity.fillna(0) #convert all nan values to zeros
    
    dfCounts = pd.DataFrame.from_dict(roiCountDict, orient='index',columns=colOrder).reindex(dfArea.index)
    dfCounts = dfCounts.fillna(0) #convert all nan values to zeros
    
    #save density and count files to csv
    dfDensity.to_csv(path+'/results/dfCreated/dfDensity.csv')
    dfCounts.to_csv(path+'/results/dfCreated/dfCounts.csv')
    
    print("Cell state densities and raw counts calculated.")
    
    #now plot densities in stacked bar charts
    
    areaList = ['T','IA','TAS','NAP']
    
    #to match color scheme from other analyses
    colorList = ['Other CD4 Th Cells_Other CD44-', 'Tregs_ExTreg',
           'CD8 T Cells_Other CD44-', 'Tregs_Naive',
           'CD4 Th1 helper Cells_Other CD44-', 'Tregs_mTreg',
           'Other CD4 Th Cells_Other CD44+', 'CD8 T Cells_Temra',
           'CD8 T Cells_Other CD44+', 'CD8 T Cells_Tem',
           'CD4 Th1 helper Cells_Tem', 'CD8 T Cells_Naive',
           'CD4 Th1 helper Cells_Other CD44+', 'CD4 Th1 helper Cells_Temra',
           'CD4 Th1 helper Cells_T Effector', 'CD8 T Cells_Early Exhaustion',
           'CD8 T Cells_T Effector', 'CD8 T Cells_Terminal Exhaustion']
    
    palette = dict(zip(colorList,plotly.colors.qualitative.Dark24)) 
    
    #group all T cells together
    paletteParent = {'CD3+ T Cells': '#FFD700',
     'B Cells': '#620042',
     'Myeloid Cells': '#1616A7',
     'Fibroblasts': '#DA60CA',
     'Tumor': '#6C4516'}
    
    #read  density file
    dfAll = pd.read_csv(path+'/results/dfCreated/dfDensity.csv',index_col=0)
    
    #add histopath annotation: T, IA, TAS, NAP
    patternH = re.compile(r".*_.*_(.*)_.*")
    
    histList = [] #list for hist types
    
    for roi in dfAll.index:
        hist = re.match(patternH,roi).group(1)
        histList.append(hist)
    
    dfAll['hist'] = histList
    
    #add up cell densities into bulk densities (this is the same as adding counts of children and dividng total count by area)
    dfAll['CD3+ T Cells'] = dfAll['CD8 T Cells_Naive']+dfAll['CD8 T Cells_T Effector']+dfAll['CD8 T Cells_Tem']+dfAll['CD8 T Cells_Temra']+dfAll['CD8 T Cells_Early Exhaustion']+dfAll['CD8 T Cells_Terminal Exhaustion']+dfAll['CD8 T Cells_Other CD44-']+dfAll['CD8 T Cells_Other CD44+']+dfAll['CD4 Th1 helper Cells_T Effector']+dfAll['CD4 Th1 helper Cells_Tem']+dfAll['CD4 Th1 helper Cells_Temra']+dfAll['CD4 Th1 helper Cells_Other CD44-']+dfAll['CD4 Th1 helper Cells_Other CD44+']+dfAll['Other CD4 Th Cells_Other CD44-']+dfAll['Other CD4 Th Cells_Other CD44+']+dfAll['Tregs_ExTreg']+dfAll['Tregs_Naive']+dfAll['Tregs_mTreg']
    dfAll['B Cells'] = dfAll['B Cells_Total']
    dfAll['Myeloid Cells'] = dfAll['Myeloid Cells_Total']
    dfAll['Fibroblasts'] = dfAll['Fibroblasts_Total']
    dfAll['Tumor'] = dfAll['Tumor_Ki67+']+dfAll['Tumor_Ki67-']
    
    #add tx as a column
    dfAll['tx'] = np.where(dfAll.index.str.contains('V'),'aCD40','Tx-Naive')
    
    #empty df to store averages
    dfAvg = pd.DataFrame()
    
    #separate calculations per histopath site
    for h in areaList:
        #fist subset to histopath type
        dfH = dfAll[dfAll['hist'] == h]
        #then get tx cohort average cell densities
        dfTx = dfH.groupby('tx').mean(numeric_only=True)
        dfTx['hist'] = h
        dfAvg = pd.concat([dfAvg,dfTx])
        
    dfAvg['tx'] = dfAvg.index
     
    histRangeDict = {'T':[3000,140,215],'IA':[7500,1625,2725],'TAS':[2100,180,280],'NAP':[7000,160,315]}
    
    #plot separately for each histopath site
    for h in dfAvg['hist'].unique():
        
        #subset to specific histopath site
        dfAvgH = dfAvg[dfAvg['hist'] == h]
        
        #get range_y for corresponding hist
        rangeList = histRangeDict[h]
    
        fig = px.bar(dfAvgH,x='tx',y=dfAvgH.columns[-7:-2],title=h,range_y=[0,rangeList[0]],labels={'value':'Density (cells/mm2)','tx':'Hist: '+h},width=300,color_discrete_map=paletteParent)
        fig.update_layout(showlegend=False)
    #     fig.update_layout(legend={'traceorder':'reversed'})
        fig.write_image(path+'/results/figures/figure2A_'+h+'_lineage.png')
    
    
        fig = px.bar(dfAvgH,x='tx',y=dfAvgH.columns[0:8],title=h,range_y=[0,rangeList[1]],labels={'value':'Density (cells/mm2)','tx':'Hist: '+h},width=300,color_discrete_map=palette)
        fig.update_layout(showlegend=False)
    #     fig.update_layout(legend={'traceorder':'reversed'})
        fig.write_image(path+'/results/figures/figure2A_'+h+'_CD8.png')
    
    
        fig = px.bar(dfAvgH,x='tx',y=dfAvgH.columns[8:18],title=h,range_y=[0,rangeList[2]],labels={'value':'Density (cells/mm2)','tx':'Hist: '+h},width=300,color_discrete_map=palette)
        fig.update_layout(showlegend=False)
    #     fig.update_layout(legend={'traceorder':'reversed'})
        fig.write_image(path+'/results/figures/figure2A_'+h+'_CD4.png')

    ##FIGURE 2B - T CELL FUNCTIONALITY BARCODES
    print('Calculating T cell functionality barcodes...')

    #First create dfBarcodeMaster, dfBarcodeCounts, and dfBarcodeDensity csvs
    tList = ['CD8 T Cells','CD4 Th1 helper Cells','Other CD4 Th Cells','Tregs']
    
    #empty df to hold binary combos for each T cell across all csvs in dataset
    dfBarcodeMaster = pd.DataFrame()
    
    #loop through each of the csvs
    for file in csvList:
    
        #get original mIHC csv
        df = pd.read_csv(path+'/data/mIHC_files/'+file+'.csv',index_col=0)
    
        #add cell state column
        df['state'] = df['CellType']+'_'+df['DefinedName'] #add new column with cell type and defined function together
    
        #subset to just T cells
        dfTcells = df[df['CellType'].isin(tList)]
    
        #new df for each csv's t cells which will get concatenated with all the other csv's dfs
        dfBarcode = pd.DataFrame()
        dfBarcode['state'] = dfTcells['state']
        dfBarcode['file'] = file
    
        #now make all columns strings - to be able to concatenate
        dfTcells = dfTcells.astype(str)
    
        #add columns that concatenates all 10 binary marker vaules - create barcode, separated by '_'
        dfBarcode['barcode'] = dfTcells[['Cellsp_PD1p','Cellsp_TOXp','Cellsp_TIM3p','Cellsp_LAG3p','Cellsp_CD39p','Cellsp_EOMESp','Cellsp_CD38p','Cellsp_CD44p','Cellsp_TCF17p','Cellsp_TBETp']].agg('_'.join, axis=1) #without ki67, grzb -> this column is used in ML
        dfBarcode['barcodeKG'] = dfTcells[['Cellsp_PD1p','Cellsp_TOXp','Cellsp_TIM3p','Cellsp_LAG3p','Cellsp_CD39p','Cellsp_EOMESp','Cellsp_CD38p','Cellsp_CD44p','Cellsp_TCF17p','Cellsp_TBETp','Cellsp_KI67p','Cellsp_GRZBp']].agg('_'.join, axis=1) #with ki67, grzb -> this column is used in RCN analysis
    
        dfBarcodeMaster = pd.concat([dfBarcodeMaster,dfBarcode])
    
    #save dfBarcodeMaster as csv for later use
    dfBarcodeMaster.to_csv(path+'/results/dfCreated/dfBarcodeMaster.csv')    
    
    #now groupby IRcombo and save counts and densities of each functionality barcode per file
    ##continuation of above block
    group = dfBarcodeMaster.groupby('barcode')
    
    #get counts of files for each unique barcode
    groupCohortCounts = group.apply(lambda x: x['file'].value_counts())
    dfGroup = pd.DataFrame(groupCohortCounts)
    
    dfBarcodeCounts = dfGroup.unstack().T.fillna(0)
    
    #save as csv
    dfBarcodeCounts.to_csv(path+'/results/dfCreated/dfBarcodeCounts.csv')
    
    #get areas to convert counts to densities
    areaList = [] #empty list to store correct order of csv areas
    
    #get region tissue areas
    dfArea = pd.read_csv(path+'/data/metadata/dfArea.csv',index_col=0)
    
    for file in dfBarcodeCounts.index:
        area = dfArea.loc[file[1],'area'] #get region's corresponding area
        areaList.append(area)
    
    #add correspoindng csv's area to the csv
    dfBarcodeCounts['area'] = areaList
    
    #divide each count by area to get density of t cells with specific IR barcode
    dfBarcodeDensity = dfBarcodeCounts.iloc[:,:-1].div(dfBarcodeCounts['area'], axis=0)
    
    #save as csv
    dfBarcodeDensity.to_csv(path+'/results/dfCreated/dfBarcodeDensity.csv')
    print("T cell functionality barcodes calculated.")
    
    #now plot fig 2B
    #load dfBarcodeDensity
    df = pd.read_csv(path+'/results/dfCreated/dfBarcodeDensity.csv',index_col=0)
    
    #set file name to be index
    df = df.set_index('Unnamed: 1')
    
    #subset to just T, IA, TAS, NAP regions
    areaList = ['T','IA','TAS','NAP']
    patternH = re.compile(r".*_.*_(.*)_.*")
    
    #get hist annotation
    histList = []
    for roi in df.index:
        hist = re.match(patternH,roi).group(1)
        histList.append(hist)
    
    df['hist'] = histList    
    
    #add a column for the tx cohort
    df['tx'] = np.where(df.index.str.contains('V'),'aCD40','naive')
    
    #plot avg density of barcoded cells across tx cohorts and histopath sites
    dfAvg = pd.DataFrame()
    
    for h in areaList:
        #fist subset to histopath type
        dfH = df[df['hist'] == h]
        #then get histopath average cell densities for each tx cohort
        dfTx = dfH.groupby('tx').mean(numeric_only=True)
        dfTx['hist'] = h
        dfAvg = pd.concat([dfAvg,dfTx])
        
    #add back tx column
    dfAvg['tx'] = dfAvg.index
    
    #plot separate plots for each histopath site
    histRangeDict = {'T':150,'IA':1070,'TAS':155,'NAP':70}
    
    for h in dfAvg['hist'].unique():
        #subset to specific histopath site
        dfAvgH = dfAvg[dfAvg['hist'] == h]
        #get range_y for corresponding hist
        rangeY = histRangeDict[h]
    
        #get df of top 15 barcodes for each tx cohort
        dfTopN = dfAvgH.iloc[:,:-2].T.nlargest(n=15,columns=['naive'])
        dfTop40 = dfAvgH.iloc[:,:-2].T.nlargest(n=15,columns=['aCD40'])
    
        #add translated column
        dfTopN['barcode'] = dfTopN.index
        dfTop40['barcode'] = dfTop40.index
    
        #create function to translate barcode into new columns
        def barTranslate(b):
    
            # barcode order = 'PD1+_TOX+_TIM3+_LAG3+_CD39+_EOMES+_CD38+_CD44+_TCF1/7+_TBET+'
            barcodeKeyDict = {0:'PD1+',1:'TOX+',2:'TIM3+',3:'LAG3+',4:'CD39+',5:'EOMES+',6:'CD38+',7:'CD44+',8:'TCF1/7+',9:'TBET+'}
    
            #translate barcode to proteins
            barcode = b.replace('_', '') #drop _ in barcode
    
            #start with empty string to add to
            markerString = ''
    
            for i in range(len(barcode)):
                binary = barcode[i] #check if it's a 0 or 1
                if binary == '1': #if it's 1, get the correspoinding marker at that index
                    marker = barcodeKeyDict[i]
                    markerString = markerString+' '+marker
    
            if markerString == '': #if no markers are positive
                markerString = 'Negative for all'
    
            return markerString #return the translated barcode
    
        #translate the barcode
        dfTopN['barcodeT'] = dfTopN['barcode'].apply(barTranslate)
        dfTop40['barcodeT'] = dfTop40['barcode'].apply(barTranslate)
    
        #compare the top barcodes from each tx cohort so you can color code accordingly in plots
        bothBarList = list(set(dfTopN['barcodeT']).intersection(set(dfTop40['barcodeT'])))
    
        #add color coding column
        dfTopN['color'] = np.where(dfTopN['barcodeT'].isin(bothBarList),'both','diff')
        dfTop40['color'] = np.where(dfTop40['barcodeT'].isin(bothBarList),'both','diff')
    
        colorMapN = {'both':'#8C564B','diff':'#FF7F0E'}
        colorMap40 = {'both':'#8C564B','diff':'#1F77B4'}
    
        #plot density bars of top 15 barcodes from each tx cohort separately
        fig = px.bar(dfTopN,x='barcodeT',y='naive',title='Tx-Naive: '+h,range_y=[0,rangeY],color='color',color_discrete_map=colorMapN,labels={'naive':'Density (cells/mm2)','barcodeT':'Top 15 T Cell Dysfunctional Barcodes'},width=400,height=500)
    #     fig.update_layout(xaxis = {'categoryorder':'total descending','visible':False,'showticklabels':False})
        fig.update_layout(xaxis = {'categoryorder':'total descending'})
        fig.update_layout(xaxis_title=None)
        fig.write_image(path+'/results/figures/figure2B_Naive_'+h+'.png')
    
        fig = px.bar(dfTop40,x='barcodeT',y='aCD40',title='aCD40: '+h,range_y=[0,rangeY],color='color',color_discrete_map=colorMap40,labels={'aCD40':'Density (cells/mm2)','barcodeT':'Top 15 T Cell Dysfunctional Barcodes'},width=400,height=500)
    #     fig.update_layout(xaxis = {'categoryorder':'total descending','visible':False,'showticklabels':False})
        fig.update_layout(xaxis = {'categoryorder':'total descending'})
        fig.update_layout(xaxis_title=None)
        fig.write_image(path+'/results/figures/figure2B_aCD40_'+h+'.png')    

    ##FIGURE 2C - SPATIAL INTERACTIONS
    
    #suppress pandas performance warning when creating dfNormI - confirmed there is no issue
    warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)

    #first create dfInteractions
    #set distance threshold
    distThresh = 40 #40px = 20µm
    
    #create column list with correct naming conventions - in alphabetical order of the cell pairs (use colOrder defined above)
    colList = []
    for i in range(len(colOrder)):
        c1 = colOrder[i]
        for j in range(len(colOrder))[i:]:
            c2 = colOrder[j]        
            colName = c1+'_'+c2+'_'+str(distThresh)
            colList.append(colName)
    
    #create empty df to store interactions later
    dfInteractions = pd.DataFrame(index=csvList,columns=colList)
    
    def sortStates(cell1,cell2):
        orderDict = dict(enumerate(colOrder))
        for o,c in orderDict.items():
            if c == cell1:
                o1 = o
            if c == cell2:
                o2 = o
        if o1 <= o2:
            return cell1, cell2
        else:
            return cell2, cell1
    
    print('Calculating spatial interactions. This will likely take several (~3+) hours...')
    
    
    #keep track of progress
    progress = 0
    
    for file in csvList:            
        
        #read csv
        df = pd.read_csv(path+'/data/mIHC_files/'+file+'.csv',index_col=0)
        df = df[df['CellType'] != 'Other Cells'] #drop other cells from downstream analyses
        df['state'] = df['CellType']+'_'+df['DefinedName'] #add new column with cell type and defined function together (state)
    
        #get x,y values in array and create distance matrix
        coordArray = df[['Location_Center_X','Location_Center_Y']].values
        matrix = pd.DataFrame(distance_matrix(coordArray,coordArray),index=df.index,columns=df.index)
    
        #create pandas series of multi-index//distance value for just one half of matrix (lower triangle)
        v = matrix.values
        i, j = np.tril_indices_from(v,-1)
        matS = pd.Series(v[i,j], [matrix.index[i], matrix.columns[j]])
    
        #get indices of values less than X distance
        neighIdxList = list(matS[matS.le(distThresh)].index.values)
    
        #empty list to store sorted tuples of neighboring cells
        neighCellList = []
    
        #print corresponding cell state of neighboring cells from their indices; map back to original df
        for i in neighIdxList:
            idx1,idx2 = i
            cell1 = df.loc[idx1,'state']
            cell2 = df.loc[idx2,'state']
    
            #add neighbor pair's cell types sorted to list
            neighCellList.append(sortStates(cell1,cell2)) #first sort tuple of (cell1,cell2), then put it back into a tuple form
    
        neighFreqDict = Counter(neighCellList)
    
        #save neighFreqDict as dataframe, first adjust syntax for column names
        neighFreqDict2 = {}
        for pair,count in neighFreqDict.items():
            colName = pair[0]+'_'+pair[1]+'_'+str(distThresh)
            neighFreqDict2[colName] = count
        
        #add data for the roi to dfInteractions
        dfInteractions.loc[file] = neighFreqDict2
        
        #print progress based on how many regions have been analyzed
        progress = progress + 1
        if progress == 75:
            print('~25% complete...')
        if progress == 150:
            print('~50% complete...')
        if progress == 225:
            print('~75% complete...')
    
    print('100% complete.')
    #na values given to pairs with no interactions, fill them with zeros instead
    dfInteractions = dfInteractions.fillna(0)
    
    dfInteractions.to_csv(path+'/results/dfCreated/dfInteractions.csv')
    print("Spatial interactions calculated.")
    
    #now plot interaction matrices, first normalize by density of cells involved in interaction
    #load dfInteractions
    df = pd.read_csv(path+'/results/dfCreated/dfInteractions.csv',index_col=0)
    
    #normalize by density, divide raw interactions count by summed density of cell types involved
    #read density file
    dfAll = pd.read_csv(path+'/results/dfCreated/dfDensity.csv',index_col=0)
    
    #create empty df to store normalized interaction amounts
    dfNormI = pd.DataFrame()
    
    #divide each interaction count by summed densities of cells types involved
    pattern1 = re.compile(r"(.*_.*)_.*_.*_..")
    pattern2 = re.compile(r".*_.*_(.*_.*)_..")
    
    for col in df.columns:
        cell1 = re.match(pattern1,col).group(1)
        cell2 = re.match(pattern2,col).group(1)
    
        for roi in df.index:
            #get summed densities for that region
            dens1 = dfAll.loc[roi,cell1]
            dens2 = dfAll.loc[roi,cell2]
    
            if cell1 != cell2: #if not the same cell types, then add densities
                tot = dens1 + dens2
            elif cell1 == cell2: #if same cell types, just use one of the densities
                tot = dens1
    
            #normalize = interactions/density total
            ints = df.loc[roi,col]
            if ints == 0:
                normI = 0
            else:
                normI = ints/tot
    
            #add roi,normalized value to new df
            dfNormI.loc[roi,col] = normI
    
    #add a column for the tx cohort
    dfNormI['tx'] = np.where(dfNormI.index.str.contains('V'),'aCD40','naive')
    
    #add hist type column
    histList = []
    patternH = re.compile(r".*_.*_(.*)_.*")
    
    for roi in dfNormI.index:
        hist = re.match(patternH,roi).group(1)
        histList.append(hist)
    dfNormI['hist'] = histList
    
    #plot average normalized interactions across tx cohorts and histopath sites
    dfAvg = pd.DataFrame()
    
    for h in ['T','IA','TAS','NAP']:
        #fist subset to histopath type
        dfH = dfNormI[dfNormI['hist'] == h]
        #then get histopath average cell densities for each tx cohort
        dfTx = dfH.groupby('tx').mean(numeric_only=True)
        dfTx['hist'] = h
        dfAvg = pd.concat([dfAvg,dfTx])    
        
    #create dict of hist:colorbar scale maxval for plotting
    scaleDict = {'T':1.35,'IA':0.35,'TAS':1.08,'NAP':0.93}
        
    #Create empty dfMirror to capture heatmap data
    dfMirror = pd.DataFrame(index=colOrder,columns=colOrder)
        
    #loop through each row of dfAvg and create a heatmap of the normalized interaction amounts for each tx cohort/histopath site
    for index, row in dfAvg.iterrows():
                
        #patterns to identify cells involved in interaction
        pattern1 = re.compile(r"(.*_.*)_.*_.*_..")
        pattern2 = re.compile(r".*_.*_(.*_.*)_..")
    
        for cells,val in row.items():
            
            #don't do anything w/ the histopath column
            if cells == 'hist':
                break
    
            #get the cell types involved in the interaction
            cell1 = re.match(pattern1,cells).group(1)
            cell2 = re.match(pattern2,cells).group(1)
    
            #update dfMirror with interaction values - need to make sure both sides of table get filled w/ same values
            dfMirror.loc[cell1,cell2] = val 
            dfMirror.loc[cell2,cell1] = val
    
        #now plot dfMirror as heatmap
        #convert all values to be floats
        dfMirror = dfMirror.apply(pd.to_numeric, errors='coerce')
    
        #log10+1 correct dfMirror
        dfMirror = np.log10(dfMirror+1)
    
        #create graph layout
        fig, graph = plt.subplots(figsize=(12,10))
        
        #get corresponding color bar scale maxVal for given histopath site
        maxVal = scaleDict[row['hist']]
        
        #plot the clusttermap with the colors
        graph = sns.heatmap(data=dfMirror,yticklabels=True,xticklabels=True,cmap='crest',vmin=0, vmax=maxVal)
    
        #add x-axis label and title to graph
        graph.set_title('Average Cell-Cell Interactions for '+index+': '+row['hist'])
    
        plt.savefig(path+'/results/figures/figure2C_'+index+'_'+row['hist']+'.png',format='png',bbox_inches='tight')
        plt.close()
            
    #Create supplementary tables S3, S4, S5 and save to 'tables' folder    
    #load RAW COUNT files for cell states, barcodes, interactions    
    #cell state counts file
    df1 = pd.read_csv(path+'/results/dfCreated/dfCounts.csv',index_col=0)
    #barcode counts and reset index
    df2 = pd.read_csv(path+'/results/dfCreated/dfBarcodeCounts.csv',index_col=0).set_index('Unnamed: 1')
    #interaction counts
    df3 = pd.read_csv(path+'/results/dfCreated/dfInteractions.csv',index_col=0)
    
    #update df2 - translate barcode into marker expression
    def barTranslate(b):
        # barcode order = 'PD1+_TOX+_TIM3+_LAG3+_CD39+_EOMES+_CD38+_CD44+_TCF1/7+_TBET+'
        barcodeKeyDict = {0:'PD1+',1:'TOX+',2:'TIM3+',3:'LAG3+',4:'CD39+',5:'EOMES+',6:'CD38+',7:'CD44+',8:'TCF1/7+',9:'TBET+'}
        #translate barcode to proteins
        barcode = b.replace('_', '') #drop _ in barcode
        #start with empty string to add to
        markerString = ''
        for i in range(len(barcode)):
            binary = barcode[i] #check if it's a 0 or 1
            if binary == '1': #if it's 1, get the correspoinding marker at that index
                marker = barcodeKeyDict[i]
                markerString = markerString+' '+marker
        if markerString == '': #if no markers are positive
            markerString = 'Negative for all'
        return markerString #return the translated barcode
    
    #now translate the barcode
    newColMapDict = {}
    for col in df2.columns:
        barT = barTranslate(col) #call translate function
        newCol = barT
        newColMapDict[col] = newCol
    df2 = df2.rename(columns=newColMapDict)
    
    #sum across all ROIs
    df1Sum = df1.sum(axis=0).sort_values(ascending=False)
    df2Sum = df2.sum(axis=0).sort_values(ascending=False)
    df3Sum = df3.sum(axis=0).sort_values(ascending=False)[:-8] #the bottom 8 are zeros (they are not included in ML models)
    
    #save summed dfs as csv files in tables folder
    df1Sum.to_csv(path+'/results/tables/SupplementaryTable3_CellStates.csv')
    df2Sum.to_csv(path+'/results/tables/SupplementaryTable4_Barcodes.csv')
    df3Sum.to_csv(path+'/results/tables/SupplementaryTable5_Interactions.csv')
    
    print("Figures 2A-2C saved to 'figures' folder.")
    print("Supplementary Tables S3-S5 saved to 'tables' folder.")
    print('Figure 2 complete.')
    
    

def fig3():
    '''
    Machine learning models classify aCD40-treated TMEs as having reduced T cell exhaustion phenotypes.
    This function creates figures 3A-3C, 3E and supplemental figures S2A-S2E
    Input parameters:
        None   
    Outputs:
        Saves csv to 'dfCreated' folder for file containing feature importance results
        Saves plots to 'figures' folder for figures 3A-3C, 3E and supplemental figures S2A-S2E
    '''

    import os
    import pandas as pd
    import warnings
    
    #ignore df concatenation warnings - just a performance warning and perf is fine
    warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)
    
    print('\n\n***FIGURE 3 - machine learning to predict treatment status***\n')

    #get current working directory
    path = os.getcwd()
    figNum = 'fig3'
        
    makeClassifierFigures(path=path,figNum=figNum)
    makeBarcodeMatrix(path=path,figNum=figNum)
  
    print("\nFigures 3A-3C, 3E and Supplementary Figures S2A-S2E saved to 'figures' folder.")
    print('Figure 3 complete.')
    
    

def fig4():
    '''
    Machine learning model classifies long disease-free survivors as having more T cell effector functionality following aCD40 therapy.
    This function creates figures 4A-4C, 4E and supplemental figures S3A, S3B
    Input parameters:
        None 
    Outputs:
        Saves csv to 'dfCreated' folder for file containing feature importance results
        Saves plots to 'figures' folder for figures 4A-4C, 4E and supplemental figures S3A, S3B
    '''

    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    
    #ignore df concatenation warnings - just a performance warning and perf is fine
    warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)

    print('\n\n***FIGURE 4 - machine learning to predict DFS***\n')

    #get current working directory
    path = os.getcwd()
    figNum = 'fig4'
    plt.style.use('default') #revert to default matplotlib style, otherwise figures have grey background
    
    
    makeClassifierFigures(path=path,figNum=figNum)
    makeBarcodeMatrix(path=path,figNum=figNum)
    
    print("\nFigures 4A-4C, 4E and Supplementary Figures S3A-S3E saved to 'figures' folder.")
    print('Figure 4 complete.')
    
    
    
def fig5():
    '''
    Cellular neighborhood analysis identifies spatial organization of T cells to correlate with DFS following aCD40 therapy.
    This function creates figures 5B-5H and supplemental figures S4A-S4F
    Input parameters:
        None
    Outputs:
        Saves csvs to 'dfCreated' folder for files containing recurrent cellular neighborhood results
        Saves plots to 'figures' folder for figures 5B-5H and supplemental figures S4A-S4F
    '''
    
    import os
    import glob
    import pandas as pd
    import plotly.express as px
    import plotly
    import numpy as np
    import re
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib import pyplot as plt

    print('\n\n***FIGURE 5 - recurrent cellular neighborhoods***\n')

    #get current working directory
    path = os.getcwd()

    #revert to default matplot.ib style, otherwise figures have grey background
    plt.style.use('default')
    
    #get list of aCD40 IA regions
    csvList = [os.path.basename(x)[0:-4] for x in glob.glob(path+'/data/mIHC_files/V[0123456789]*_IA_*.csv')]
    csvList.sort()

    #make neighborhoods for all cells in aCD40 IA regions
    makeNeighborhoods(path=path,csvList=csvList)

    #run elbow method
    elbowMethod(path=path,file='dfNeighborhoods',steps=21)

    #cluster neighborhoods into 7 RCNs
    clusterNeighborhoods(path=path,file='dfNeighborhoods',k=7)
    
    
    #supplemental figures S4B,C
    #get counts/% of cells assigned to each cluster    
    df = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)
    #update cluster numbers to match figs
    df['cluster'] = df['cluster'].replace({0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'})
    cols = ['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7']
    dfRaw = pd.DataFrame(dict(df['cluster'].value_counts()),index=['raw'],columns=cols).T
    dfPerc = pd.DataFrame(dict(df['cluster'].value_counts(normalize=True)),index=['perc'],columns=cols).T
    
    fig = px.bar(dfRaw,labels={'index':'RCN','value':'Number of Cells'})
    fig.write_image(path+'/results/figures/figureS4B.png')
    
    fig = px.bar(dfPerc*100,labels={'index':'RCN','value':'Percentage of Cells'})
    fig.write_image(path+'/results/figures/figureS4C.png')    
    
    #plot counts per region, patient - supp figs S4D, E
    clusterCountFrequency(path=path)
    
    #Figure 5B - average makeup of the 7 RCNs in stacked bar chart    
    #use to reorder columns in for loop
    colList = ['countCD8 T Cells_Naive%',
                         'countCD8 T Cells_T Effector%',
                         'countCD8 T Cells_Tem%',
                         'countCD8 T Cells_Temra%',
                         'countCD8 T Cells_Early Exhaustion%',
                         'countCD8 T Cells_Terminal Exhaustion%',
                         'countCD8 T Cells_Other CD44-%',
                         'countCD8 T Cells_Other CD44+%',
                         'countCD4 Th1 helper Cells_T Effector%',
                         'countCD4 Th1 helper Cells_Tem%',
                         'countCD4 Th1 helper Cells_Temra%',
                         'countCD4 Th1 helper Cells_Other CD44-%',
                         'countCD4 Th1 helper Cells_Other CD44+%',
                         'countOther CD4 Th Cells_Other CD44-%',
                         'countOther CD4 Th Cells_Other CD44+%',
                         'countTregs_Naive%',
                         'countTregs_mTreg%',
                         'countTregs_ExTreg%',
                         'countB Cells_Total%',
                         'countMyeloid Cells_Total%',
                         'countFibroblasts_Total%',
                         'countTumor_Ki67-%',
                         'countTumor_Ki67+%']
    
    #to match color scheme from barcode analysis
    colorList = ['Other CD4 Th Cells_Other CD44-', 'Tregs_ExTreg',
           'CD8 T Cells_Other CD44-', 'Tregs_Naive',
           'CD4 Th1 helper Cells_Other CD44-', 'Tregs_mTreg',
           'Other CD4 Th Cells_Other CD44+', 'CD8 T Cells_Temra',
           'CD8 T Cells_Other CD44+', 'CD8 T Cells_Tem',
           'CD4 Th1 helper Cells_Tem', 'CD8 T Cells_Naive',
           'CD4 Th1 helper Cells_Other CD44+', 'CD4 Th1 helper Cells_Temra',
           'CD4 Th1 helper Cells_T Effector', 'CD8 T Cells_Early Exhaustion',
           'CD8 T Cells_T Effector', 'CD8 T Cells_Terminal Exhaustion',
            'B Cells_Total',
            'Myeloid Cells_Total',
            'Fibroblasts_Total',
            'Tumor_Ki67-',
            'Tumor_Ki67+']
    
    palette = dict(zip(colorList,plotly.colors.qualitative.Dark24)) 
    
    #dictionary to drop the 'count' and '%' characters
    newColDict = dict(zip(colList,['CD8 T Cells_Naive',
                                 'CD8 T Cells_T Effector',
                                 'CD8 T Cells_Tem',
                                 'CD8 T Cells_Temra',
                                 'CD8 T Cells_Early Exhaustion',
                                 'CD8 T Cells_Terminal Exhaustion',
                                 'CD8 T Cells_Other CD44-',
                                 'CD8 T Cells_Other CD44+',
                                 'CD4 Th1 helper Cells_T Effector',
                                 'CD4 Th1 helper Cells_Tem',
                                 'CD4 Th1 helper Cells_Temra',
                                 'CD4 Th1 helper Cells_Other CD44-',
                                 'CD4 Th1 helper Cells_Other CD44+',
                                 'Other CD4 Th Cells_Other CD44-',
                                 'Other CD4 Th Cells_Other CD44+',
                                 'Tregs_Naive',
                                 'Tregs_mTreg',
                                 'Tregs_ExTreg',
                                 'B Cells_Total',
                                 'Myeloid Cells_Total',
                                 'Fibroblasts_Total',
                                 'Tumor_Ki67-',
                                 'Tumor_Ki67+']))
        
    #read clustered file
    df = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)
    
    #filter df to not include the additional annotation columns
    df = df.drop(['index','file','seed state','barcode','barcodeKG','neighT idx list'],axis=1)
    
    #groupby cluster column and take the averages of all of the other columns for each group
    dfCluster = df.groupby(['cluster']).mean()
    
    #reorder columns
    dfCluster = dfCluster[colList]
    
    #rename columns to drop the 'count' and '%' characters
    dfCluster = dfCluster.rename(columns=newColDict)
    
    #reorder clusters to match prior order
    dfCluster = dfCluster.rename(index={0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'})
    dfCluster = dfCluster.sort_index()
    
    #plot stacked bars - Figure 5B
    fig = px.bar(dfCluster,y=dfCluster.columns,barmode='stack',labels={'cluster':'Recurrent Cellular Neighborhood','value':'Average Cellular Composition'},color_discrete_map=palette)
    fig.update_layout(legend_traceorder="reversed",bargap=0.1)
    fig.write_image(path+'/results/figures/figure5B.png')    
    
    #create new csvs for each aCD40 IA region with cluster (RCN) assignment
    createCsvsWithClusterCol(path=path,csvList=csvList)    

    #Create the scatter plots of cell locations for supplemental Figure S4F and Figure 5C, 5H
    #cluster color coding
    clustPalette = dict(zip(['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7'],plotly.colors.qualitative.Plotly))
    
    #dictionary storing corresponding RCN assignment for each original cluster number
    mapDict = {'0.0':'RCN1','1.0':'RCN4','2.0':'RCN7','3.0':'RCN6','4.0':'RCN2','5.0':'RCN5','6.0':'RCN3'}

    #get sorting of all cell states for plotting original scatter
    def sortState(column):
        """Sort function"""
        colList = ['CD8 T Cells_Naive',
                     'CD8 T Cells_T Effector',
                     'CD8 T Cells_Tem',
                     'CD8 T Cells_Temra',
                     'CD8 T Cells_Early Exhaustion',
                     'CD8 T Cells_Terminal Exhaustion',
                     'CD8 T Cells_Other CD44-',
                     'CD8 T Cells_Other CD44+',
                     'CD4 Th1 helper Cells_T Effector',
                     'CD4 Th1 helper Cells_Tem',
                     'CD4 Th1 helper Cells_Temra',
                     'CD4 Th1 helper Cells_Other CD44-',
                     'CD4 Th1 helper Cells_Other CD44+',
                     'Other CD4 Th Cells_Other CD44-',
                     'Other CD4 Th Cells_Other CD44+',
                     'Tregs_Naive',
                     'Tregs_mTreg',
                     'Tregs_ExTreg',
                     'B Cells_Total',
                     'Myeloid Cells_Total',
                     'Fibroblasts_Total',
                     'Tumor_Ki67-',
                     'Tumor_Ki67+']
        sortDict = {col: order for order, col in enumerate(colList)}
        return column.map(sortDict)
    
    def sortCluster(column):
        """Sort function"""
        colList = ['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7']
        sortDict = {col: order for order, col in enumerate(colList)}
        return column.map(sortDict)
    
    #loop through each region
    for file in csvList:
    
        #read updated csv with cluster number added
        df = pd.read_csv(path+'/results/dfCreated/updatedCsvs/'+file+'_RCN.csv',index_col=0)
        
        #update cluster numbers with corresponding RCN numbers (to match other results, since they get reordered)
        df['cluster'] = df['cluster'].replace(mapDict)

        
        #drop other cells and cells not assigned to a cluster (no neighbors)
        df = df[df['CellType'] != 'Other Cells']
        df = df[df['cluster'].isin(['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7'])]
    
        #add state column to get child level cell state defn
        df['state'] = df['CellType']+'_'+df['DefinedName']            
    
        df = df.sort_values(by='cluster', key=sortCluster)
    
        #show plot of neighborhood clusters - each color is a neighborhood
        fig = px.scatter(df,x='Location_Center_X',y='Location_Center_Y',color='cluster',color_discrete_map=clustPalette,hover_name=df.index,width=600,height=500)
        fig.update(layout_showlegend=False)
        fig.update_traces(marker={'size': 6})
        fig.update_layout(xaxis={'visible': False},yaxis={'visible': False},margin={'l':0,'r':0,'t':0,'b':0})
    
        #save scatters to 'figure S4F' folder
        fig.write_image(path+'/results/figures/figureS4F/'+file+'_RCN.png')
        
        #save representative region as Figure 5C
        if file == 'V1711_ROI01_IA_D':
            fig.write_image(path+'/results/figures/figure5C_RCN.png')
    
        #save RCN scatter used in Figure 5H
        if file == 'V7520_ROI03_IA_I':
            fig.update_layout(height=300)
            fig.write_image(path+'/results/figures/figure5H_Scatter.png')
    
        #sort all cell states
        df = df.sort_values(by='state', key=sortState)
    
        #plot all original cell states
        fig = px.scatter(df,x='Location_Center_X',y='Location_Center_Y',color='state',color_discrete_map=palette,hover_name=df.index,width=600,height=500)
        fig.update(layout_showlegend=False)
        fig.update_traces(marker={'size': 6})
        fig.update_layout(xaxis={'visible': False},yaxis={'visible': False},margin={'l':0,'r':0,'t':0,'b':0})
        
        #SAVE scatter
        fig.write_image(path+'/results/figures/figureS4F/'+file+'_Original.png')
        
        #save representative region as Figure 5C
        if file == 'V1711_ROI01_IA_D':
            fig.write_image(path+'/results/figures/figure5C_Original.png')
                
    #Figure 5D, E
    #create function to translate barcode into new columns
    def barTranslate(b):
        # barcode order = 'PD1+_TOX+_TIM3+_LAG3+_CD39+_EOMES+_CD38+_CD44+_TCF1/7+_TBET+'
        barcodeKeyDict = {0:'PD1+',1:'TOX+',2:'TIM3+',3:'LAG3+',4:'CD39+',5:'EOMES+',6:'CD38+',7:'CD44+',8:'TCF1/7+',9:'TBET+'}
        #translate barcode to proteins
        barcode = b.replace('_', '') #drop _ in barcode
        #start with empty string to add to
        markerString = ''
        for i in range(len(barcode)):
            binary = barcode[i] #check if it's a 0 or 1
            if binary == '1': #if it's 1, get the correspoinding marker at that index
                marker = barcodeKeyDict[i]
                markerString = markerString+' '+marker
        if markerString == '': #if no markers are positive
            markerString = 'Negative for all'
        return markerString #return the translated barcode
    
    #read neighborhood clusters csv
    df = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)
    
    #translate the barcode
    df['barcodeT'] = df['barcode'].apply(barTranslate)
    
    #get df with all features listed and subset to barcode features only
    dfAllFeat = pd.read_csv(path+'/results/dfCreated/dfTopFeatures_fig4.csv',index_col=0)
    barList = [b[9:] for b in dfAllFeat[dfAllFeat['Feature'].str.contains('Barcode')]['Feature']]
    dfBar = df[df['barcodeT'].isin(barList)]
    
    #add DFS labels to dfBar
    #first get dfClin to calculate long vs short
    dfClin = pd.read_csv(path+'/data/metadata/clinicalData.csv',index_col=0)
    
    #subset to just desired tx cohort based on histopath type
    dfClin = dfClin[dfClin['tx'] == 'aCD40']
    #now calculate median DFS time across cohort
    med = dfClin['DFS'].median() #get median of cohort survival - based on pt average in dfClin
    #add short/long label to dfClin patients
    dfClin['MLclass'] = np.where(dfClin['DFS'] <= med,'Short DFS','Long DFS') #0=long survival, 1=short survival
    
    #match patient w/ file
    patternP = re.compile(r"(.*)_.*_.*_.*") #patient ID
    ptList = []
    
    for file in dfBar['file']:
        pt = re.match(patternP,file).group(1)
        ptList.append(pt)
    
    #add column for patient
    dfBar['pt'] = ptList
    
    #merge on pt to get the ML class
    dfBar = dfBar.merge(dfClin['MLclass'],left_on='pt',right_on='sample')
    
    #subset into 2 dfs - raw counts of barcodes in each cluster
    dfBarS = dfBar[dfBar['MLclass'] == 'Short DFS']
    dfBarL = dfBar[dfBar['MLclass'] == 'Long DFS']
    
    dfBarClustS = dfBarS.groupby(['barcodeT','cluster']).size().unstack(fill_value=0)
    dfBarClustL = dfBarL.groupby(['barcodeT','cluster']).size().unstack(fill_value=0)
    
    dfBarClustS = dfBarClustS.rename({0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'},axis=1)
    dfBarClustL = dfBarClustL.rename({0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'},axis=1)
    
    #now check for all clusters present in the DFS long or short df
    kList = ['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7'] #list of all clusters that should exist
    
    #reorder columns in short df to be 1-7
    dfBarClustS = dfBarClustS[kList]
    
    #prior knowledge- barcodes in dfBarClustL do not exist in RCN7, so need to add column with zeros for it
    dfBarClustL['RCN7'] = 0
    #reorder columns
    dfBarClustL = dfBarClustL[kList]
    
    #now convert raw counts to % for each df
    #short DFS
    dfBarClustS['total'] = dfBarClustS.sum(axis=1) #create total column
    dfBarClustL['total'] = dfBarClustL.sum(axis=1) #create total column
    for c in dfBarClustS.columns[:-1]:
        dfBarClustS[str(c)+'%'] = dfBarClustS[c]/dfBarClustS['total']*100 
        dfBarClustL[str(c)+'%'] = dfBarClustL[c]/dfBarClustL['total']*100
    
    #add DFS column for color coding
    dfBarClustS['DFS'] = 'Short DFS'
    dfBarClustL['DFS'] = 'Long DFS'
    
    #add barcode column for color coding
    dfBarClustS['barcode'] = dfBarClustS.index
    dfBarClustL['barcode'] = dfBarClustL.index
    
    #reindex before concatenating
    dfBarClustS['reidx'] = dfBarClustS['DFS'] +': '+dfBarClustS['barcode']
    dfBarClustS.index = dfBarClustS['reidx']
    
    dfBarClustL['reidx'] = dfBarClustL['DFS'] +': '+dfBarClustL['barcode']
    dfBarClustL.index = dfBarClustL['reidx']
    
    dfBarClustBoth = pd.concat([dfBarClustS,dfBarClustL])
    
    #get color palettes
    palette1 = {'Short DFS':'purple','Long DFS':'green'} #short=purple,long=green
    colorBarList = ['#7f3c8d','#11a579','#3969ac','#f2b701','#e73f74','#80ba5a','#e68310','#008695','#cf1c90','#f97b72','#a5aa99','#964b00','#000000']
    palette2 = dict(zip(sorted(dfBarClustBoth['barcode'].unique()),colorBarList))
    grouping1 = dfBarClustBoth['DFS']
    grouping2 = dfBarClustBoth['barcode']
    colors1 = pd.Series(grouping1,name='DFS').map(palette1)
    colors2 = pd.Series(grouping2,name='Barcode').map(palette2)
    dfColors = pd.concat([colors1,colors2],axis=1)
    
    #drop the raw count columns and rename to drop the '%'s
    dfBarClustBoth = dfBarClustBoth[[col for col in dfBarClustBoth if '%' in col]]
    dfBarClustBoth = dfBarClustBoth.rename({'RCN1%':'RCN1','RCN2%':'RCN2','RCN3%':'RCN3','RCN4%':'RCN4','RCN5%':'RCN5','RCN6%':'RCN6','RCN7%':'RCN7'},axis=1)
    
    #plot the clusttermap with the colors - log10+1 normalize the %(out of 100)
    graph = sns.clustermap(np.log10(dfBarClustBoth+1),method='ward',metric='euclidean',cmap='mako_r',row_colors=dfColors,yticklabels = True,figsize=(12, 10))
    
    #add x-axis label and title to graph
    ax = graph.ax_heatmap
    ax.set_xlabel("\nRCN")
    ax.set_ylabel("Barcode")
    
    #add legends
    handles1 = [Patch(facecolor=palette1[c]) for c in palette1]
    leg1 = ax.legend(handles1, palette1, title='DFS', bbox_to_anchor=(1.15, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    handles2 = [Patch(facecolor=palette2[c]) for c in palette2]
    leg2 = ax.legend(handles2, palette2, title='Barcode', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='center right')
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    
    plt.savefig(path+'/results/figures/figure5D.png',format='png',bbox_inches='tight')
    plt.close()
    
    #Figure 5E
    rowList = list(graph.data2d.index) #get order of rows
    
    #divide rows into 2 clusters based on ordering in fig 5D
    c1List = rowList[:13]
    
    #add cluster column
    dfBarClustBoth['Cluster'] = np.where(dfBarClustBoth.index.isin(c1List),'C1 (Long DFS)','C2 (Short DFS)')
    
    dfBarClustBothGroup = dfBarClustBoth.groupby('Cluster').mean() #get average RCN composition for C1 vs C2
    fig = px.bar(dfBarClustBothGroup,labels={'value':'Average Fraction of RCNs','cluster':'RCN'},width=500)
    fig.update_layout(legend_traceorder="reversed",bargap=0.05)
    fig.write_image(path+'/results/figures/figure5E.png')    
    
    ##FIGURES 5F AND 5G - Ki67, GrzB analyses
    #first calculate overall T cell ki67, grzb positivity regardless of RCN assignment
    #read in master barcode file
    dfCombo = pd.read_csv(path+'/results/dfCreated/dfBarcodeMaster.csv',index_col=0)
    
    #add annotation columns to the df - pt tx hist
    patternP = re.compile(r"(.*)_.*_.*_.*") #patient ID
    patternH = re.compile(r".*_.*_(.*)_.*") #histopath
    
    ptList = []
    histList = []
    
    for file in dfCombo['file']:
        pt = re.match(patternP,file).group(1)
        ptList.append(pt)
        h = re.match(patternH,file).group(1)
        histList.append(h)
        
    dfCombo['pt'] = ptList
    dfCombo['tx'] = np.where(dfCombo['pt'].str.contains('V'),'aCD40','Tx-Naive')
    dfCombo['hist'] = histList
    
    #add ki67/grzb columns
    patternK = re.compile(r"._._._._._._._._._._(.)_.") #ki67 is the 11th marker
    patternG = re.compile(r"._._._._._._._._._._._(.)") #grzb is the 12th marker
    
    kList = []
    gList = []
    
    for barcode in dfCombo['barcodeKG']:
        ki67 = re.match(patternK,barcode).group(1)
        grzb = re.match(patternG,barcode).group(1)
        kList.append(ki67)
        gList.append(grzb)
        
    dfCombo['KI67'] = kList
    dfCombo['GRZB'] = gList
    
    #now subset to aCD40 IA regions
    dfCombo = dfCombo[(dfCombo['hist'] == 'IA') & (dfCombo['tx'] == 'aCD40')]
    
    #get perc of all T cells in aCD40 IA regions that are KI67+, GRZB+ (drop Tregs from GRZB analysis)
    #these values to be plotted as the solid line
    allPercK = (dfCombo['KI67'].value_counts(normalize=True)*100)['1'] #want value corresponnding to 1
    allPercGex = (dfCombo[~dfCombo['state'].str.contains('Tregs')]['GRZB'].value_counts(normalize=True)*100)['1'] #want value corresponding to 1
    
    ##now look at neighborhoods -> do the t cells living in the cd44+ t cell neighborhood have more ki67/grzb?
    #read clustered neighborhood csv
    df = pd.read_csv(path+'/results/dfCreated/dfNeighborhoodClusters.csv', index_col=0)
    
    #update cluster column to match rcn ordering in figures
    mapDict ={0:'RCN1',1:'RCN4',2:'RCN7',3:'RCN6',4:'RCN2',5:'RCN5',6:'RCN3'}
    df['cluster'] = df['cluster'].replace(mapDict)
    
    #empty dicts to store positive %s for each cluster/marker
    kPosPercDict = {}
    gExPosPercDict = {} #exclude tregs
    
    #get the functional capacity of the T cells assigned to each cluster
    rcnList = ['RCN1','RCN2','RCN3','RCN4','RCN5','RCN6','RCN7']
    for rcn in rcnList:    
        #subset df to just cells assigned to 1 RCN
        dfC = df[df['cluster'] == rcn]
    
        #subset to just t cells (that therefore have a barcode)
        dfC = dfC[dfC['barcodeKG'] != '--']
            
        #add ki67/grzb columns
        patternK = re.compile(r"._._._._._._._._._._(.)_.") #ki67 is the 11th marker
        patternG = re.compile(r"._._._._._._._._._._._(.)") #grzb is the 12th marker
    
        kList = []
        gList = []
    
        for b in dfC['barcodeKG']:
            ki67 = re.match(patternK,b).group(1)
            grzb = re.match(patternG,b).group(1)
            kList.append(ki67)
            gList.append(grzb)
    
        dfC['KI67'] = kList
        dfC['GRZB'] = gList
          
        #see how many are ki67, grzb positive
        ##KI67
        percK = dfC['KI67'].value_counts(normalize=True)*100
        
        ##GRZB - exclude tregs
        percGex = dfC[~dfC['seed state'].str.contains('Tregs')]['GRZB'].value_counts(normalize=True)*100
        
        #store positive %s to plot
        kPosPercDict[rcn] = percK['1']
        gExPosPercDict[rcn] = percGex['1']
    
    #store amounts in df to plot
    dfPosPerc = pd.DataFrame({'% KI67+':pd.Series(kPosPercDict),'% GRZB+':pd.Series(gExPosPercDict)})
    
    #plot % positive  
    #Figure 5F - ki67
    fig = px.bar(dfPosPerc,y='% KI67+',range_y=[0,6],labels={'index':'Recurrent Cellular Neighborhood','% KI67+':'Percent T Cells KI67+'},width=350,height=500)
    fig.update_xaxes(tickmode='linear')
    fig.add_hline(y=allPercK,line_dash='dash') #add a horizontal dashed line at the overall value
    fig.write_image(path+'/results/figures/figure5F.png')
    
    #Figure 5G - grzb
    fig = px.bar(dfPosPerc,y='% GRZB+',range_y=[0,6],labels={'index':'Recurrent Cellular Neighborhood','% GRZB+':'Percent T Cells GRZB+'},width=350,height=500)
    fig.update_xaxes(tickmode='linear')
    fig.add_hline(y=allPercGex,line_dash='dash') #add a horizontal dashed line at the overall value
    fig.write_image(path+'/results/figures/figure5G.png')

    print("Figures 5B-5H and Supplementary Figures S4A-S4F saved to 'figures' folder.")
    print('Figure 5 complete.')
    print('\nALL ANALYSES COMPLETE.\n')
    
    

if __name__=="__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
