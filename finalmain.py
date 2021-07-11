import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import LabelEncoder

st.title("Population Growth Prediction. A comparison between 3 different models")
dataset= st.sidebar.selectbox("Select Dataset", ("None","Population By Country 2020","World Population Prospects","Countries of the World","Fertility Data Set"))
models= st.sidebar.selectbox ("Select Model", ("None","KNN","SVM","Decision Tree"))
result= st.sidebar.selectbox("Results Comparison (All Model)",("None","Dataset 1","Dataset 2","Dataset 3","Dataset 4"))

if dataset == "None":
  st.write("This research examines the population growth on the economic development between the developed and developing countries. The effect of population growth can be positive or negative depending on the circumstances. A large population has the potential to be great for economic development: after all, the more people you have, the more work is done, and the more work is done, the more value. The objective of this is to predict the population growth; to benefits the country especially on economic perspective. Data used in this research is a public dataset collected from online platform such as Kaggle and UC Irvine Machine Learning Repository. Three algorithms are proposed to predict the growth.")

#Section Population by Country 2020
elif dataset == "Population By Country 2020":   
  st.write("Description: This dataset is about a population growth by the year 2020. The dataset used is a public dataset obtained from Kaggle. It contains over 235 samples data with 11 variables. Variables inputs involved is Yearly Change, Net Change, Migrants, Fertility Rate, Urban Population and World Share. We will predict the population growth by using the data target of Population (2020).")
  st.write("")
  st.write("The training set is consists of 80 percent (%) samples taken that has been manually divided prior to the process of training the model.")
  st.write("The testing set is consists of 20 percent (%) samples taken that has been manually divided prior to the process of testing the model. ")
  st.write('     ')
  st.write('     ')

  data_train = pd.read_csv('population_training.csv')  
  input_data_train = data_train.drop(columns=['No','Country (or dependency)','Population (2020)','Density (P/KmÂ²)','Land Area (KmÂ²)'])
  target_data_train = data_train['Population (2020)']

  data_test = pd.read_csv('population_testing.csv')
  input_data_test = data_test.drop(columns =['No', 'Country (or dependency)','Population (2020)','Density (P/KmÂ²)','Land Area (KmÂ²)'])
  target_data_test = data_test['Population (2020)']

  #KNN MODEL
  regLinear = SVR(kernel = 'linear')
  regLinear.fit(input_data_train,target_data_train)
  outputLinear = regLinear.predict(input_data_test)
  mseLinear = mean_squared_error(outputLinear,target_data_test)

  regSigmoid = SVR (kernel ='sigmoid')
  regSigmoid.fit(input_data_train,target_data_train)
  outputSigmoid = regSigmoid.predict(input_data_test)
  mseSigmoid = mean_squared_error (outputSigmoid,target_data_test)

  regPoly = SVR (kernel ='poly')
  regPoly.fit(input_data_train,target_data_train)
  outputPoly = regPoly.predict(input_data_test)
  msePoly = mean_squared_error (outputPoly,target_data_test)

  regRBF = SVR (kernel ='rbf')
  regRBF.fit(input_data_train,target_data_train)
  outputRBF = regRBF.predict(input_data_test)
  mseRBF = mean_squared_error (outputRBF,target_data_test)

  #KNN MODEL
  reg1 = KNeighborsRegressor(n_neighbors = 1)
  reg1.fit(input_data_train,target_data_train)    
  output1 = reg1.predict(input_data_test)
  mse1 = mean_squared_error(output1,target_data_test)
      
  reg11 = KNeighborsRegressor(n_neighbors = 11)
  reg11.fit(input_data_train,target_data_train)
  output11 = reg11.predict(input_data_test)
  mse11 = mean_squared_error (output11,target_data_test)
        
  reg21 = KNeighborsRegressor(n_neighbors = 21)
  reg21.fit(input_data_train,target_data_train)
  output21 = reg21.predict(input_data_test)
  mse21 = mean_squared_error (output21,target_data_test)
      
  reg31 = KNeighborsRegressor(n_neighbors = 31)
  reg31.fit(input_data_train,target_data_train)
  output31 = reg31.predict(input_data_test)
  mse31 = mean_squared_error (output31,target_data_test)

  #DECISION TREE MODEL
  regressor = DecisionTreeRegressor(random_state = 0)
  regressor.fit(input_data_train,target_data_train)
          
  output = regressor.predict(input_data_test)
  mse = mean_squared_error(output,target_data_test)
            
  regressor1 = DecisionTreeRegressor(random_state = 1)
  regressor1.fit(input_data_train,target_data_train)
  output1 = regressor1.predict(input_data_test)
  mse1 = mean_squared_error(output1,target_data_test)

  regressor2 = DecisionTreeRegressor(random_state = 2)
  regressor2.fit(input_data_train,target_data_train)
  output2 = regressor2.predict(input_data_test)
  mse2 = mean_squared_error(output2,target_data_test)

  regressor3 = DecisionTreeRegressor(random_state = 3)
  regressor3.fit(input_data_train,target_data_train)
  output3 = regressor3.predict(input_data_test)
  mse3 = mean_squared_error(output3,target_data_test)

  if models == "SVM":
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. It is based on the hyperplane created to determine the sample used into its category.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))

    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("**Results:**")
      st.write("Linear Kernel Mse Value: ",mseLinear)
      st.write("Sigmoid Kernel Mse Value: ",mseSigmoid)
      st.write("Poly Kernel Mse Value: ",msePoly)
      st.write("RBF Kernel Mse Value: ",mseRBF)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for SVM Results")
      
      st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 5.227665318208424e+16  |   \u2713  |
      | Sigmoid       | 9.551672803634626e+16  |           |
      | Poly          | 8.548444927361835e+16  |           |
      | RBF           | 9.551673257414352e+16  |           |

      """)

    else:
      model = pickle.load(open('pop_predictSVM', 'rb'))          #load the trained model file(choose the best kernel from auto run result)
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_report():
        
        year_change = st.slider('Year Change',-0.0247,0.0384,0.025)
        net_change = st.slider('Net Change',-383840,13586631, 10000)
        migrants = st.slider('Migrants',-653249,954806, 21200)
        fert_rate = st.slider('Fertility Rate', 0.0,7.0, 1.2)
        med_age = st.slider('Med Age',0,48, 22)
        urban_pop = st.slider('Urban Population',0.0,1.0, 0.34)
        world_share = st.slider('World Share',0.0,0.1847, 0.0211)
        
        user_report_data = {
        'year_change':year_change,
        'net_change':net_change,
        'migrants':migrants,
        'fert_rate':fert_rate,
        'med_age':med_age,
        'urban_pop':urban_pop,
        'world_share':world_share}
        report_data = pd.DataFrame(user_report_data,index=[0])
        return report_data

      
      user_data = user_report()
      st.write(user_data)

      population = model.predict(user_data)
      st.write("**Population Prediction:**")
      st.write(str(np.round(population[0],2)))


  elif models == "KNN":
    st.write("K-Nearest Neighbour is based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.")
    st.write('')
    options = st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options =="None":
      st.write('')
    elif options == "Auto Run":
      st.write("**Results:**")
      st.write("Nearest Neighbors (1) Mse Value: ",mse1)
      st.write("Nearest Neighbors (11) Mse Value: ",mse11)
      st.write("Nearest Neighbors (21) Mse Value: ",mse21)
      st.write("Nearest Neighbors (31) Mse Value: ",mse31)
      st.write('   ')
      st.write('   ')
      st.write("Table for KNN Results")
      
      st.write("""

        | Nearest Neighbors | MSE                    | Best MSE  |
        | ----------------- |:----------------------:| ---------:|
        | 1                 | 8.881467327134877e+16  |\u2713     |
        | 11                | 9.056216982759574e+16  |           |
        | 21                | 9.134004259219086e+16  |           |
        | 31                | 9.191472288652403e+16  |           |

        """)
    else:

      model = pickle.load(open('pop_predictKNN', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_report():
        
        year_change = st.slider('Year Change',-0.0247,0.0384,0.025)
        net_change = st.slider('Net Change',-383840,13586631, 10000)
        migrants = st.slider('Migrants',-653249,954806, 21200)
        fert_rate = st.slider('Fertility Rate', 0.0,7.0, 1.2)
        med_age = st.slider('Med Age',0,48, 22)
        urban_pop = st.slider('Urban Population',0.0,1.0, 0.34)
        world_share = st.slider('World Share',0.0,0.1847, 0.0211)

        user_report_data = {
        'year_change':year_change,
        'net_change':net_change,
        'migrants':migrants,
        'fert_rate':fert_rate,
        'med_age':med_age,
        'urban_pop':urban_pop,
        'world_share':world_share}
        report_data = pd.DataFrame(user_report_data,index=[0])
        return report_data

      user_data = user_report()
      st.write(user_data)

      population = model.predict(user_data)
      st.write("**Population Prediction:**")
      st.write(str(np.round(population[0],2)))
  
  elif models =="Decision Tree":
    st.write("Decision Tree is an approach for supervised learning. It is a predictive models that use a set of binary rules to calculate a target value.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":

      st.write("**Results:**")
      st.write("Random State (0) Mse Value: ",mse)
      st.write("Random State (1) Mse Value: ",mse1)
      st.write("Random State (2) Mse Value: ",mse2)
      st.write("Random State (3) Mse Value: ",mse3)
      st.write('    ')
      st.write('    ')
      st.write("Table for Decision Tree Results")
    
      st.write("""

        | Random State  | MSE                   | Best MSE  |
        | ------------- |:------------------:   | ---------:|
        | 0             | 8.869330951253005e+16 | \u2713    |
        | 1             | 8.881467327134877e+16 |           |
        | 2             | 8.926844637914965e+16 |           |
        | 3             | 8.926844637914965e+16 |           |

        """)
    else:
      model = pickle.load(open('pop_predictDecision', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_report():
        year_change = st.slider('Year Change',-0.0247,0.0384,0.025)
        net_change = st.slider('Net Change',-383840,13586631, 10000)
        migrants = st.slider('Migrants',-653249,954806, 21200)
        fert_rate = st.slider('Fertility Rate', 0.0,7.0, 1.2)
        med_age = st.slider('Med Age',0,48, 22)
        urban_pop = st.slider('Urban Population',0.0,1.0, 0.34)
        world_share = st.slider('World Share',0.0,0.1847, 0.0211)

        user_report_data = {
        'year_change':year_change,
        'net_change':net_change,
        'migrants':migrants,
        'fert_rate':fert_rate,
        'med_age':med_age,
        'urban_pop':urban_pop,
        'world_share':world_share}
        report_data = pd.DataFrame(user_report_data,index=[0])
        return report_data

      user_data = user_report()
      st.write(user_data)

      population = model.predict(user_data)
      st.write("**Population Prediction:**")
      st.write(str(np.round(population[0],2)))
#End of Section Population by Country 2020

#Section World Population Prospect Dataset
elif dataset == "World Population Prospects":
  st.write("Description: This dataset is describing the World Population Prospects in 2019. The dataset is publicly available to everyone and is able to be used in predicting and estimating population. This dataset was prepared by the Population Division of the Department of Economic and Social Affairs of United Nations Secretariat. It contains over 1329 samples data with 14 variables involved. The growth rate prediction will be from the year 1953 to 2018 and have an 5 years difference apart. Variables inputs involved is TFR, NRR, CBR, Births, LEx, IMR, Deaths, CNMR, NatIncr and SRB. The GrowthRate will be the variable target for this dataset.")
  st.write("The training set is consists of 80 percent (%) samples taken that has been manually divided prior to the process of training the model.")
  st.write("The testing set is consists of 20 percent (%) samples taken that has been manually divided prior to the process of testing the model. ")
  st.write('     ')
  st.write('     ')

  data_trainWPP = pd.read_csv('WPP2019Train.csv')
  input_data_trainWPP = data_trainWPP.drop(columns= ['No','LocID','Location','MidPeriod','GrowthRate'])
  target_data_trainWPP = data_trainWPP['GrowthRate']


  data_testWPP = pd.read_csv('WPP2019Test.csv')
  input_data_testWPP = data_testWPP.drop(columns= ['No','LocID','Location','MidPeriod','GrowthRate'])
  target_data_testWPP = data_testWPP['GrowthRate']

  #SVM Model
  reglinear = LinearSVR(random_state=0, tol=1e-5,verbose = 1 ,max_iter = -1)
  reglinear.fit(input_data_trainWPP,target_data_trainWPP)
  outputlinearWPP = reglinear.predict(input_data_testWPP)
  mselinearWPP = mean_squared_error (outputlinearWPP,target_data_testWPP)

  regsigmoid = SVR (kernel ='sigmoid') 
  regsigmoid.fit(input_data_trainWPP,target_data_trainWPP)
  outputsigmoidWPP = regsigmoid.predict(input_data_testWPP)
  msesigmoidWPP = mean_squared_error (outputsigmoidWPP,target_data_testWPP)

  regpoly = SVR (kernel ='poly')
  regpoly.fit(input_data_trainWPP,target_data_trainWPP)
  outputpolyWPP = regpoly.predict(input_data_testWPP)
  msepolyWPP = mean_squared_error (outputpolyWPP,target_data_testWPP)

  regressionRBF = SVR (kernel = 'rbf')
  regressionRBF.fit(input_data_trainWPP,target_data_trainWPP)
  output_testRBFWPP = regressionRBF.predict(input_data_testWPP)
  mseRBFWPP = mean_squared_error(output_testRBFWPP,target_data_testWPP)

  #KNN Model
  reg1WPP = KNeighborsRegressor(n_neighbors = 5)
  reg1WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output1WPP = reg1WPP.predict(input_data_testWPP)
  mse1WPP = mean_squared_error(output1WPP,target_data_testWPP)    

  reg2WPP = KNeighborsRegressor(n_neighbors = 10)
  reg2WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output2WPP = reg2WPP.predict(input_data_testWPP)
  mse2WPP = mean_squared_error(output2WPP,target_data_testWPP)    

  reg3WPP = KNeighborsRegressor(n_neighbors = 20)
  reg3WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output3WPP = reg3WPP.predict(input_data_testWPP)
  mse3WPP = mean_squared_error(output3WPP,target_data_testWPP)    

  reg4WPP = KNeighborsRegressor(n_neighbors = 50)
  reg4WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output4WPP = reg4WPP.predict(input_data_testWPP)
  mse4WPP = mean_squared_error(output4WPP,target_data_testWPP)

  #DecisionTree Model
  regressorWPP = DecisionTreeRegressor(random_state = 0)
  regressorWPP.fit(input_data_trainWPP,target_data_trainWPP)
  outputWPP = regressorWPP.predict(input_data_testWPP)
  mseDTRWPP = mean_squared_error(outputWPP,target_data_testWPP)

  regressor1WPP = DecisionTreeRegressor(random_state = 1)
  regressor1WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output1WPP = regressor1WPP.predict(input_data_testWPP)
  mseDTR1WPP = mean_squared_error(output1WPP,target_data_testWPP)

  regressor2WPP = DecisionTreeRegressor(random_state = 2)
  regressor2WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output2WPP = regressor2WPP.predict(input_data_testWPP)
  mseDTR2WPP = mean_squared_error(output2WPP,target_data_testWPP)

  regressor3WPP = DecisionTreeRegressor(random_state = 3)
  regressor3WPP.fit(input_data_trainWPP,target_data_trainWPP)
  output3WPP = regressor3WPP.predict(input_data_testWPP)
  mseDTR3WPP = mean_squared_error(output3WPP,target_data_testWPP)

  if models == "SVM":
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. It is based on the hyperplane created to determine the sample used into its category.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("RBF Kernel Mse Value",mseRBFWPP)
      st.write("Sigmoid Kernel Mse Value",msesigmoidWPP)
      st.write("Poly Kernel Mse Value",msepolyWPP)
      st.write("Linear Kernel Mse Value",mselinearWPP)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for SVM Results")
      
      st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 2.79774603257329       |\u2713     |
      | Sigmoid       | 1611.548072607126      |           |
      | Poly          | 1.0348646890043431     |           |
      | RBF           | 0.8621802847261002     |           |

      """)  
    else:
      modelWPP = pickle.load(open('WPP_predictSVM', 'rb'))      #load the trained model file(choose the best kernel from auto run result)
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportWPP():
        TFR = st.slider('Total Fertility',0.98,7.55,1.2)
        NRR = st.slider('Net Reproduction Rate',0.474,3.0,2.4)
        CBR = st.slider('Crude Birth Rate',7.377,50.794,40.00)
        Births = st.slider('Births',5.696,701277.9,1230.0)
        LEx = st.slider('Life Expectancy at Birth',43.07,84.63,53.4)
        IMR = st.slider('Infant Mortality Rate',1.254,122.09,65.67)
        Deaths = st.slider('Deaths',2.544,286276.2,3002.4)
        CNMR = st.slider('Net Migration Rate',-54.746,134.414,141.78)
        NatIncr = st.slider('Rate of Natural Increase',-6.426,38.598,19.5)
        SRB = st.slider('Gender Ratio at Birth',1.009,1.17,0.5)

        user_report_dataWPP = {'TFR':TFR,
        'NRR':NRR,
        'CBR':CBR,
        'Births':Births,
        'LEx':LEx,
        'IMR':IMR,
        'Deaths':Deaths,
        'CNMR':CNMR,
        'NatIncr':NatIncr,
        'SRB':SRB}
        report_dataWPP = pd.DataFrame(user_report_dataWPP,index=[0])
        return report_dataWPP

      user_dataWPP = user_reportWPP()
      st.write(user_dataWPP)

      Births = modelWPP.predict(user_dataWPP)
      st.write("**Growth Rate:**")
      st.write(str(np.round(Births[0],2)))
      
  elif models == "KNN":
    st.write("K-Nearest Neighbour is based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("Nearest Neighbors (5) Mse Value:",mse1WPP)
      st.write("Nearest Neighbors (10) Mse Value:",mse2WPP)
      st.write("Nearest Neighbors (20) Mse Value:",mse3WPP)
      st.write("Nearest Neighbors (50) Mse Value:",mse4WPP)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for KNN Results")
      
      st.write("""

      | Nearest Neigbors | MSE                    | Best MSE  |
      | -------------    |:----------------------:| ---------:|
      | 5                | 0.3168512484690554     |   \u2713  |
      | 10               | 0.35321433439739414    |           |
      | 20               | 0.40524706396579807    |           |
      | 50               | 0.5368429481837134     |           |

      """)

    else:
      modelWPP = pickle.load(open('WPP_predictKNN', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportWPP():
        TFR = st.slider('Total Fertility',0.98,7.55,1.2)
        NRR = st.slider('Net Reproduction Rate',0.474,3.0,2.4)
        CBR = st.slider('Crude Birth Rate',7.377,50.794,40.00)
        Births = st.slider('Births',5.696,701277.9,1230.0)
        LEx = st.slider('Life Expectancy at Birth',43.07,84.63,53.4)
        IMR = st.slider('Infant Mortality Rate',1.254,122.09,65.67)
        Deaths = st.slider('Deaths',2.544,286276.2,3002.4)
        CNMR = st.slider('Net Migration Rate',-54.746,134.414,141.78)
        NatIncr = st.slider('Rate of Natural Increase',-6.426,38.598,19.5)
        SRB = st.slider('Gender Ratio at Birth',1.009,1.17,0.5)

        user_report_dataWPP = {'TFR':TFR,
        'NRR':NRR,
        'CBR':CBR,
        'Births':Births,
        'LEx':LEx,
        'IMR':IMR,
        'Deaths':Deaths,
        'CNMR':CNMR,
        'NatIncr':NatIncr,
        'SRB':SRB}
        report_dataWPP = pd.DataFrame(user_report_dataWPP,index=[0])
        return report_dataWPP

      user_dataWPP = user_reportWPP()
      st.write(user_dataWPP)

      Births = modelWPP.predict(user_dataWPP)
      st.write("**Growth Rate:**")
      st.write(str(np.round(Births[0],2)))

  elif models == "Decision Tree":
    st.write("Decision Tree is an approach for supervised learning. It is a predictive models that use a set of binary rules to calculate a target value.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("Random State (0)Mse Value: ",mseDTRWPP)
      st.write("Random State (1)Mse Value: ",mseDTR1WPP)
      st.write("Random State (2)Mse Value: ",mseDTR2WPP)
      st.write("Random State (3)Mse Value: ",mseDTR3WPP)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for Decision Tree Results")
      
      st.write("""

      | Random State| MSE                    | Best MSE  |
      | ------------|:----------------------:| ---------:|
      | 0           | 0.0659211986970684     |           |
      | 1           | 0.07181817915309446    |           |
      | 2           | 0.044878469055374605   |           |
      | 3           | 0.02503618241042345    | \u2713    |

      """)
    else:
      modelWPP = pickle.load(open('WPP_predictDecision', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportWPP():
        TFR = st.slider('Total Fertility',0.98,7.55,1.2)
        NRR = st.slider('Net Reproduction Rate',0.474,3.0,2.4)
        CBR = st.slider('Crude Birth Rate',7.377,50.794,40.00)
        Births = st.slider('Births',5.696,701277.9,1230.0)
        LEx = st.slider('Life Expectancy at Birth',43.07,84.63,53.4)
        IMR = st.slider('Infant Mortality Rate',1.254,122.09,65.67)
        Deaths = st.slider('Deaths',2.544,286276.2,3002.4)
        CNMR = st.slider('Net Migration Rate',-54.746,134.414,141.78)
        NatIncr = st.slider('Rate of Natural Increase',-6.426,38.598,19.5)
        SRB = st.slider('Gender Ratio at Birth',1.009,1.17,0.5)

        user_report_dataWPP = {'TFR':TFR,
        'NRR':NRR,
        'CBR':CBR,
        'Births':Births,
        'LEx':LEx,
        'IMR':IMR,
        'Deaths':Deaths,
        'CNMR':CNMR,
        'NatIncr':NatIncr,
        'SRB':SRB}
        report_dataWPP = pd.DataFrame(user_report_dataWPP,index=[0])
        return report_dataWPP

      user_dataWPP = user_reportWPP()
      st.write(user_dataWPP)

      Births = modelWPP.predict(user_dataWPP)
      st.write("**Growth Rate:**")
      st.write(str(np.round(Births[0],2)))
#End of Section World Population Prospect Dataset

#Section of Countries of the World Dataset        
elif dataset == "Countries of the World":
  st.write("Description: This database is obtained from Kaggle. The total sample data in this dataset is 227 with 10 variables. Variables inputs involved is Area, Population Density, Net migration, Bithrate, Deathrate and Infant Mortality. While, the variable target is the Population.")
  st.write("The training dataset contain 80 percent (%) of total sample data which is 182 of sample data.")
  st.write("The testing dataset contain 20 percent (%) of total sample data which is 45 of sample data.")
  st.write('')
  st.write('')

  data_trainCW = pd.read_csv('countries of the world training.csv')
  input_data_trainCW = data_trainCW.drop(columns=['No','Population','Country','Region','Coastline','GDP','Literacy','Phones','Arable','Crops','Other','Climate','Agriculture','Industry','Service'])
  target_data_trainCW = data_trainCW['Population']

  data_testCW = pd.read_csv('countries of the world testing.csv')
  input_data_testCW = data_testCW.drop(columns=['No','Population','Country','Region','Coastline','GDP','Literacy','Phones','Arable','Crops','Other','Climate','Agriculture','Industry','Service'])
  target_data_testCW = data_testCW['Population']

  #SVM Model
  regressionLinearCW = SVR (kernel = 'linear')
  regressionLinearCW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputLinearCW = regressionLinearCW.predict(input_data_testCW)
  mseLinearCW = mean_squared_error(predicted_outputLinearCW,target_data_testCW)

  regressionsigmoidCW = SVR(kernel = 'sigmoid')
  regressionsigmoidCW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputsigmoidCW = regressionsigmoidCW.predict(input_data_testCW)
  msesigmoidCW = mean_squared_error(predicted_outputsigmoidCW,target_data_testCW)

  regressionpolyCW= SVR(kernel = 'poly')
  regressionpolyCW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputpolyCW = regressionpolyCW.predict(input_data_testCW)
  msepolyCW = mean_squared_error(predicted_outputpolyCW,target_data_testCW)

  regressionRBFCW = SVR(kernel = 'rbf')
  regressionRBFCW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputRBFCW = regressionRBFCW.predict(input_data_testCW)
  mseRBFCW = mean_squared_error(predicted_outputRBFCW,target_data_testCW)

  #KNN Model
  knn_modelCW = KNeighborsRegressor(n_neighbors = 5)
  knn_modelCW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputKNN_CW = knn_modelCW.predict(input_data_testCW)
  mseKNN_CW = mean_squared_error(predicted_outputKNN_CW,target_data_testCW)

  knn_model2CW = KNeighborsRegressor(n_neighbors = 10)
  knn_model2CW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputKNN2_CW = knn_model2CW.predict(input_data_testCW)
  mseKNN2_CW= mean_squared_error(predicted_outputKNN2_CW,target_data_testCW)

  knn_model3CW = KNeighborsRegressor(n_neighbors = 20)
  knn_model3CW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputKNN3_CW = knn_model3CW.predict(input_data_testCW)
  mseKNN3_CW = mean_squared_error(predicted_outputKNN3_CW,target_data_testCW)

  knn_model4CW = KNeighborsRegressor(n_neighbors = 50)
  knn_model4CW.fit(input_data_trainCW,target_data_trainCW)
  predicted_outputKNN4_CW = knn_model4CW.predict(input_data_testCW)
  mseKNN4_CW = mean_squared_error(predicted_outputKNN4_CW,target_data_testCW)

  #DecisionTree Model
  DecisionTreeModelCW = DecisionTreeRegressor(random_state = 0)
  DecisionTreeModelCW.fit(input_data_trainCW,target_data_trainCW)
  DecisionTreePredictCW = DecisionTreeModelCW.predict(input_data_testCW)
  mseDecisionTreeCW = mean_squared_error(DecisionTreePredictCW,target_data_testCW)

  regressor1CW = DecisionTreeRegressor(random_state = 1)
  regressor1CW.fit(input_data_trainCW,target_data_trainCW)
  output1CW = regressor1CW.predict(input_data_testCW)
  mse1CW = mean_squared_error(output1CW,target_data_testCW)

  regressor2CW = DecisionTreeRegressor(random_state = 2)
  regressor2CW.fit(input_data_trainCW,target_data_trainCW)
  output2CW = regressor2CW.predict(input_data_testCW)
  mse2CW = mean_squared_error(output2CW,target_data_testCW)

  regressor3CW = DecisionTreeRegressor(random_state = 3)
  regressor3CW.fit(input_data_trainCW,target_data_trainCW)
  output3CW = regressor3CW.predict(input_data_testCW)
  mse3CW = mean_squared_error(output3CW,target_data_testCW)

  if models == "SVM":
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. It is based on the hyperplane created to determine the sample used into its category.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options =="None":
      st.write('')
    elif options == "Auto Run":
      st.write("Linear Kernel Mse Value",mseLinearCW)
      st.write("Sigmoid Kernel Mse Value",msesigmoidCW)
      st.write("Poly Kernel Mse Value",msepolyCW)
      st.write("RBF Kernel Mse Value",mseRBFCW)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for SVM Results")
      
      st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 2042599121813602.5     | \u2713    |
      | Sigmoid       | 3330558548729037.0     |           |
      | Poly          | 3317891415020798.5     |           |
      | RBF           | 3330559124655074.0     |           |

      """)
    else:
      modelCW = pickle.load(open('CW_predictSVM', 'rb'))            #load the trained model file(choose the best kernel from auto run result) 
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportCW():
        Area = st.slider('Area sq.mi',0.0,29.74,7.55)
        Pop_density = st.slider('Population Density',0.0,50.73, 20.68)
        Net_migration = st.slider('Net Migration',0.0,191.19, 60.12)
        Infant_mortality = st.slider('Infant Mortality',-20.99,23.06, 7.6)
        Birthrate = st.slider('Birth Rate',0.0,16271.5,300.79)
        Deathrate = st.slider('Death Rate', 2,17075200,10000)
        

        user_report_dataCW = {'Area':Area,
        'Pop_density':Pop_density,
        'Net_migration':Net_migration,
        'Infant_mortality':Infant_mortality,
        'Birthrate':Birthrate,
        'Deathrate':Deathrate}
        
        report_dataCW = pd.DataFrame(user_report_dataCW,index=[0])
        return report_dataCW

      user_dataCW = user_reportCW()
      st.write(user_dataCW)

      pop = modelCW.predict(user_dataCW)
      st.write("**Population Prediction:**")
      st.write(str(np.round(pop[0],2)))

  elif models == "KNN":
    st.write("K-Nearest Neighbour is based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options =="None":
      st.write('')
    elif options == "Auto Run":
      st.write("Nearest Neighbors (5) Mse Value:",mseKNN_CW)
      st.write("Nearest Neighbors (10) Mse Value:",mseKNN2_CW)
      st.write("Nearest Neighbors (20) Mse Value:",mseKNN3_CW)
      st.write("Nearest Neighbors (50) Mse Value:",mseKNN4_CW)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for KNN Results")
      
      st.write("""

      | Nearest Neighbors | MSE                    | Best MSE  |
      | ----------------- |:----------------------:| ---------:|
      | 5                 | 5275041969406962.0     |           |
      | 10                | 1220926351647041.2     |           |
      | 20                | 1027771164661988.5     | \u2713    |
      | 50                | 1699057071932716.5     |           |

      """)
    else:
      modelCW = pickle.load(open('CW_predictKNN', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportCW():
        Area = st.slider('Area sq.mi',0.0,29.74,7.55)
        Pop_density = st.slider('Population Density',0.0,50.73, 20.68)
        Net_migration = st.slider('Net Migration',0.0,191.19, 60.12)
        Infant_mortality = st.slider('Infant Mortality',-20.99,23.06, 7.6)
        Birthrate = st.slider('Birth Rate',0.0,16271.5,300.79)
        Deathrate = st.slider('Death Rate', 2,17075200,10000)
        

        user_report_dataCW = {'Area':Area,
        'Pop_density':Pop_density,
        'Net_migration':Net_migration,
        'Infant_mortality':Infant_mortality,
        'Birthrate':Birthrate,
        'Deathrate':Deathrate}
        
        report_dataCW = pd.DataFrame(user_report_dataCW,index=[0])
        return report_dataCW

      user_dataCW = user_reportCW()
      st.write(user_dataCW)

      pop = modelCW.predict(user_dataCW)
      st.write("**Population Prediction:**")
      st.write(str(np.round(pop[0],2)))

  elif models == "Decision Tree":
    st.write("Decision Tree is an approach for supervised learning. It is a predictive models that use a set of binary rules to calculate a target value.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options =="Auto Run":
      st.write("Random State (0)Mse Value: ",mseDecisionTreeCW)
      st.write("Random State (1)Mse Value: ",mse1CW)
      st.write("Random State (2)Mse Value: ",mse2CW)
      st.write("Random State (3)Mse Value: ",mse3CW)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for Decision Tree Results")
      
      st.write("""

      | Random State  | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | 0             | 2025710564692667.5     |           |
      | 1             | 1980534313183443.5     | \u2713    |
      | 2             | 2242180305523907.5     |           |
      | 3             | 2184473050961561.5     |           |

      """)
    else:
      modelCW = pickle.load(open('CW_predictDecision', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportCW():
        Area = st.slider('Area sq.mi',0.0,29.74,7.55)
        Pop_density = st.slider('Population Density',0.0,50.73, 20.68)
        Net_migration = st.slider('Net Migration',0.0,191.19, 60.12)
        Infant_mortality = st.slider('Infant Mortality',-20.99,23.06, 7.6)
        Birthrate = st.slider('Birth Rate',0.0,16271.5,300.79)
        Deathrate = st.slider('Death Rate', 2,17075200,10000)
        

        user_report_dataCW = {'Area':Area,
        'Pop_density':Pop_density,
        'Net_migration':Net_migration,
        'Infant_mortality':Infant_mortality,
        'Birthrate':Birthrate,
        'Deathrate':Deathrate}
        
        report_dataCW = pd.DataFrame(user_report_dataCW,index=[0])
        return report_dataCW

      user_dataCW = user_reportCW()
      st.write(user_dataCW)

      pop = modelCW.predict(user_dataCW)
      st.write("**Population Prediction:**")
      st.write(str(np.round(pop[0],2)))
#End of Section Countries of the World Dataset

#Section of Fertility Dataset
elif dataset == "Fertility Data Set":
  st.write("The data that is used for this study is gathered from a secondary source. It is retrieved from Machine Learning Repository website. This data is from 2010 volunteer research done by World Health Organization (WHO). Sperm concentrations are related to socio-demographic data, environmental factors, health status, and life habits in UC Irvine machine learning repository which consists of 100 instances and 10 attributes with the class stating normal or altered. Variables inputs involved is Childish diseases, Accident or serious trauma, Surgical intervention, High fevers in the last year, Smoking habit, Frequency of alcohol consumption, Number of hours spent sitting per day. While, the variable target is the Age.")
  st.write("The training set is consists of 80 percent (%) samples taken that has been manually divided prior to the process of training the model.")
  st.write("The testing set is consists of 20 percent (%) samples taken that has been manually divided prior to the process of testing the model. ")
  st.write('')
  st.write('')

  data_trainFT = pd.read_csv('fertility randomly split _training.csv')
  le = LabelEncoder()
  data_trainFT['Childish diseases'] = le.fit_transform(data_trainFT['Childish diseases'])
  data_trainFT['Smoking habit'] = le.fit_transform(data_trainFT['Smoking habit'])
  data_trainFT['Accident or serious trauma'] = le.fit_transform(data_trainFT['Accident or serious trauma'])
  data_trainFT['Surgical intervention'] = le.fit_transform(data_trainFT['Surgical intervention'])
  data_trainFT['High fevers in the last year'] = le.fit_transform(data_trainFT['High fevers in the last year'])
  data_trainFT['Frequency of alcohol consumption'] = le.fit_transform(data_trainFT['Frequency of alcohol consumption'])
  data_trainFT['Number of hours spent sitting per day'] = le.fit_transform(data_trainFT['Number of hours spent sitting per day'])
  data_trainFT['Diagnosis'] = le.fit_transform(data_trainFT['Diagnosis'])

  input_data_trainFT = data_trainFT.drop(columns = ['Season','Age','Diagnosis'])
  target_data_trainFT = data_trainFT['Age']


  data_testFT = pd.read_csv('fertility randomly split _testing.csv')
  le = LabelEncoder()
  data_testFT['Childish diseases'] = le.fit_transform(data_testFT['Childish diseases'])
  data_testFT['Smoking habit'] = le.fit_transform(data_testFT['Smoking habit'])
  data_testFT['Accident or serious trauma'] = le.fit_transform(data_testFT['Accident or serious trauma'])
  data_testFT['Surgical intervention'] = le.fit_transform(data_testFT['Surgical intervention'])
  data_testFT['High fevers in the last year'] = le.fit_transform(data_testFT['High fevers in the last year'])
  data_testFT['Frequency of alcohol consumption'] = le.fit_transform(data_testFT['Frequency of alcohol consumption'])
  data_testFT['Number of hours spent sitting per day'] = le.fit_transform(data_testFT['Number of hours spent sitting per day'])
  data_testFT['Diagnosis'] = le.fit_transform(data_testFT['Diagnosis'])

  input_data_testFT = data_testFT.drop(columns = ['Season', 'Age', 'Diagnosis'])
  target_data_testFT = data_testFT['Age']

  #SVM Model
  regressionFT = SVR(kernel = 'linear')
  regressionFT.fit(input_data_trainFT, target_data_trainFT)
  predicted_outputFT = regressionFT.predict(input_data_testFT)
  mseFT = mean_squared_error (predicted_outputFT, target_data_testFT)

  regressionsigmoidFT = SVR(kernel = 'sigmoid')
  regressionsigmoidFT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputsigmoidFT = regressionsigmoidFT.predict(input_data_testFT)
  msesigmoidFT = mean_squared_error(predicted_outputsigmoidFT,target_data_testFT)

  regressionpolyFT = SVR(kernel = 'poly')
  regressionpolyFT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputpolyFT = regressionpolyFT.predict(input_data_testFT)
  msepolyFT = mean_squared_error(predicted_outputpolyFT,target_data_testFT)

  regressionRBFFT = SVR(kernel = 'rbf')
  regressionRBFFT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputRBFFT = regressionRBFFT.predict(input_data_testFT)
  mseRBFFT = mean_squared_error(predicted_outputRBFFT,target_data_testFT)

  #KNN Model
  knn_modelFT = KNeighborsRegressor (n_neighbors = 1)
  knn_modelFT.fit(input_data_trainFT, target_data_trainFT)
  knnout_regressorFT = knn_modelFT.predict(input_data_testFT)
  mseKNN1_FT = mean_squared_error (knnout_regressorFT, target_data_testFT)

  knn_model2FT = KNeighborsRegressor(n_neighbors = 11)
  knn_model2FT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputKNN2_FT = knn_model2FT.predict(input_data_testFT)
  mseKNN2_FT = mean_squared_error(predicted_outputKNN2_FT,target_data_testFT)

  knn_model3FT = KNeighborsRegressor(n_neighbors = 21)
  knn_model3FT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputKNN3_FT = knn_model3FT.predict(input_data_testFT)
  mseKNN3_FT = mean_squared_error(predicted_outputKNN3_FT,target_data_testFT)

  knn_model4FT = KNeighborsRegressor(n_neighbors = 31)
  knn_model4FT.fit(input_data_trainFT,target_data_trainFT)
  predicted_outputKNN4_FT = knn_model4FT.predict(input_data_testFT)
  mseKNN4_FT = mean_squared_error(predicted_outputKNN4_FT,target_data_testFT)

  #Decision Tree Model
  regressor0FT = DecisionTreeRegressor(random_state = 0)
  regressor0FT.fit(input_data_trainFT,target_data_trainFT)
  output0FT = regressor0FT.predict(input_data_testFT)
  mse0FT = mean_squared_error(output0FT,target_data_testFT)

  regressor1FT = DecisionTreeRegressor(random_state = 1)
  regressor1FT.fit(input_data_trainFT,target_data_trainFT)
  output1FT = regressor1FT.predict(input_data_testFT)
  mse1FT = mean_squared_error(output1FT,target_data_testFT)

  regressor2FT = DecisionTreeRegressor(random_state = 2)
  regressor2FT.fit(input_data_trainFT,target_data_trainFT)
  output2FT = regressor2FT.predict(input_data_testFT)
  mse2FT = mean_squared_error(output2FT,target_data_testFT)

  regressor3FT = DecisionTreeRegressor(random_state = 3)
  regressor3FT.fit(input_data_trainFT,target_data_trainFT)
  output3FT = regressor3FT.predict(input_data_testFT)
  mse3FT = mean_squared_error(output3FT,target_data_testFT)
  #End of Section Fertility Dataset

  if models == "SVM":
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. It is based on the hyperplane created to determine the sample used into its category.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("Linear Kernel Mse Value",mseFT)
      st.write("Sigmoid Kernel Mse Value",msesigmoidFT)
      st.write("Poly Kernel Mse Value",msepolyFT)
      st.write("RBF Kernel Mse Value",mseRBFFT)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for SVM Results")
      
      st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 6.516821192285063      |           |
      | Sigmoid       | 16.488059687501558     |           |
      | Poly          | 5.7983308664287225     |           |
      | RBF           | 6.253728579244045      |   \u2713  |

      """)
      
    else:
      modelFT = pickle.load(open('FT_predictSVM', 'rb'))                #load the trained model file(choose the best kernel from auto run result) 
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportFT():
        Childish_diseases = st.slider('Childish diseases',0,1,0)
        Accident_seriousTrauma = st.slider('Accident or serious trauma',0,1, 0)
        Surgical_intervention = st.slider('Surgical intervention',0,1,0)
        High_fever = st.slider('High fever in the last year',0,2,1)
        Frequency_alcohol = st.slider('Frequency of alcohol consumption',1,4,2)
        Smoking_habit = st.slider('Smoking habit',0,2,1)
        Number_sitting = st.slider('Number of hours spent sitting per day',0,342,10)
        
        
        user_report_dataFT = {'Childish_diseases':Childish_diseases,
        'Accident_seriousTrauma':Accident_seriousTrauma,
        'Surgical_intervention':Surgical_intervention,
        'High_fever':High_fever,
        'Frequency_alcohol':Frequency_alcohol,
        'Smoking_habit':Smoking_habit,
        'Number_sitting':Number_sitting}
        
        
        report_dataFT = pd.DataFrame(user_report_dataFT,index=[0])
        return report_dataFT

      user_dataFT = user_reportFT()
      st.write(user_dataFT)

      fert = modelFT.predict(user_dataFT)
      st.write("**Predicted Age:**")
      st.write(str(np.round(fert[0],2)))

  elif models == "KNN":
    st.write("K-Nearest Neighbour is based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("**Results:**")
      st.write("Nearest Neighbors (1) Mse Value: ",mseKNN1_FT)
      st.write("Nearest Neighbors (11) Mse Value: ",mseKNN2_FT)
      st.write("Nearest Neighbors (21) Mse Value: ",mseKNN3_FT)
      st.write("Nearest Neighbors (31) Mse Value: ",mseKNN4_FT)
      st.write('   ')
      st.write('   ')
      st.write("Table for KNN Results")
      
      st.write("""

        | Nearest Neighbors | MSE                    | Best MSE  |
        | ----------------- |:----------------------:| ---------:|
        | 1                 | 6.75                   |           |
        | 11                | 4.947933884297521      |           |
        | 21                | 4.794557823129251      | \u2713    |
        | 31                | 5.119875130072841      |           |

        """)
    else:
      modelFT = pickle.load(open('FT_predictKNN', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportFT():
        Childish_diseases = st.slider('Childish diseases',0,1,0)
        Accident_seriousTrauma = st.slider('Accident or serious trauma',0,1, 0)
        Surgical_intervention = st.slider('Surgical intervention',0,1,0)
        High_fever = st.slider('High fever in the last year',0,2,1)
        Frequency_alcohol = st.slider('Frequency of alcohol consumption',1,4,2)
        Smoking_habit = st.slider('Smoking habit',0,2,1)
        Number_sitting = st.slider('Number of hours spent sitting per day',0,342,10)
        
        
        user_report_dataFT = {'Childish_diseases':Childish_diseases,
        'Accident_seriousTrauma':Accident_seriousTrauma,
        'Surgical_intervention':Surgical_intervention,
        'High_fever':High_fever,
        'Frequency_alcohol':Frequency_alcohol,
        'Smoking_habit':Smoking_habit,
        'Number_sitting':Number_sitting}
        
        
        report_dataFT = pd.DataFrame(user_report_dataFT,index=[0])
        return report_dataFT

      user_dataFT = user_reportFT()
      st.write(user_dataFT)

      fert = modelFT.predict(user_dataFT)
      st.write("** Predicted Age:**")
      st.write(str(np.round(fert[0],2)))
    
  elif models == "Decision Tree":
    st.write("Decision Tree is an approach for supervised learning. It is a predictive models that use a set of binary rules to calculate a target value.")
    st.write('')
    options= st.selectbox ("How Would You Like to Test the Model?", ("None","Auto Run","User Input"))
    if options == "None":
      st.write('')
    elif options == "Auto Run":
      st.write("Random State (0)Mse Value: ",mse0FT)
      st.write("Random State (1)Mse Value: ",mse1FT)
      st.write("Random State (2)Mse Value: ",mse2FT)
      st.write("Random State (3)Mse Value: ",mse3FT)
      st.write('                    ')
      st.write('                    ')

      st.write("Table for Decision Tree Results")
      
      st.write("""

      | Random State  | MSE        | Best MSE  |
      | ------------- |:----------:| ---------:|
      | 0             | 6.9125     | \u2713    |
      | 1             | 7.825      |           |
      | 2             | 7.775      |           |
      | 3             | 8.4125     |           |

      """)
    else:
      modelFT = pickle.load(open('FT_predictDecision', 'rb'))
      st.header('User Input Parameters')
      st.write("Move the slider to input a prediction value")
      def user_reportFT():
        Childish_diseases = st.slider('Childish diseases',0,1,0)
        Accident_seriousTrauma = st.slider('Accident or serious trauma',0,1, 0)
        Surgical_intervention = st.slider('Surgical intervention',0,1,0)
        High_fever = st.slider('High fever in the last year',0,2,1)
        Frequency_alcohol = st.slider('Frequency of alcohol consumption',1,4,2)
        Smoking_habit = st.slider('Smoking habit',0,2,1)
        Number_sitting = st.slider('Number of hours spent sitting per day',0,342,10)
        
        
        user_report_dataFT = {'Childish_diseases':Childish_diseases,
        'Accident_seriousTrauma':Accident_seriousTrauma,
        'Surgical_intervention':Surgical_intervention,
        'High_fever':High_fever,
        'Frequency_alcohol':Frequency_alcohol,
        'Smoking_habit':Smoking_habit,
        'Number_sitting':Number_sitting}
        
        report_dataFT = pd.DataFrame(user_report_dataFT,index=[0])
        return report_dataFT

      user_dataFT = user_reportFT()
      st.write(user_dataFT)

      fert = modelFT.predict(user_dataFT)
      st.write("**Predicted Age:**")
      st.write(str(np.round(fert[0],2)))



#Result Comparison Section

if result =="None":
  st.write('')
elif result == "Dataset 1":
  st.write('')
  st.write('')
  st.write("**Result for Population By Country-2020 dataset. Below is the result summary of all tested models:**")

  st.write("Table 1: SVM Results")
  print("/n")
    
  st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 5.227665318208424e+16  |   \u2713  |
      | Sigmoid       | 9.551672803634626e+16  |           |
      | Poly          | 8.548444927361835e+16  |           |
      | RBF           | 9.551673257414352e+16  |           |

      """)
  st.write("Based on the results in Table 1, the results shows that the kernel Linear produce the best MSE value compared to other kernels. Thus, the Linear kernel is chosen to be compared with the KNN and Decision Tree model.")

  st.write('                    ')
  st.write('                    ')
  st.write("Table 2: KNN Results")
    
  st.write("""

        | Nearest Neighbors | MSE                    | Best MSE  |
        | ----------------- |:----------------------:| ---------:|
        | 1                 | 8.881467327134877e+16  |\u2713     |
        | 11                | 9.056216982759574e+16  |           |
        | 21                | 9.134004259219086e+16  |           |
        | 31                | 9.191472288652403e+16  |           |

        """)
  st.write("Based on the results in Table 2, the results shows that number of  nearest neighbors 1 produce the best MSE value compared to other nearest neighbors. Thus, the value of 1 is chosen to be compared with the SVM and Decision Tree model.")
  st.write('                              ')
  st.write('                              ')
  st.write("Table 3: Decision Tree Results")
    
  st.write("""

        | Random State  | MSE                   | Best MSE  |
        | ------------- |:------------------:   | ---------:|
        | 0             | 8.869330951253005e+16 | \u2713    |
        | 1             | 8.881467327134877e+16 |           |
        | 2             | 8.926844637914965e+16 |           |
        | 3             | 8.926844637914965e+16 |           |

        """)
  st.write("Based on the results in Table 3, the results shows that number of random state 0 produce the best MSE value compared to other random state. Thus, the random state 0 is chosen to be compared with the KNN and SVM model.")


  st.write('                    ')
  st.write('                    ')
  st.write("Table 4: Comparison Results of All Models")
    
  st.write("""

    | Model         | MSE                   | Best MSE  |
    | ------------- |:------------------:   | ---------:|
    | SVM           | 5.227665318208424e+16 | \u2713    |
    | KNN           | 8.881467327134877e+16 |           |
    | Decision Tree | 8.869330951253005e+16 |           |
    
    """)
  st.write("Table 4 shows the comparison of the chosen results of each model in one table. Based on this table, the results shows that the SVM produced the best MSE value compared to other models.")

elif result =="Dataset 2":
  st.write('')
  st.write('')
  st.write("**Result for World Population Prospect Dataset. Below is the result summary of all tested models:**")

  st.write("Table 1: SVM Results")
  
  st.write("""

    | Kernel        | MSE                    | Best MSE  |
    | ------------- |:----------------------:| ---------:|
    | Linear        | 2.79774603257329       |           |
    | Sigmoid       | 1611.548072607126      |           |
    | Poly          | 1.0348646890043431     |           |
    | RBF           | 0.8621802847261002     |  \u2713   |

    """)
  st.write("Based on the results in Table 1, the results shows that the kernel RBF produce the best MSE value compared to other kernels. Thus, the Linear kernel is chosen to be compared with the KNN and Decision Tree model.")

  st.write('                    ')
  st.write('                    ')
  st.write("Table 2: KNN Results")
    
  st.write("""

    | Nearest Neigbors | MSE                    | Best MSE  |
    | -------------    |:----------------------:| ---------:|
    | 5                | 0.3168512484690554     |  \u2713   |
    | 10               | 0.35321433439739414    |           |
    | 20               | 0.40524706396579807    |           |
    | 50               | 0.5368429481837134     |           |

    """)

  st.write("Based on the results in Table 2, the results shows that number of nearest neighbors 5 produce the best MSE value compared to other nearest neighbors. Thus, the value of 20 is chosen to be compared with the SVM and Decision Tree model.")
  st.write('                              ')
  st.write('                              ')
  st.write("Table 3: Decision Tree Results")
    
  st.write("""

    | Random State| MSE                    | Best MSE  |
    | ------------|:----------------------:| ---------:|
    | 0           | 0.0659211986970684     |           |
    | 1           | 0.07181817915309446    |           |
    | 2           | 0.044878469055374605   |           |
    | 3           | 0.02503618241042345    | \u2713    |

     """)
  st.write("Based on the results in Table 3, the results shows that number of random state 3 produce the best MSE value compared to other random state. Thus, the random state 3 is chosen to be compared with the KNN and SVM model.")

  st.write('')
  st.write('')
  st.write("Table 4: Comparison Results of All Models")
    
  st.write("""

    | Model         | MSE                   | Best MSE  |
    | ------------- |:------------------:   | ---------:|
    | SVM           | 0.8621802847261002    |           |
    | KNN           | 0.3168512484690554    |           |
    | Decision Tree | 0.02503618241042345   | \u2713    |
    
    """)
  st.write("Table 4 shows the comparison of the chosen results of each model in one table. Based on this table, the results shows that the KNN and Decision Tree produced the best MSE value compared to SVM.")

elif result == "Dataset 3":
  st.write('')
  st.write('')
  st.write("**Result for Countries of the World Dataset. Below is the result summary of all tested models:**")

  st.write("Table 1: SVM Results")
  print("/n")
    
  st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 2042599121813602.5     | \u2713    |
      | Sigmoid       | 3330558548729037.0     |           |
      | Poly          | 3317891415020798.5     |           |
      | RBF           | 3330559124655074.0     |           |

      """)
  st.write("Based on the results in Table 1, the results shows that the kernel Linear produce the best MSE value compared to other kernels. Thus, the Linear kernel is chosen to be compared with the KNN and Decision Tree model.")

  st.write('                    ')
  st.write('                    ')
  st.write("Table 2: KNN Results")
    
  st.write("""

      | Nearest Neighbors | MSE                    | Best MSE  |
      | ----------------- |:----------------------:| ---------:|
      | 5                 | 5275041969406962.0     |           |
      | 10                | 1220926351647041.2     |           |
      | 20                | 1027771164661988.5     | \u2713    |
      | 50                | 1699057071932716.5     |           |

      """)
  st.write("Based on the results in Table 2, the results shows that number of nearest neighbors 20 produce the best MSE value compared to other nearest neighbors. Thus, the value of 20 is chosen to be compared with the SVM and Decision Tree model.")
  st.write('                              ')
  st.write('                              ')
  st.write("Table 3: Decision Tree Results")
    
  st.write("""

      | Random State  | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | 0             | 2025710564692667.5     |           |
      | 1             | 1980534313183443.5     | \u2713    |
      | 2             | 2242180305523907.5     |           |
      | 3             | 2184473050961561.5     |           |

      """)
  st.write("Based on the results in Table 3, the results shows that number of random state 1 produce the best MSE value compared to other random state. Thus, the random state 1 is chosen to be compared with the KNN and SVM model.")


  st.write('                    ')
  st.write('                    ')
  st.write("Table 4: Comparison Results of All Models")
    
  st.write("""

    | Model         | MSE                | Best MSE  |
    | ------------- |:------------------:| ---------:|
    | SVM           | 2042599121813602.5 |           |
    | KNN           | 1027771164661988.5 | \u2713    |
    | Decision Tree | 1980534313183443.5 |           |
    
    """)
  st.write("Table 4 shows the comparison of the chosen results of each model in one table. Based on this table, the results shows that the KNN produced the best MSE value compared to other models.")

else:
  st.write('')
  st.write('')
  st.write("**Result for Fertility Dataset. Below is the result summary of all tested models:**")

  st.write("Table 1: SVM Results")
  print("/n")
    
  st.write("""

      | Kernel        | MSE                    | Best MSE  |
      | ------------- |:----------------------:| ---------:|
      | Linear        | 6.516821192285063      |           |
      | Sigmoid       | 16.488059687501558     |           |
      | Poly          | 5.7983308664287225     |           |
      | RBF           | 6.253728579244045      |   \u2713  |

      """)
  st.write("Based on the results in Table 1, the results shows that the kernel RBF produce the best MSE value compared to other kernels. Thus, the RBF kernel is chosen to be compared with the KNN and Decision Tree model.")

  st.write('                    ')
  st.write('                    ')
  st.write("Table 2: KNN Results")
    
  st.write("""

        | Nearest Neighbors | MSE                    | Best MSE  |
        | ----------------- |:----------------------:| ---------:|
        | 1                 | 6.75                   |           |
        | 11                | 4.947933884297521      |           |
        | 21                | 4.794557823129251      | \u2713    |
        | 31                | 5.119875130072841      |           |

        """)
  st.write("Based on the results in Table 2, the results shows that number of nearest neighbors 21 produce the best MSE value compared to other nearest neighbors. Thus, the value of 21 is chosen to be compared with the SVM and Decision Tree model.")
  st.write('                              ')
  st.write('                              ')
  st.write("Table 3: Decision Tree Results")
    
  st.write("""

      | Random State  | MSE        | Best MSE  |
      | ------------- |:----------:| ---------:|
      | 0             | 6.9125     | \u2713    |
      | 1             | 7.825      |           |
      | 2             | 7.775      |           |
      | 3             | 8.4125     |           |

      """)
  st.write("Based on the results in Table 3, the results shows that number of random state 0 produce the best MSE value compared to other random state. Thus, the random state 0 is chosen to be compared with the KNN and SVM model.")


  st.write('                    ')
  st.write('                    ')
  st.write("Table 4: Comparison Results of All Models")
    
  st.write("""

    | Model         | MSE                | Best MSE  |
    | ------------- |:------------------:| ---------:|
    | SVM           | 6.253728579244045  |           |
    | KNN           | 4.794557823129251  | \u2713    |
    | Decision Tree | 6.9125             |           |
    
    """)
  st.write("Table 4 shows the comparison of the chosen results of each model in one table. Based on this table, the results shows that the KNN produced the best MSE value compared to other models.")
  # End of Result Comparison Section