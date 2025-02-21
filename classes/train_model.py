
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import streamlit as st
class TrainModel():
    def __init__(self, data, target, features):
        
        # encode the categorical variables 
        self.features= features
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        self.X_train=X_train
        self.y_train=y_train
        logit_model = sm.Logit(y_train, X_train)  # Initialize the model
        result = logit_model.fit()  # Fit the model

        # Extract coefficients and put them in a DataFrame
        coef = result.params
        coef_df = coef.reset_index()
        # rename feature const to intercept
        coef_df['index'].replace('const', 'Intercept', inplace=True)
        coef_df.columns = ['Parameter', 'Value']
        self.coef_df = coef_df
        self.p_values = result.pvalues
        st.expander("Summary of the trained model", expanded=False).write(result.summary())

    def selectFeatures(self):
        # Backward elimination
        significance_level= 0.05
        current_features= self.features
        while len(current_features) >1 :
            X_current =self.X_train[current_features]
            model = sm.Logit(self.y_train, X_current).fit(disp=False)

            # Get the predictor with the highest p-value
            p_values= model.pvalues
            highest_p_value= p_values.max()
            feature_to_remove=p_values.idxmax()
            st.write("P-values of the features",p_values)

            if highest_p_value >significance_level:
                st.write(f"Removing feature {feature_to_remove} with p-value {highest_p_value}")
                current_features.remove(feature_to_remove)  # Remove the predictor
            else:
                print("All p-values are below the significance level")
                print("Final features",current_features)
                print("Final p-values",model.pvalues) 
                break  # Stop if all p-values are below the significance level
        # Final model with selected features
        
        X_selected = self.X_train[current_features]
        X_selected = sm.add_constant(X_selected)
        final_model = sm.Logit(self.y_train, X_selected).fit()
        # st.write("Final Model on Selected Features",final_model.summary())
        coef = final_model.params
        coef_df = coef.reset_index()
        # rename feature const to intercept
        coef_df['index'].replace('const', 'Intercept', inplace=True)
        coef_df.columns = ['Parameter', 'Value']
        self.coef_df = coef_df
