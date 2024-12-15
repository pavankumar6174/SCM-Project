import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score,mean_squared_error,classification_report,confusion_matrix
from SBMBOT import queries_page
from streamlit_option_menu import option_menu


# Load your dataset
df = pd.read_csv('C:\\Users\\10139519\\NTTDATA-PROJ-PRACTICE\\SCM-Demo\\Data\\cleaned_data.csv')
data1=pd.read_csv('C:\\Users\\10139519\\NTTDATA-PROJ-PRACTICE\\SCM-Demo\\Data\\Dataset.csv',encoding='ISO-8859-1')
data=pd.concat([df,data1['Product Name']],axis=1)

# Load mappings from the uploaded JSON files
with open('../Data/dictionary.json', 'r') as file:
    mappings = json.load(file)

with open('../Data/dictionary1.json', 'r') as file:
    new_mappings = json.load(file)

 # Load mappings and model data
with open('../Data/dictionary2.json', 'r') as dict1:
    depart = json.load(dict1)

product_encoding=data['Product Name'].value_counts().to_dict()
data['Product Name']=data['Product Name'].map(product_encoding)


# Extract mappings dynamically
customer_segment_mapping = mappings["Customer Segment"]
customer_state_mapping = mappings["Customer State"]
order_state_mapping = mappings["Order State"]
order_status_mapping = mappings["Order Status"]
department_name_mapping = mappings["Department Name"]
shipping_mode_mapping = mappings["Shipping Mode"]

# Initialize session state for tabs if not already set
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Home"

# Function to render content based on selected tab
def render_content(selected_tab):
    if selected_tab == "Home":
        st.header("Supply Chain Management")
        st.write("Welcome to the Home page of the Supply Chain Management system.")
    elif selected_tab == "DashBoard":
        st.header("Dashboard")
        # Creating custom CSS for the tab design
        st.markdown(
            """
            <style>
            .stTabs [role="tab"] {
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 10px 20px;
                margin: 0 5px;
                cursor: pointer;
                background-color: #f0f0f0;
                transition: background-color 0.3s ease;
            }
            .stTabs [role="tab"]:hover {
                background-color: #e0e0e0;
            }
            .stTabs [role="tab"]:selected {
                background-color: #d0d0d0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Creating tabs
        tab1, tab2 = st.tabs(["Single Product", "Multi Product"])

        with tab1:
            st.header('Single Product Analysis')


            # Calculate Days for shipping (real) based on Shipping Mode
            shipping_mode_avg = df[['Days for shipping (real)', 'Shipping Mode']].groupby('Shipping Mode').mean()

            # Calculate Shipping Mode Rank
            shipping_mode_rank = shipping_mode_avg['Days for shipping (real)'].rank(method='dense').astype(int)

            # Map the rank back to the dataset
            df['Shipping Mode Rank'] = df['Shipping Mode'].map(shipping_mode_rank)

            # Calculate Order delay
            df['Order delay'] = df['Days for shipment (scheduled)'] - df['Days for shipping (real)']

            # Prepare the data
            X = df.drop(columns=['Late_delivery_risk'])
            y = df['Late_delivery_risk']

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

            # Train the Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Calculate model accuracy
            y_pred = model.predict(X_test)

            ran=RandomForestClassifier()
            ran.fit(X_train,y_train)

            st.subheader('Feature Importance')

            sing_report_df=pd.DataFrame(
                {
                    'columns':X.columns,
                    'values':ran.feature_importances_*100
                }
            )

            st.bar_chart(data=sing_report_df,x='columns',y='values')

            st.subheader('Classification Report')

            st.code(classification_report(y_test,y_pred),language='text')





        with tab2:
            st.header('Multi Product Analyis')
            st.subheader('Correlation Between Features')

            # Compute the correlation matrix
            corr_matrix = data.corr()

            # Create the heatmap figure using Plotly Graph Objects
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,   # Correlation values for heatmap
                x=corr_matrix.columns,  # Column names for x-axis
                y=corr_matrix.columns,  # Column names for y-axis
                colorscale='Greys',      # Color scale for the heatmap
                zmin=-1,                # Minimum correlation value
                zmax=1,                 # Maximum correlation value
                colorbar=dict(title="Correlation", ticksuffix=" "), # Color bar customization
                text=corr_matrix.round(2).values,  # Display the correlation values on the heatmap
                hovertemplate='%{y} - %{x}: %{text}<extra></extra>',  # Hover template to show correlation values
            ))

            # Update layout for better presentation
            fig.update_layout(
                title="Correlation Matrix Heatmap",  # Title of the heatmap
                title_x=0.5,  # Center title
                xaxis=dict(
                    title="Features",  # X-axis label
                    tickangle=45,  # Rotate tick labels for better readability
                ),
                yaxis=dict(
                    title="Features",  # Y-axis label
                    tickangle=-45,  # Rotate tick labels for better readability
                ),
                width=800,  # Set the width of the plot
                height=600,  # Set the height of the plot
                template="plotly_dark",  # Use a dark theme for the plot
            )

            # Display the heatmap in the Streamlit app
            st.plotly_chart(fig)

            logistic_Regression=LogisticRegression()
            RandomForest=RandomForestClassifier()
            DecisionTree=DecisionTreeClassifier()
            xgboostt=xgb.XGBClassifier()

            algorithm=[logistic_Regression,RandomForest,DecisionTree,xgboostt]

            XX=data.drop(columns=['Late_delivery_risk'])
            yy=data['Late_delivery_risk']

            x_train,x_test,y_trn,y_tst=train_test_split(XX,yy,test_size=0.25,random_state=4235)


            model,accuracy1,mse,classificationReport,confusionmatirx=[],[],[],[],[]

            for al in algorithm:
                al.fit(x_train,y_trn)
                model.append(al)
                y_pred=al.predict(x_test)
                accuracy1.append(accuracy_score(y_tst,y_pred))
                mse.append(mean_squared_error(y_tst,y_pred))
                classificationReport.append(classification_report(y_tst,y_pred))
                confusionmatirx.append(confusion_matrix(y_tst,y_pred))
            
            report_df=pd.DataFrame({
                'models':['Logistic','Random Forest','Decision Tree','XGBOOST'],
                'Accuracy':accuracy1,
                'mse':mse,
                'classificationReport':classificationReport,
                'confusionmatirx':confusionmatirx
            }
            )

            st.subheader('Accuracy of Multiple ALgorithms ')
            
            st.bar_chart(data=report_df,x='models',y='Accuracy',x_label='ALgorithms',y_label='Accuracy')

            # st.subheader('Mean Square Error for Multiple Products')

            # st.bar_chart(data=report_df,x='models',y='mse',x_label='ALgorithms',y_label='MSE')

            st.subheader('Classification Report for Logistic Regression')

            st.code(classificationReport[0],language='text')
            

    elif selected_tab == "Model Prediction":
        st.header("Model Prediction")
        st.write("Here you can interact with the predictive models.")

        # Creating custom CSS for the tab design
        st.markdown(
            """
            <style>
            .stTabs [role="tab"] {
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 10px 20px;
                margin: 0 5px;
                cursor: pointer;
                background-color: #f0f0f0;
                transition: background-color 0.3s ease;
            }
            .stTabs [role="tab"]:hover {
                background-color: #e0e0e0;
            }
            .stTabs [role="tab"]:selected {
                background-color: #d0d0d0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Creating tabs
        tab1, tab2 = st.tabs(["Single Product", "Multi Product"])

        with tab1:
            # Calculate Days for shipping (real) based on Shipping Mode
            shipping_mode_avg = df[['Days for shipping (real)', 'Shipping Mode']].groupby('Shipping Mode').mean()

            # Calculate Shipping Mode Rank
            shipping_mode_rank = shipping_mode_avg['Days for shipping (real)'].rank(method='dense').astype(int)

            # Map the rank back to the dataset
            df['Shipping Mode Rank'] = df['Shipping Mode'].map(shipping_mode_rank)

            # Calculate Order delay
            df['Order delay'] = df['Days for shipment (scheduled)'] - df['Days for shipping (real)']

            # Prepare the data
            X = df.drop(columns=['Late_delivery_risk'])
            y = df['Late_delivery_risk']

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

            # Train the Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Calculate model accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

            # Define the prediction function
            def predict_late_delivery(input_data):
                prediction = model.predict(input_data)[0]
                prediction_label = "Late Delivery Risk" if prediction == 1 else "No Late Delivery Risk"
                return prediction_label
            
            col11,col12=st.columns(2)
            with col11:
                customer_id=st.selectbox('Product ID : ',list(data1['Product Category Id'].unique()))
            with col12:
                category_name=st.selectbox('Category Name',list(data1['Category Name'].unique()))
            # Row 1: Customer Segment and Customer State
            col1, col2 = st.columns(2)
            with col1:
                customer_segment = st.selectbox("Customer Segment", ["Select"] + list(customer_segment_mapping.keys()))
            with col2:
                customer_state = st.selectbox("Customer State", ["Select"] + list(customer_state_mapping.keys()))

            # Row 2: Order State and Order City
            col3, col4 = st.columns(2)
            with col3:
                order_state = st.selectbox("Order State", ["Select"] + list(order_state_mapping.keys()))
            with col4:
                order_city = st.selectbox("Order City", ["Select"] + (list(new_mappings[order_state]['cities'].keys()) if order_state != "Select" else []))

            # Row 3: Order Status and Order Item Quantity
            col5, col6 = st.columns(2)
            with col5:
                order_status = st.selectbox("Order Status", ["Select"] + list(order_status_mapping.keys()))
            with col6:
                order_item_quantity = st.number_input("Order Item Quantity", min_value=0, step=1, value=0, key="item_qty")

            # Row 4: Department Name and Shipping Mode
            col7, col8 = st.columns(2)
            with col7:
                department_name = st.selectbox("Department Name", ["Select"] + list(department_name_mapping.keys()))
            with col8:
                shipping_mode = st.selectbox("Shipping Mode", ["Select"] + list(shipping_mode_mapping.keys()))

            # Row 5: Days for shipment (scheduled) and Product Price
            col9, col10 = st.columns(2)
            with col9:
                days_for_shipment_scheduled = st.number_input("Days for Shipment (Scheduled)", min_value=0, step=1, value=0)
            with col10:
                product_price = st.number_input("Product Price", min_value=0.0, step=0.01, value=0.0)

            # Button to trigger prediction
            if st.button("Predict"):
                # Validate Inputs
                if (
                    customer_segment == "Select" or
                    customer_state == "Select" or
                    order_state == "Select" or
                    order_city == "Select" or
                    order_status == "Select" or
                    department_name == "Select" or
                    shipping_mode == "Select" or
                    days_for_shipment_scheduled == 0 or
                    product_price == 0.0
                ):
                    st.error("Please fill out all input fields.")
                else:
                    # Prepare Input Data (Mapping and Handling)
                    customer_segment_value = customer_segment_mapping[customer_segment]
                    customer_state_value = customer_state_mapping[customer_state]
                    order_city_value = new_mappings[order_state]['cities'][order_city]
                    order_state_value = order_state_mapping[order_state]
                    order_status_value = order_status_mapping[order_status]
                    department_name_value = department_name_mapping[department_name]
                    shipping_mode_value = shipping_mode_mapping[shipping_mode]
                    
                    # Calculate the real days for shipping (this is a simplified logic, you can adjust accordingly)
                    days_for_shipping_real = shipping_mode_avg.loc[shipping_mode_value, 'Days for shipping (real)']
                    
                    # Calculate Order Delay (based on scheduled and real shipping times)
                    order_delay = days_for_shipment_scheduled - days_for_shipping_real
                    
                    # Shipping Mode Rank (this is also simplified)
                    shipping_mode_rank_value = shipping_mode_rank[shipping_mode_value]

                    # Prepare input data for prediction
                    input_data = pd.DataFrame([[
                        days_for_shipping_real,
                        days_for_shipment_scheduled,
                        customer_segment_value,
                        customer_state_value,
                        order_city_value,
                        order_state_value,
                        order_status_value,
                        order_item_quantity,
                        department_name_value,
                        product_price,
                        shipping_mode_value,
                        order_delay,
                        shipping_mode_rank_value
                    ]], columns=[
                        'Days for shipping (real)', 
                        'Days for shipment (scheduled)', 
                        'Customer Segment', 
                        'Customer State', 
                        'Order City', 
                        'Order State',
                        'Order Status',
                        'Order Item Quantity', 
                        'Department Name',
                        'Product Price', 
                        'Shipping Mode', 
                        'Order delay', 
                        'Shipping Mode Rank'
                    ])

                    # Perform Prediction (the actual prediction function)
                    prediction = predict_late_delivery(input_data)
                    
                    # Display the prediction result
                    if prediction == 'No Late Delivery Risk':
                        st.success(f"Prediction: {prediction}")
                    else:
                        st.error(f"Prediction: {prediction}")
                    st.info(f"Model Accuracy: {accuracy:.2f}%")

        with tab2:
            data.drop_duplicates(inplace=True)
            # st.dataframe(data)
            XX=data.drop(columns=['Late_delivery_risk'])
            yy=data['Late_delivery_risk']

            
            x_train,x_test,y_trn,y_tst=train_test_split(XX,yy,test_size=0.25,random_state=4235)

            logistic=LogisticRegression()

            logistic.fit(x_train,y_trn)


            # Row 1: Customer Segment and Customer State
            col1, col2 = st.columns(2)
            with col1:
                customer_segment1 = st.selectbox("Customer Segment for Multiple Products", list(customer_segment_mapping.keys()))
            with col2:
                customer_state1 = st.selectbox("Customer State for Multiple Products", list(customer_state_mapping.keys()))

            # Allow selecting multiple departments and products
            col3, col4 = st.columns(2)
            with col3:
                depart_names = st.multiselect("Select Department Names for Multiple Products", list(depart.keys()))
            with col4:
                product_names = []
                for depart_name in depart_names:
                    product_names += list(depart.get(depart_name, {}).keys())  # Safe access using get
                product_names = list(set(product_names))  # Ensure unique product names
                selected_products = st.multiselect("Select Products", product_names)

            # Store 'Days for Scheduled' for each selected product
            days_for_scheduled = {}

            # Initialize flag for checking if all fields are filled
            all_inputs_filled = True
            predictions = {}

            # Loop through each selected department and product to create inputs
            for depart_name in depart_names:
                for product_name in selected_products:

                    # Check if the product exists under the selected department
                    if product_name not in depart.get(depart_name, {}):
                        continue  # Skip this product and continue with the next one

                    # Create product-specific dropdown with Days for Scheduled input
                    with st.expander(f"Configure details for {product_name} in {depart_name}"):

                        # Input fields for Order State and City
                        order_state1 = st.selectbox(f"Order State for {product_name}", list(depart[depart_name][product_name].keys()), key=f"order_state_{depart_name}_{product_name}")
                        order_city1 = st.selectbox(f"Order City for {product_name}", list(depart[depart_name][product_name][order_state1]), key=f"order_city_{depart_name}_{product_name}")

                        # Fields for Price and Quantity
                        col7, col8 = st.columns(2)
                        with col7:
                            order_price = st.number_input(f"Product Price for {product_name}", min_value=0, step=1, key=f"price_{depart_name}_{product_name}")
                        with col8:
                            order_quantity = st.number_input(f"Quantity Ordered for {product_name}", min_value=1, step=1, key=f"quantity_{depart_name}_{product_name}")

                        # Shipping Mode and Order Status
                        col9, col10 = st.columns(2)
                        with col9:
                            shipping_mode_order = st.selectbox(f"Shipping Mode for {product_name}", list(shipping_mode_mapping.keys()), key=f"shipping_mode_{depart_name}_{product_name}")
                        with col10:
                            order_status_int = st.selectbox(f"Order Status for {product_name}", list(order_status_mapping.keys()), key=f"status_{depart_name}_{product_name}")

                        # Input for "Days for Scheduled"
                        days_for_scheduled[(depart_name, product_name)] = st.number_input(
                            f"Days for Scheduled for {product_name}",
                            min_value=0, step=1, key=f"days_scheduled_{depart_name}_{product_name}"
                        )

                        # Check if all required fields are filled for the product
                        if not (order_price and order_quantity and shipping_mode_order and order_status_int and days_for_scheduled[(depart_name, product_name)] > 0):
                            all_inputs_filled = False

                    # Store the prediction button for later display
                    if all_inputs_filled:
                        # Calculate the real shipping time
                        days_for_shipping_real = shipping_mode_avg.loc[shipping_mode_mapping[shipping_mode_order], 'Days for shipping (real)']
                        
                        # Calculate order delay
                        order_delay =days_for_shipping_real- days_for_scheduled[(depart_name, product_name)]  
                        
                        # Shipping Mode Rank
                        shipping_mode_rank_value = shipping_mode_rank[shipping_mode_mapping[shipping_mode_order]]

                        # Prepare input list for the prediction
                        input_list = [
                            days_for_shipping_real,
                            days_for_scheduled[(depart_name, product_name)],
                            customer_segment_mapping[customer_segment1],
                            customer_state_mapping[customer_state1],
                            new_mappings[order_state1]['cities'][order_city1],
                            order_state_mapping[order_state1],
                            order_status_mapping[order_status_int],
                            order_quantity,
                            department_name_mapping[depart_name],
                            order_price,
                            shipping_mode_mapping[shipping_mode_order],
                            order_delay,
                            shipping_mode_rank_value,
                            product_encoding[product_name]
                        ]
                        
                        # Get prediction
                        prediction = logistic.predict([input_list])

                        # print(prediction)
                        
                        # Store the prediction result
                        prd = lambda x: 'No Late Delivery Risk' if x <= 0 else 'Risk On Late Delivery'
                        predictions[(depart_name, product_name)] = prd(prediction)
                    else:
                        predictions[(depart_name, product_name)] = None  # Mark as None if fields aren't filled

            # Display the "Predict Late Delivery Risk for All Products" button at the bottom
            if st.button("Predict Late Delivery Risk for All Products"):
                # Display all predictions for selected products and departments
                for (depart_name, product_name), pred in predictions.items():
                    if pred:  # Only display predictions that are available
                        if pred == "Risk On Late Delivery":
                            st.error(f"Prediction for Late Delivery Risk of **{product_name}** in **{depart_name}**: **{pred}**")
                        else:
                            st.success(f"Prediction for Late Delivery Risk of **{product_name}** in **{depart_name}**: **{pred}**")


    elif selected_tab == "SCMBOT":
        st.header("SCMBOT")
        st.write("This is the SCMBOT (Supply Chain Management Bot).")
        queries_page()

# # Sidebar for selecting pages (sync with tabs)
tabs = ["Home", "DashBoard", "Model Prediction", "SCMBOT"]
selected_sidebar_tab = st.sidebar.radio("Choose a page:", tabs)
# Sidebar navigation using streamlit-option-menu
# with st.sidebar:
#     tab = option_menu(
#         "Navigation",  # Title of the menu
#         ["Home", "DashBoard", "Model Prediction", "Jarvis"],  # Options
#         icons=["house", "bar-chart", "robot", "question-circle"],  # Icons for each tab
#         menu_icon="cast",  # Icon for the menu
#         default_index=0,  # Default active tab
#         orientation="vertical"  # Tabs in a vertical orientation
#     )

# Sync the selection between sidebar and custom tabs
if selected_sidebar_tab != st.session_state.selected_tab:
    st.session_state.selected_tab = selected_sidebar_tab



# Render content based on selected tab
render_content(st.session_state.selected_tab)
