import streamlit.components.v1 as components
import lime.lime_tabular
import streamlit
import mahotas
import pickle
import pandas
import numpy
import cv2

from sklearn.model_selection import train_test_split

def get_model():
    return pickle.load(open('./etc/xgb_model.dat', 'rb'))

def convert_byteio_image(string):
    array = numpy.frombuffer(string, numpy.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

streamlit.markdown("<h1 style='text-align: center; color: black;'>Web application for the diagnosis of COVID-19 using X-ray images</h1>", unsafe_allow_html=True)

streamlit.sidebar.title('Settings')

uploaded_image = streamlit.sidebar.file_uploader("Choose an image", type = 'png', accept_multiple_files = False)

model = get_model()

if uploaded_image is not None:

    bytes_data = uploaded_image.getvalue()

    image = convert_byteio_image(bytes_data)

    if (image.shape != (256, 256)):    
        image = cv2.resize(image, (256, 256))

    features = mahotas.features.haralick(image, compute_14th_feature = True, return_mean = True).reshape(1, 14)

    pred = model.predict(features)   

    probs = model.predict_proba(features)  

    streamlit.markdown("<h3 style='text-align: center; color: black;'>Image</h3>", unsafe_allow_html=True)

    col1, col2, col3 = streamlit.columns([0.2, 5, 0.2])

    col2.image(image, use_column_width=True)

    pred_output = "Patient affected by COVID-19 with {:.2%} certainty".format(probs[0][1]) if pred[0] == 1 else "Patient within the normal range with {:.2%} certainty".format(probs[0][0]) 

    streamlit.markdown("<h4 style='text-align: center; color: black;'>" + pred_output + "</h4>", unsafe_allow_html=True)

if streamlit.sidebar.button("Explain prediction"):

    streamlit.markdown("<h3 style='text-align: center; color: black;'>Interpretation</h3>", unsafe_allow_html=True)

    with streamlit.spinner('Calculating...'):        

        df = pandas.read_csv('./etc/features.csv', delimiter = ';')

        X = df.drop('Label', axis = 1)
        y = df['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = X.columns, class_names = ['Normal', 'COVID-19'], 
                                                                feature_selection = 'lasso_path', discretize_continuous = True)
        
        exp = explainer.explain_instance(features.reshape(14,), model.predict_proba, num_features = 14)
        
        components.html(exp.as_html(predict_proba = False), height = 800)

