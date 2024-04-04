# import numpy as np
# import pickle
# import streamlit as st
# from PIL import Image
# import cv2

# # # create a file uploader with Streamlit
# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # # if an image is uploaded
# # if uploaded_file is not None:
# #     # read the image
# #     image = Image.open(uploaded_file)

# #     # display the image
# #     st.image(image, caption="Uploaded image", use_column_width=True)




# # loading the saved model
# loaded_model = pickle.load(open('F:/projectstreamlit/trained_data.sav', 'rb'))



# # creating a function for Prediction

# def detect(input_data):
    

#     image_a = 'F:/projectstreamlit/Lumpy_Skin_19.png'
    
#     img = cv2.imread(image_a)
#     img = cv2.resize(img,(150,150))
#     img = np.reshape(img, (1, 150, 150, 3))
#     classes = (loaded_model.predict(img) > 0.5).astype("int32")
#     if(classes==1):
#         return "Normal skin"
#     else:
#         return "Lumpy Skin"
  
    
  
# def main():
    
    
#     # giving a title
#     st.title('Animal Disease Detection')
    
#     image = None
#     # create a file uploader with Streamlit
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     # if an image is uploaded
#     if uploaded_file is not None:
#         # read the image
#         image = Image.open(uploaded_file)

#     # display the image
#     st.image(image, caption="Uploaded image", use_column_width=True)


#     # creating a button for Prediction
#     if st.button('Test Result'):
#         result = detect(image)
#     st.success(result)
    
# if __name__ == '__main__':
#     main()




import numpy as np
import pickle
import streamlit as st
from PIL import Image
import cv2

# loading the saved model
loaded_model = pickle.load(open('F:/projectstreamlit/trained_data.sav', 'rb'))

# creating a function for Prediction
def detect(image):
    if image is None:
        return "Please upload an image"
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, (1, 150, 150, 3))
    classes = (loaded_model.predict(img) > 0.5).astype("int32")
    if classes == 1:
        return "Normal skin"
    else:
        return "Lumpy Skin"
def main():
    # giving a title
    st.title('Lumpy Skin Disease Detection')
    
    # create a file uploader with Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # if an image is uploaded
    if uploaded_file is not None:
        # read the image
        image = Image.open(uploaded_file)

        # display the image
        st.image(image, caption="Uploaded image", use_column_width=True)

        # creating a button for Prediction
        if st.button('Test Result'):
            result = detect(image)
            if result is not None:
                st.success("Result: {}".format(result))
    
if __name__ == '__main__':
    main()
